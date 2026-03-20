import argparse
import csv
import json
import random
import re
from pathlib import Path
import numpy as np
import scipy.ndimage as ndimage
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset

## this file is to vectorize(make token) the image

DEFAULT_ORGAN_DIR = Path("data/PKG - CT-ORG/CT-ORG/OrganSegmentations")
DEFAULT_SAVE_DIR = Path("output/experiments/organ_seg")

HU_CLIP = (-1000.0, 400.0)
DEFAULT_ORGAN_NAME_MAP = {
    1: "liver",
    2: "bladder",
    3: "lung",
    4: "kidney",
    5: "bone",
    6: "brain",
}



def resolve_run_paths(save_dir, run_tag):
    run_root = Path(save_dir) / run_tag
    train_dir = run_root / "train"
    token_dir = run_root / "tokens"
    pred_dir = run_root / "pred" / "organ_seg_predictions"
    model_dir = run_root / "model"
    return {
        "run_root": run_root,
        "train_dir": train_dir,
        "token_dir": token_dir,
        "pred_dir": pred_dir,
        "model_dir": model_dir,
        "training_summary_csv": train_dir / "organ_seg_training_summary.csv",
        "token_csv": token_dir / "organ_imaging_tokens.csv",
        "model_path": model_dir / "organ_seg_unet.pt",
    }


def ensure_output_dirs(run_paths):
    run_paths["run_root"].mkdir(parents=True, exist_ok=True)
    run_paths["train_dir"].mkdir(parents=True, exist_ok=True)
    run_paths["token_dir"].mkdir(parents=True, exist_ok=True)
    run_paths["pred_dir"].mkdir(parents=True, exist_ok=True)
    run_paths["model_dir"].mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_case_id(path):
    m = re.search(r"(\d+)", path.name)
    if not m:
        return ""
    return m.group(1)


def find_case_pairs(organ_dir):
    pairs = []
    volume_files = sorted(organ_dir.glob("volume-*.nii.gz"))
    for vol in volume_files:
        case_id = parse_case_id(vol)
        if not case_id:
            continue
        lab = organ_dir / f"labels-{case_id}.nii.gz"
        if lab.exists():
            pairs.append((case_id, vol, lab))
    pairs.sort(key=lambda x: int(x[0]))
    return pairs


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_nifti_array(path):
    arr = nib.load(str(path)).get_fdata()
    return np.asarray(arr, dtype=np.float32)


def normalize_ct(volume):
    clipped = np.clip(volume, HU_CLIP[0], HU_CLIP[1])
    return ((clipped - HU_CLIP[0]) / (HU_CLIP[1] - HU_CLIP[0])).astype(np.float32)


def align_label_to_volume(label, volume_shape):
    if tuple(label.shape) == tuple(volume_shape):
        return label.astype(np.int64)
    zoom = (
        volume_shape[0] / label.shape[0],
        volume_shape[1] / label.shape[1],
        volume_shape[2] / label.shape[2],
    )
    aligned = ndimage.zoom(label, zoom=zoom, order=0)
    aligned = np.rint(aligned).astype(np.int64)
    if tuple(aligned.shape) != tuple(volume_shape):
        fixed = np.zeros(volume_shape, dtype=np.int64)
        z = min(volume_shape[0], aligned.shape[0])
        y = min(volume_shape[1], aligned.shape[1])
        x = min(volume_shape[2], aligned.shape[2])
        fixed[:z, :y, :x] = aligned[:z, :y, :x]
        aligned = fixed
    return aligned


def select_slice_indices(indices, max_count):
    if not indices:
        return []
    if max_count is None or max_count <= 0 or len(indices) <= max_count:
        return [int(x) for x in indices]
    out = []
    n = len(indices)
    m = max_count
    for i in range(m):
        pos = int(round(i * (n - 1) / max(m - 1, 1)))
        out.append(int(indices[pos]))
    return sorted(set(out))


def build_context_stack(volume, z, num_context_slices, slice_stride):
    depth = int(volume.shape[0])
    channels = []
    for offset in range(-num_context_slices, num_context_slices + 1):
        idx = z + offset * slice_stride
        idx = max(0, min(depth - 1, idx))
        channels.append(np.asarray(volume[idx], dtype=np.float32))
    return np.stack(channels, axis=0).astype(np.float32)


def resize_multichannel_and_label(image_chw, label_hw, image_size):
    target_h, target_w = int(image_size), int(image_size)
    zoom_h = target_h / image_chw.shape[1]
    zoom_w = target_w / image_chw.shape[2]
    image_resized = ndimage.zoom(image_chw, zoom=(1.0, zoom_h, zoom_w), order=1)
    label_resized = ndimage.zoom(label_hw, zoom=(zoom_h, zoom_w), order=0)
    image_resized = np.asarray(image_resized, dtype=np.float32)
    label_resized = np.asarray(np.rint(label_resized), dtype=np.int64)
    return image_resized, label_resized


class CTORG25DSliceDataset(Dataset):
    def __init__(self, pairs, max_cases, image_size, num_context_slices, slice_stride, max_slices_per_case):
        self.samples = []
        self.case_to_indices = {}
        self.representative_indices = []
        selected = pairs[:max_cases]
        total = len(selected)
        for i, (case_id, vol_path, lab_path) in enumerate(selected, start=1):
            print(f"[load] {i}/{total} case_id={case_id}", flush=True)
            volume = normalize_ct(load_nifti_array(vol_path))
            label = align_label_to_volume(load_nifti_array(lab_path), volume.shape)

            foreground_per_slice = (label > 0).reshape(label.shape[0], -1).sum(axis=1)
            fg_indices = np.where(foreground_per_slice > 0)[0].tolist()
            if not fg_indices:
                fg_indices = [int(label.shape[0] // 2)]
            sampled_indices = select_slice_indices(fg_indices, max_slices_per_case)
            rep_z = int(np.argmax(foreground_per_slice)) if int(foreground_per_slice.max()) > 0 else int(label.shape[0] // 2)
            if rep_z not in sampled_indices:
                sampled_indices.append(rep_z)
            sampled_indices = sorted(set(int(z) for z in sampled_indices))

            case_sample_indices = []
            for z in sampled_indices:
                image_chw = build_context_stack(volume, z, num_context_slices, slice_stride)
                mask_hw = np.asarray(label[z], dtype=np.int64)
                image_chw, mask_hw = resize_multichannel_and_label(image_chw, mask_hw, image_size)
                sample = {
                    "case_id": case_id,
                    "volume_path": str(vol_path),
                    "label_path": str(lab_path),
                    "slice_idx": int(z),
                    "image": image_chw,
                    "mask": mask_hw,
                    "is_representative": int(z) == int(rep_z),
                }
                case_sample_indices.append(len(self.samples))
                self.samples.append(sample)

            self.case_to_indices[case_id] = case_sample_indices
            rep_indices = [idx for idx in case_sample_indices if self.samples[idx]["is_representative"]]
            if rep_indices:
                self.representative_indices.append(rep_indices[0])
            elif case_sample_indices:
                self.representative_indices.append(case_sample_indices[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = torch.from_numpy(item["image"]).float()
        mask = torch.from_numpy(item["mask"]).long()
        return {
            "image": image,
            "mask": mask,
            "case_id": item["case_id"],
            "slice_idx": item["slice_idx"],
            "volume_path": item["volume_path"],
            "label_path": item["label_path"],
        }


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels, token_dim):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = DoubleConv(c2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)
        self.token_head = nn.Conv2d(c3, token_dim, kernel_size=1)

    def forward(self, x, return_token_map=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        logits = self.head(d1)

        if return_token_map:
            token_map = self.token_head(b)
            return logits, token_map
        return logits, None


def infer_num_classes(dataset, num_classes_arg):
    if num_classes_arg and num_classes_arg > 1:
        return int(num_classes_arg)
    max_label = 0
    for sample in dataset.samples:
        max_label = max(max_label, int(sample["mask"].max()))
    return int(max_label + 1)


def build_organ_name_map(num_classes):
    out = {}
    for organ_id in range(1, num_classes):
        out[organ_id] = DEFAULT_ORGAN_NAME_MAP.get(organ_id, f"organ_{organ_id}")
    return out


def split_train_val_case_ids(case_ids, val_ratio, seed):
    if len(case_ids) <= 1 or val_ratio <= 0:
        return list(case_ids), []
    case_ids = list(case_ids)
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    val_n = int(round(len(case_ids) * val_ratio))
    val_n = max(1, val_n)
    val_n = min(val_n, len(case_ids) - 1)
    val_ids = sorted(case_ids[:val_n])
    train_ids = sorted(case_ids[val_n:])
    return train_ids, val_ids


def case_ids_to_sample_indices(dataset, case_ids):
    indices = []
    for cid in case_ids:
        indices.extend(dataset.case_to_indices.get(cid, []))
    return sorted(indices)


def multiclass_dice(logits, target, num_classes):
    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
        dice_vals = []
        eps = 1e-6
        for cls in range(1, num_classes):
            pred_c = (pred == cls).float()
            tgt_c = (target == cls).float()
            inter = (pred_c * tgt_c).sum()
            denom = pred_c.sum() + tgt_c.sum()
            if float(denom) == 0.0:
                continue
            dice_vals.append(float((2.0 * inter + eps) / (denom + eps)))
        if not dice_vals:
            return 0.0
        return float(sum(dice_vals) / len(dice_vals))


def evaluate_model(model, loader, device, num_classes, criterion):
    if loader is None or len(loader) == 0:
        return {"loss": None, "dice": None}

    model.eval()
    loss_sum = 0.0
    dice_sum = 0.0
    step_count = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            logits, _ = model(image, return_token_map=False)
            loss = criterion(logits, target)
            loss_sum += float(loss.item())
            dice_sum += multiclass_dice(logits, target, num_classes)
            step_count += 1
    return {
        "loss": loss_sum / max(step_count, 1),
        "dice": dice_sum / max(step_count, 1),
    }


def is_improved(metric_name, current_value, best_value, min_delta):
    if current_value is None:
        return False
    if best_value is None:
        return True
    if "loss" in metric_name:
        return current_value < (best_value - min_delta)
    return current_value > (best_value + min_delta)


def snapshot_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_one_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    num_classes,
    early_stop_metric,
    early_stop_patience,
    early_stop_min_delta,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logs = []

    best_state = None
    best_epoch = None
    best_metric_value = None
    no_improve_count = 0
    stopped_early = False

    if val_loader is None and early_stop_metric.startswith("val_"):
        print("[warn] val loader is empty, fallback early_stop_metric=train_loss", flush=True)
        early_stop_metric = "train_loss"

    for epoch in range(1, epochs + 1):
        model.train()
        print(f"[train] epoch {epoch}/{epochs} start", flush=True)
        train_loss_sum = 0.0
        train_dice_sum = 0.0
        train_steps = 0
        total_steps = len(train_loader)
        for step_idx, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            logits, _ = model(image, return_token_map=False)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_dice_sum += multiclass_dice(logits, target, num_classes)
            train_steps += 1
            if step_idx == 1 or step_idx == total_steps or step_idx % 10 == 0:
                print(
                    f"[train] epoch {epoch}/{epochs} step {step_idx}/{total_steps} "
                    f"loss={float(loss.item()):.4f}",
                    flush=True,
                )

        train_loss = train_loss_sum / max(train_steps, 1)
        train_dice = train_dice_sum / max(train_steps, 1)
        val_metrics = evaluate_model(model, val_loader, device, num_classes, criterion)
        val_loss = val_metrics["loss"]
        val_dice = val_metrics["dice"]

        current_monitor = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
        }.get(early_stop_metric)

        improved = is_improved(
            metric_name=early_stop_metric,
            current_value=current_monitor,
            best_value=best_metric_value,
            min_delta=early_stop_min_delta,
        )
        if improved:
            best_metric_value = current_monitor
            best_epoch = epoch
            best_state = snapshot_state_dict(model)
            no_improve_count = 0
        else:
            no_improve_count += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss if val_loss is not None else "",
            "val_dice": val_dice if val_dice is not None else "",
            "monitor_metric": early_stop_metric,
            "monitor_value": current_monitor if current_monitor is not None else "",
            "is_best_epoch": 1 if improved else 0,
            "no_improve_count": no_improve_count,
        }
        logs.append(row)

        val_loss_text = "NA" if val_loss is None else f"{val_loss:.4f}"
        val_dice_text = "NA" if val_dice is None else f"{val_dice:.4f}"
        monitor_text = "NA" if current_monitor is None else f"{current_monitor:.4f}"
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"val_loss={val_loss_text} val_dice={val_dice_text} "
            f"monitor({early_stop_metric})={monitor_text} "
            f"best_epoch={best_epoch}",
            flush=True,
        )

        if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
            stopped_early = True
            print(
                f"[early_stop] stop at epoch={epoch}, "
                f"best_epoch={best_epoch}, "
                f"patience={early_stop_patience}",
                flush=True,
            )
            break

    return {
        "logs": logs,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_metric": best_metric_value,
        "monitor_metric": early_stop_metric,
        "stopped_early": stopped_early,
    }


def l2_normalize_1d(vec):
    norm = torch.linalg.norm(vec, ord=2)
    if float(norm) <= 0.0:
        return vec
    return vec / norm


def extract_and_save_tokens(model, dataset, device, num_classes, pred_dir, organ_name_map):
    token_rows = []
    model.eval()

    rep_indices = list(dataset.representative_indices)
    with torch.no_grad():
        total = len(rep_indices)
        for i, sample_idx in enumerate(rep_indices, start=1):
            sample = dataset.samples[sample_idx]
            case_id = sample["case_id"]
            slice_idx = int(sample["slice_idx"])
            print(f"[token] {i}/{total} case_id={case_id} slice={slice_idx}", flush=True)
            image_np = sample["image"]
            mask_np = sample["mask"]

            image = torch.from_numpy(image_np).unsqueeze(0).to(device)
            logits, token_map = model(image, return_token_map=True)
            pred = torch.argmax(logits, dim=1)[0]

            np.savez_compressed(
                pred_dir / f"case_{case_id}_slice_{slice_idx}.npz",
                pred_mask=pred.cpu().numpy().astype(np.int16),
                gt_mask=mask_np.astype(np.int16),
                slice_idx=np.asarray([slice_idx], dtype=np.int16),
            )

            token_map_up = F.interpolate(
                token_map,
                size=pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0]

            gt_mask_tensor = torch.from_numpy(mask_np).to(device)
            for organ_id in range(1, num_classes):
                region_pred = pred == organ_id
                if int(region_pred.sum().item()) > 0:
                    region = region_pred
                    source = "pred"
                else:
                    region_gt = gt_mask_tensor == organ_id
                    if int(region_gt.sum().item()) == 0:
                        continue
                    region = region_gt
                    source = "gt_fallback"

                token = token_map_up[:, region].mean(dim=1)
                token = l2_normalize_1d(token)
                token_rows.append(
                    {
                        "case_id": case_id,
                        "slice_idx": slice_idx,
                        "organ_id": int(organ_id),
                        "organ_name": organ_name_map.get(int(organ_id), f"organ_{organ_id}"),
                        "mask_source": source,
                        "voxel_count": int(region.sum().item()),
                        "token_json": json.dumps([float(x) for x in token.cpu().tolist()]),
                    }
                )
    return token_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="CT-ORG organ segmentation + organ imaging tokens (2.5D).",
        allow_abbrev=False,
    )
    parser.add_argument("--organ-dir", type=str, default=str(DEFAULT_ORGAN_DIR))
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--run-tag", type=str, default="search_base24")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="0 means use all CT-ORG cases; >0 means use first N cases.",
    )
    parser.add_argument(
        "--max-slices-per-case",
        type=int,
        default=0,
        help="0 means use all foreground slices per case; >0 means cap to N slices.",
    )
    parser.add_argument("--num-context-slices", type=int, default=1, help="1 means 3-channel 2.5D.")
    parser.add_argument("--slice-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--early-stop-metric", type=str, default="val_dice", choices=["val_dice", "val_loss", "train_loss"])
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    run_tag = args.run_tag.strip()
    if not run_tag:
        raise SystemExit("--run-tag must not be empty")
    if args.max_cases < 0:
        raise SystemExit("--max-cases must be >= 0 (0 means all cases)")
    if args.max_slices_per_case < 0:
        raise SystemExit("--max-slices-per-case must be >= 0 (0 means all slices)")
    if args.num_context_slices < 0:
        raise SystemExit("--num-context-slices must be >= 0")
    if args.slice_stride <= 0:
        raise SystemExit("--slice-stride must be >= 1")
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise SystemExit("--val-ratio must be in [0, 1)")

    set_seed(args.seed)
    run_paths = resolve_run_paths(args.save_dir, run_tag)
    ensure_output_dirs(run_paths)

    organ_dir = Path(args.organ_dir)
    print(f"[start] organ_dir={organ_dir}", flush=True)
    print(f"[start] save_dir={args.save_dir} run_tag={run_tag}", flush=True)
    print(
        f"[start] max_cases={args.max_cases} max_slices_per_case={args.max_slices_per_case} "
        f"full_cases={1 if args.max_cases == 0 else 0} "
        f"full_slices={1 if args.max_slices_per_case == 0 else 0} "
        f"context={args.num_context_slices} stride={args.slice_stride} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} "
        f"val_ratio={args.val_ratio} "
        f"early_stop=({args.early_stop_metric},patience={args.early_stop_patience},min_delta={args.early_stop_min_delta})",
        flush=True,
    )
    print(
        f"[deps] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
        f"numpy={np.__version__}",
        flush=True,
    )

    pairs = find_case_pairs(organ_dir)
    if not pairs:
        raise SystemExit(f"no volume-label pair found under: {organ_dir}")
    print(f"[start] found_pairs={len(pairs)}", flush=True)

    selected_cases = len(pairs) if args.max_cases == 0 else min(args.max_cases, len(pairs))
    selected_max_slices = None if args.max_slices_per_case == 0 else int(args.max_slices_per_case)

    print("[stage] build dataset", flush=True)
    dataset = CTORG25DSliceDataset(
        pairs=pairs,
        max_cases=selected_cases,
        image_size=args.image_size,
        num_context_slices=args.num_context_slices,
        slice_stride=args.slice_stride,
        max_slices_per_case=selected_max_slices,
    )
    if len(dataset) == 0:
        raise SystemExit("dataset is empty after filtering")

    num_classes = infer_num_classes(dataset, args.num_classes)
    organ_name_map = build_organ_name_map(num_classes)
    in_channels = args.num_context_slices * 2 + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case_ids = sorted(dataset.case_to_indices.keys(), key=lambda x: int(x))
    train_case_ids, val_case_ids = split_train_val_case_ids(case_ids, args.val_ratio, args.seed)
    train_indices = case_ids_to_sample_indices(dataset, train_case_ids)
    val_indices = case_ids_to_sample_indices(dataset, val_case_ids)

    print(f"device: {device}")
    print(f"cases_selected: {len(case_ids)}")
    print(f"slices_selected: {len(dataset)}")
    print(f"train_cases: {len(train_case_ids)} train_slices: {len(train_indices)}")
    print(f"val_cases: {len(val_case_ids)} val_slices: {len(val_indices)}")
    print(f"num_classes: {num_classes} in_channels: {in_channels}")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None

    print("[stage] build model", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    model = SmallUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=args.base_channels,
        token_dim=args.token_dim,
    ).to(device)

    print("[stage] training", flush=True)
    train_result = train_one_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        num_classes=num_classes,
        early_stop_metric=args.early_stop_metric,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    logs = train_result["logs"]
    if train_result["best_state"] is not None:
        model.load_state_dict(train_result["best_state"])
        print(
            f"[stage] loaded best model from epoch={train_result['best_epoch']} "
            f"{train_result['monitor_metric']}={train_result['best_metric']}",
            flush=True,
        )

    for row in logs:
        row["run_tag"] = run_tag
        row["seed"] = args.seed
        row["max_cases"] = args.max_cases
        row["train_cases"] = len(train_case_ids)
        row["val_cases"] = len(val_case_ids)
        row["train_slices"] = len(train_indices)
        row["val_slices"] = len(val_indices)
        row["batch_size"] = args.batch_size
        row["lr"] = args.lr
        row["base_channels"] = args.base_channels
        row["token_dim"] = args.token_dim
        row["image_size"] = args.image_size
        row["num_classes"] = num_classes
        row["in_channels"] = in_channels
        row["num_context_slices"] = args.num_context_slices
        row["slice_stride"] = args.slice_stride
        row["max_slices_per_case"] = args.max_slices_per_case
        row["val_ratio"] = args.val_ratio
        row["early_stop_patience"] = args.early_stop_patience
        row["early_stop_min_delta"] = args.early_stop_min_delta

    write_csv(
        run_paths["training_summary_csv"],
        [
            "epoch",
            "run_tag",
            "train_loss",
            "train_dice",
            "val_loss",
            "val_dice",
            "monitor_metric",
            "monitor_value",
            "is_best_epoch",
            "no_improve_count",
            "seed",
            "max_cases",
            "train_cases",
            "val_cases",
            "train_slices",
            "val_slices",
            "batch_size",
            "lr",
            "base_channels",
            "token_dim",
            "image_size",
            "num_classes",
            "in_channels",
            "num_context_slices",
            "slice_stride",
            "max_slices_per_case",
            "val_ratio",
            "early_stop_patience",
            "early_stop_min_delta",
        ],
        logs,
    )

    print("[stage] token export", flush=True)
    token_rows = extract_and_save_tokens(
        model=model,
        dataset=dataset,
        device=device,
        num_classes=num_classes,
        pred_dir=run_paths["pred_dir"],
        organ_name_map=organ_name_map,
    )
    for row in token_rows:
        row["run_tag"] = run_tag
    write_csv(
        run_paths["token_csv"],
        [
            "run_tag",
            "case_id",
            "slice_idx",
            "organ_id",
            "organ_name",
            "mask_source",
            "voxel_count",
            "token_json",
        ],
        token_rows,
    )

    model_path = run_paths["model_path"]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "run_tag": run_tag,
            "save_dir": str(args.save_dir),
            "num_classes": num_classes,
            "organ_name_map": {str(k): v for k, v in organ_name_map.items()},
            "base_channels": args.base_channels,
            "token_dim": args.token_dim,
            "image_size": args.image_size,
            "in_channels": in_channels,
            "num_context_slices": args.num_context_slices,
            "slice_stride": args.slice_stride,
            "max_slices_per_case": args.max_slices_per_case,
            "max_cases": args.max_cases,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_ratio": args.val_ratio,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "seed": args.seed,
            "best_epoch": train_result["best_epoch"],
            "best_metric_name": train_result["monitor_metric"],
            "best_metric_value": train_result["best_metric"],
            "stopped_early": train_result["stopped_early"],
            "train_cases": len(train_case_ids),
            "val_cases": len(val_case_ids),
            "train_slices": len(train_indices),
            "val_slices": len(val_indices),
        },
        model_path,
    )

    print(f"wrote: {run_paths['training_summary_csv']}")
    print(f"wrote: {run_paths['token_csv']}")
    print(f"wrote_pred_dir: {run_paths['pred_dir']}")
    print(f"wrote_model: {model_path}")
    print(f"best_epoch: {train_result['best_epoch']}")
    print(
        f"best_metric: {train_result['monitor_metric']}="
        f"{train_result['best_metric']}",
    )
    print(f"stopped_early: {train_result['stopped_early']}")
    print(f"token_rows: {len(token_rows)}")
    print("complete")


if __name__ == "__main__":
    main()
