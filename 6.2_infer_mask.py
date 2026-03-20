
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

# put the weight from 6.1 and apply it into the real patient, some people has missing mask and some doesn't

DEFAULT_CT_DIR = Path("output/preprocessed/ct_norm")
DEFAULT_EXPERIMENT_ROOT = Path("output/experiments/organ_seg")
DEFAULT_LEGACY_MODEL_PATH = Path("output/preprocessed/organ_tokens/models/organ_seg_unet.pt")
DEFAULT_OLD_LEGACY_MODEL_PATH = Path("data/preprocessed/organ_tokens/models/organ_seg_unet.pt")
DEFAULT_ORGAN_NAME_MAP = {
    1: "liver",
    2: "bladder",
    3: "lung",
    4: "kidney",
    5: "bone",
    6: "brain",
}


def ensure_output_dirs(output_root, mask_dir):
    output_root.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)


def resolve_model_path(model_path_arg, output_root, run_tag, allow_legacy_model_fallback):
    if model_path_arg:
        return Path(model_path_arg)

    run_model = Path(output_root) / run_tag / "model" / "organ_seg_unet.pt"
    if run_model.exists():
        return run_model
    if allow_legacy_model_fallback:
        if DEFAULT_LEGACY_MODEL_PATH.exists():
            return DEFAULT_LEGACY_MODEL_PATH
        if DEFAULT_OLD_LEGACY_MODEL_PATH.exists():
            return DEFAULT_OLD_LEGACY_MODEL_PATH
    raise RuntimeError(
        "stage6 model not found for run_tag. "
        + f"expected: {run_model}. "
        + "you can pass --model-path explicitly, or add --allow-legacy-model-fallback."
    )


def resolve_infer_paths(output_root, run_tag):

    infer_root = Path(output_root) / run_tag / "infer"
    return {
        "infer_root": infer_root,
        "mask_dir": infer_root / "masks",
        "manifest_csv": infer_root / "organ_mask_manifest.csv",
        "long_csv": infer_root / "organ_mask_long.csv",
        "token_csv": infer_root / "organ_imaging_tokens.csv",
    }


def parse_patient_id_from_ct_npz(path):
    return path.stem


def get_ct_npz_paths(ct_dir):
    paths = sorted(ct_dir.glob("*.npz"))
    return paths


def build_default_organ_map(num_classes):
    organ_map = {}
    for organ_id in range(1, num_classes):
        organ_map[organ_id] = DEFAULT_ORGAN_NAME_MAP.get(organ_id, f"organ_{organ_id}")
    return organ_map


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def parse_organ_map_from_ckpt(ckpt, num_classes):
    raw_map = ckpt.get("organ_name_map", {}) if isinstance(ckpt, dict) else {}
    parsed = {}
    for k, v in raw_map.items():
        try:
            parsed[int(k)] = str(v)
        except Exception:
            continue
    if not parsed:
        return build_default_organ_map(num_classes)
    for organ_id in range(1, num_classes):
        if organ_id not in parsed:
            parsed[organ_id] = DEFAULT_ORGAN_NAME_MAP.get(organ_id, f"organ_{organ_id}")
    return parsed


def load_model(model_path, device):
    ckpt = torch.load(str(model_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        num_classes = int(ckpt.get("num_classes", 2))
        base_channels = int(ckpt.get("base_channels", 16))
        token_dim = int(ckpt.get("token_dim", 64))
        image_size = int(ckpt.get("image_size", 256))
        in_channels = int(ckpt.get("in_channels", 1))
        num_context_slices = int(ckpt.get("num_context_slices", 0))
        slice_stride = int(ckpt.get("slice_stride", 1))
        organ_map = parse_organ_map_from_ckpt(ckpt, num_classes)
    else:
        state_dict = ckpt
        num_classes = 2
        base_channels = 16
        token_dim = 64
        image_size = 256
        in_channels = 1
        num_context_slices = 0
        slice_stride = 1
        organ_map = build_default_organ_map(num_classes)

    if "token_head.weight" not in state_dict or "token_head.bias" not in state_dict:
        raise RuntimeError(
            "checkpoint is missing token_head weights; stage 6.3 learned organ tokens require a 6.1-compatible checkpoint"
        )
    token_head_weight = state_dict["token_head.weight"]
    if int(token_head_weight.shape[0]) != token_dim:
        raise RuntimeError(
            "checkpoint token_head weight shape does not match token_dim metadata: "
            + f"weight_out={int(token_head_weight.shape[0])} token_dim={token_dim}"
        )

    model = SmallUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        token_dim=token_dim,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "failed to load 6.1-compatible organ segmentation checkpoint with learned token head: "
            + str(exc)
        ) from exc
    model.to(device)
    model.eval()
    meta = {
        "num_classes": num_classes,
        "base_channels": base_channels,
        "token_dim": token_dim,
        "image_size": image_size,
        "in_channels": in_channels,
        "num_context_slices": num_context_slices,
        "slice_stride": slice_stride,
        "organ_map": organ_map,
    }
    return model, meta


def build_context_stack(volume, z, num_context_slices, slice_stride):
    depth = int(volume.shape[0])
    channels = []
    for offset in range(-num_context_slices, num_context_slices + 1):
        idx = z + offset * slice_stride
        idx = max(0, min(depth - 1, idx))
        channels.append(np.asarray(volume[idx], dtype=np.float32))
    return np.stack(channels, axis=0).astype(np.float32)


def resize_slice_for_model(image_chw, image_size):
    h = image_chw.shape[1]
    w = image_chw.shape[2]
    zoom_h = image_size / h
    zoom_w = image_size / w
    resized = ndimage.zoom(image_chw, zoom=(1.0, zoom_h, zoom_w), order=1)
    resized = np.asarray(resized, dtype=np.float32)
    return resized


def resize_mask_back(mask_2d, target_h, target_w):
    zoom_h = target_h / mask_2d.shape[0]
    zoom_w = target_w / mask_2d.shape[1]
    resized = ndimage.zoom(mask_2d, zoom=(zoom_h, zoom_w), order=0)
    resized = np.asarray(np.rint(resized), dtype=np.int16)
    if resized.shape != (target_h, target_w):
        fixed = np.zeros((target_h, target_w), dtype=np.int16)
        y = min(target_h, resized.shape[0])
        x = min(target_w, resized.shape[1])
        fixed[:y, :x] = resized[:y, :x]
        resized = fixed
    return resized


def infer_volume_multilabel_mask_and_token_stats(
    model,
    ct_volume,
    image_size,
    batch_slices,
    device,
    num_context_slices,
    slice_stride,
    organ_map,
):
    depth = ct_volume.shape[0]
    h = ct_volume.shape[1]
    w = ct_volume.shape[2]
    pred_mask = np.zeros((depth, h, w), dtype=np.int16)
    token_dim = int(model.token_head.out_channels)
    token_sums = {
        int(organ_id): np.zeros((token_dim,), dtype=np.float64)
        for organ_id in organ_map.keys()
    }
    token_voxel_counts = {int(organ_id): 0 for organ_id in organ_map.keys()}

    resized_slices = []
    for z in range(depth):
        image_chw = build_context_stack(ct_volume, z, num_context_slices, slice_stride)
        image_chw = np.clip(image_chw, 0.0, 1.0)
        resized = resize_slice_for_model(image_chw, image_size)
        resized_slices.append(resized)

    start = 0
    while start < depth:
        end = min(depth, start + batch_slices)
        batch_np = np.stack(resized_slices[start:end], axis=0)
        batch_t = torch.from_numpy(batch_np).to(device)
        with torch.no_grad():
            logits, token_map = model(batch_t, return_token_map=True)
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int16)
            token_map_up = F.interpolate(
                token_map,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).detach().cpu()

        for i in range(pred.shape[0]):
            z = start + i
            mask_2d = resize_mask_back(pred[i], h, w)
            pred_mask[z] = mask_2d

            token_map_slice = token_map_up[i]
            mask_tensor = torch.from_numpy(mask_2d)
            for organ_id in organ_map.keys():
                organ_id = int(organ_id)
                region = mask_tensor == organ_id
                voxel_count = int(region.sum().item())
                if voxel_count == 0:
                    continue
                pooled_sum = token_map_slice[:, region].sum(dim=1).numpy().astype(np.float64)
                token_sums[organ_id] += pooled_sum
                token_voxel_counts[organ_id] += voxel_count
        start = end

    return pred_mask, token_sums, token_voxel_counts


def build_missing_flags(pred_mask, organ_map, min_organ_voxels):
    missing = {}
    voxels = {}
    for organ_id, organ_name in organ_map.items():
        count = int((pred_mask == organ_id).sum())
        voxels[organ_name] = count
        missing[organ_name] = 1 if count < min_organ_voxels else 0
    return missing, voxels


def save_patient_mask_npz(path, pred_mask, organ_map, source_ct_npz):
    organ_ids = np.asarray(list(organ_map.keys()), dtype=np.int16)
    organ_names = np.asarray([organ_map[i] for i in organ_ids], dtype=object)
    np.savez_compressed(
        path,
        multi_label_mask=pred_mask.astype(np.int16),
        organ_ids=organ_ids,
        organ_names=organ_names,
        source_ct_npz=str(source_ct_npz),
    )


def l2_normalize_token(token):
    arr = np.asarray(token, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return arr
    return arr / norm


def extract_organ_tokens_for_case(
    patient_id,
    organ_map,
    mask_npz_path,
    min_organ_voxels,
    token_sums,
    token_voxel_counts,
):
    rows = []
    for organ_id, organ_name in organ_map.items():
        organ_id = int(organ_id)
        voxel_count = int(token_voxel_counts.get(organ_id, 0))
        missing = 1 if voxel_count < min_organ_voxels else 0
        token_json = ""
        status = "missing"
        if missing == 0:
            token = token_sums[organ_id] / float(voxel_count)
            token = l2_normalize_token(token)
            token_json = json.dumps([float(x) for x in token.tolist()])
            status = "ok"
        rows.append(
            {
                "patient_id": patient_id,
                "organ_id": int(organ_id),
                "organ_name": organ_name,
                "voxel_count": voxel_count,
                "missing_img_organ": missing,
                "token_json": token_json,
                "mask_npz_path": str(mask_npz_path),
                "status": status,
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer organ masks and organ tokens on NSCLC Radiogenomics CT (docs/project.md 6.2/6.3).",
        allow_abbrev=False,
    )
    parser.add_argument("--ct-dir", type=str, default=str(DEFAULT_CT_DIR))
    parser.add_argument("--model-path", type=str, default="", help="Optional. If empty, auto-resolve from run_tag.")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--run-tag", type=str, default="search_base24")
    parser.add_argument(
        "--allow-legacy-model-fallback",
        action="store_true",
        help="Allow fallback to old non-run_tag model paths (not recommended).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="0 means run all CT cases; >0 means run first N cases.",
    )
    parser.add_argument(
        "--full-cases",
        action="store_true",
        help="Force full-dataset inference and ignore --max-cases.",
    )
    parser.add_argument("--batch-slices", type=int, default=8)
    parser.add_argument("--min-organ-voxels", type=int, default=50)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args, _ = parser.parse_known_args()
    if args.full_cases:
        args.max_cases = 0
    return args


def main():
    args = parse_args()

    run_tag = args.run_tag.strip()
    if not run_tag:
        raise SystemExit("--run-tag must not be empty")

    ct_dir = Path(args.ct_dir)
    output_root = Path(args.output_root)
    try:
        model_path = resolve_model_path(
            model_path_arg=args.model_path,
            output_root=output_root,
            run_tag=run_tag,
            allow_legacy_model_fallback=bool(args.allow_legacy_model_fallback),
        )
    except Exception as exc:
        raise SystemExit(str(exc))
    infer_paths = resolve_infer_paths(output_root, run_tag)
    infer_root = infer_paths["infer_root"]
    mask_dir = infer_paths["mask_dir"]
    manifest_csv = infer_paths["manifest_csv"]
    long_csv = infer_paths["long_csv"]
    token_csv = infer_paths["token_csv"]

    if not ct_dir.exists():
        raise SystemExit(f"ct_dir not found: {ct_dir}")
    if not model_path.exists():
        raise SystemExit(f"model_path not found: {model_path}")
    if args.max_cases < 0:
        raise SystemExit("--max-cases must be >= 0 (0 means all cases)")
    if args.batch_slices <= 0:
        raise SystemExit("--batch-slices must be >= 1")
    if args.min_organ_voxels <= 0:
        raise SystemExit("--min-organ-voxels must be >= 1")
    if args.token_dim <= 0:
        raise SystemExit("--token-dim must be >= 1")

    ensure_output_dirs(infer_root, mask_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("requested cuda but cuda is not available")
        device = torch.device(args.device)

    print(f"[start] ct_dir={ct_dir}", flush=True)
    print(f"[start] output_root={output_root} run_tag={run_tag}", flush=True)
    print(f"[start] model_path={model_path}", flush=True)
    print(
        f"[start] allow_legacy_model_fallback={bool(args.allow_legacy_model_fallback)}",
        flush=True,
    )
    print(
        f"[start] max_cases={args.max_cases} "
        f"batch_slices={args.batch_slices} "
        f"min_organ_voxels={args.min_organ_voxels} "
        f"token_dim={args.token_dim}",
        flush=True,
    )
    print(
        f"[deps] torch={torch.__version__} cuda_available={torch.cuda.is_available()} numpy={np.__version__}",
        flush=True,
    )
    print(f"[start] device={device}", flush=True)

    model, meta = load_model(model_path, device)
    num_classes = meta["num_classes"]
    image_size = meta["image_size"]
    num_context_slices = meta["num_context_slices"]
    slice_stride = meta["slice_stride"]
    organ_map = meta["organ_map"]
    model_token_dim = int(meta["token_dim"])
    if args.token_dim != model_token_dim:
        raise SystemExit(
            "--token-dim must match checkpoint token_dim for learned organ tokens: "
            + f"arg={args.token_dim} checkpoint={model_token_dim}"
        )
    print(
        f"[model] num_classes={num_classes} image_size={image_size} base_channels={meta['base_channels']} "
        f"in_channels={meta['in_channels']} context={num_context_slices} stride={slice_stride} "
        f"token_dim={model_token_dim}",
        flush=True,
    )

    ct_paths = get_ct_npz_paths(ct_dir)
    if len(ct_paths) == 0:
        raise SystemExit(f"no ct npz found in: {ct_dir}")
    total_cases = len(ct_paths)
    if args.max_cases > 0:
        ct_paths = ct_paths[: min(args.max_cases, total_cases)]
    print(
        f"[start] selected_cases={len(ct_paths)} total_cases={total_cases} "
        f"full_cases={1 if args.max_cases == 0 else 0}",
        flush=True,
    )

    manifest_rows = []
    long_rows = []
    token_rows = []
    ok_count = 0
    fail_count = 0

    for i, ct_path in enumerate(ct_paths, start=1):
        patient_id = parse_patient_id_from_ct_npz(ct_path)
        print(f"[infer] {i}/{len(ct_paths)} patient_id={patient_id}", flush=True)
        summary_row = {
            "run_tag": run_tag,
            "patient_id": patient_id,
            "ct_npz_path": str(ct_path),
            "mask_npz_path": "",
            "model_path": str(model_path),
            "batch_slices": args.batch_slices,
            "min_organ_voxels": args.min_organ_voxels,
            "status": "",
            "error": "",
            "missing_img_organ_json": "",
            "organ_voxel_count_json": "",
        }

        try:
            with np.load(ct_path, allow_pickle=True) as z:
                if "ct_volume" not in z:
                    raise RuntimeError("ct_volume key missing")
                ct_volume = np.asarray(z["ct_volume"], dtype=np.float32)

            pred_mask, token_sums, token_voxel_counts = infer_volume_multilabel_mask_and_token_stats(
                model=model,
                ct_volume=ct_volume,
                image_size=image_size,
                batch_slices=args.batch_slices,
                device=device,
                num_context_slices=num_context_slices,
                slice_stride=slice_stride,
                organ_map=organ_map,
            )
            missing_dict, voxel_dict = build_missing_flags(
                pred_mask=pred_mask,
                organ_map=organ_map,
                min_organ_voxels=args.min_organ_voxels,
            )

            out_mask_path = mask_dir / f"{patient_id}.npz"
            save_patient_mask_npz(
                path=out_mask_path,
                pred_mask=pred_mask,
                organ_map=organ_map,
                source_ct_npz=ct_path,
            )

            summary_row["mask_npz_path"] = str(out_mask_path)
            summary_row["status"] = "ok"
            summary_row["missing_img_organ_json"] = json.dumps(missing_dict)
            summary_row["organ_voxel_count_json"] = json.dumps(voxel_dict)
            manifest_rows.append(summary_row)

            for organ_id, organ_name in organ_map.items():
                long_rows.append(
                    {
                        "run_tag": run_tag,
                        "patient_id": patient_id,
                        "organ_id": organ_id,
                        "organ_name": organ_name,
                        "model_path": str(model_path),
                        "batch_slices": args.batch_slices,
                        "min_organ_voxels": args.min_organ_voxels,
                        "voxel_count": voxel_dict[organ_name],
                        "missing_img_organ": missing_dict[organ_name],
                        "mask_npz_path": str(out_mask_path),
                        "status": "ok",
                    }
                )

            case_token_rows = extract_organ_tokens_for_case(
                patient_id=patient_id,
                organ_map=organ_map,
                mask_npz_path=out_mask_path,
                min_organ_voxels=args.min_organ_voxels,
                token_sums=token_sums,
                token_voxel_counts=token_voxel_counts,
            )
            for row in case_token_rows:
                row["run_tag"] = run_tag
                row["model_path"] = str(model_path)
                row["batch_slices"] = args.batch_slices
                row["min_organ_voxels"] = args.min_organ_voxels
                row["token_dim"] = model_token_dim
            token_rows.extend(case_token_rows)

            ok_count += 1
        except Exception as exc:
            summary_row["status"] = "failed"
            summary_row["error"] = str(exc)
            manifest_rows.append(summary_row)
            for organ_id, organ_name in organ_map.items():
                long_rows.append(
                    {
                        "run_tag": run_tag,
                        "patient_id": patient_id,
                        "organ_id": organ_id,
                        "organ_name": organ_name,
                        "model_path": str(model_path),
                        "batch_slices": args.batch_slices,
                        "min_organ_voxels": args.min_organ_voxels,
                        "voxel_count": 0,
                        "missing_img_organ": 1,
                        "mask_npz_path": "",
                        "status": "failed",
                    }
                )
                token_rows.append(
                    {
                        "run_tag": run_tag,
                        "patient_id": patient_id,
                        "organ_id": organ_id,
                        "organ_name": organ_name,
                        "model_path": str(model_path),
                        "batch_slices": args.batch_slices,
                        "min_organ_voxels": args.min_organ_voxels,
                        "token_dim": args.token_dim,
                        "voxel_count": 0,
                        "missing_img_organ": 1,
                        "token_json": "",
                        "mask_npz_path": "",
                        "status": "failed",
                    }
                )
            fail_count += 1

    write_csv(
        manifest_csv,
        [
            "run_tag",
            "patient_id",
            "ct_npz_path",
            "mask_npz_path",
            "model_path",
            "batch_slices",
            "min_organ_voxels",
            "status",
            "error",
            "missing_img_organ_json",
            "organ_voxel_count_json",
        ],
        manifest_rows,
    )
    write_csv(
        long_csv,
        [
            "run_tag",
            "patient_id",
            "organ_id",
            "organ_name",
            "model_path",
            "batch_slices",
            "min_organ_voxels",
            "voxel_count",
            "missing_img_organ",
            "mask_npz_path",
            "status",
        ],
        long_rows,
    )
    write_csv(
        token_csv,
        [
            "run_tag",
            "patient_id",
            "organ_id",
            "organ_name",
            "model_path",
            "batch_slices",
            "min_organ_voxels",
            "token_dim",
            "voxel_count",
            "missing_img_organ",
            "token_json",
            "mask_npz_path",
            "status",
        ],
        token_rows,
    )

    print(f"wrote: {manifest_csv}", flush=True)
    print(f"wrote: {long_csv}", flush=True)
    print(f"wrote: {token_csv}", flush=True)
    print(f"infer_root: {infer_root}", flush=True)
    print(f"wrote_mask_dir: {mask_dir}", flush=True)
    print(f"processed_cases: {len(ct_paths)}", flush=True)
    print(f"ok_cases: {ok_count}", flush=True)
    print(f"failed_cases: {fail_count}", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
