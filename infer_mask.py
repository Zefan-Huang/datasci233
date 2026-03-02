"""
作用：
- 实现 project.md 6.2：在 NSCLC Radiogenomics CT 上推理器官 mask。
- 产出每例器官分割结果，并给出每个器官的 missing_img_organ 标记（不可见/不可用）。

输入：
- output/preprocessed/ct_norm/*.npz（来自 imaging_preprocessing.py）
- output/experiments/organ_seg/<run_tag>/model/organ_seg_unet.pt（来自 seg_model.py）

输出：
- <output_root>/<run_tag>/infer/masks/<patient_id>.npz
- <output_root>/<run_tag>/infer/organ_mask_manifest.csv
- <output_root>/<run_tag>/infer/organ_mask_long.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn


DEFAULT_CT_DIR = Path("output/preprocessed/ct_norm")
DEFAULT_EXPERIMENT_ROOT = Path("output/experiments/organ_seg")
DEFAULT_LEGACY_MODEL_PATH = Path("output/preprocessed/organ_tokens/models/organ_seg_unet.pt")
DEFAULT_OLD_LEGACY_MODEL_PATH = Path("data/preprocessed/organ_tokens/models/organ_seg_unet.pt")


def ensure_output_dirs(output_root, mask_dir):
    """Why: 推理会写 mask 和统计表，需要先创建目录。

    Content: 创建输出根目录和 mask 子目录。
    Input: output_root、mask_dir。
    Output: 目录创建完成。
    """
    output_root.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)


def resolve_model_path(model_path_arg, output_root, run_tag):
    """Why: 推理默认应优先使用同一 run_tag 的最佳模型，避免拿错版本。

    Content: 若传入 model_path 就直接用；否则先找 run_tag 模型，再回退 legacy 模型。
    Input: model_path_arg、output_root、run_tag。
    Output: 解析后的模型路径。
    """
    if model_path_arg:
        return Path(model_path_arg)

    run_model = Path(output_root) / run_tag / "model" / "organ_seg_unet.pt"
    if run_model.exists():
        return run_model
    if DEFAULT_LEGACY_MODEL_PATH.exists():
        return DEFAULT_LEGACY_MODEL_PATH
    return DEFAULT_OLD_LEGACY_MODEL_PATH


def resolve_infer_paths(output_root, run_tag):
    """Why: 不同推理实验需要隔离目录，避免覆盖前一次结果。

    Content: 根据 output_root/run_tag 生成 infer 输出路径。
    Input: output_root、run_tag。
    Output: 包含 infer_root/mask_dir/manifest_csv/long_csv 的字典。
    """
    infer_root = Path(output_root) / run_tag / "infer"
    return {
        "infer_root": infer_root,
        "mask_dir": infer_root / "masks",
        "manifest_csv": infer_root / "organ_mask_manifest.csv",
        "long_csv": infer_root / "organ_mask_long.csv",
    }


def parse_patient_id_from_ct_npz(path):
    """Why: 输出文件与统计表都需要稳定的 patient_id。

    Content: 使用 ct_norm 文件名 stem 作为 patient_id。
    Input: path（ct npz 路径）。
    Output: patient_id 字符串。
    """
    return path.stem


def get_ct_npz_paths(ct_dir):
    """Why: 需要批量推理全部 Radiogenomics CT。

    Content: 扫描 ct_dir 下全部 .npz 文件并排序。
    Input: ct_dir。
    Output: 排序后的路径列表。
    """
    paths = sorted(ct_dir.glob("*.npz"))
    return paths


def parse_organ_map(num_classes):
    """Why: long 表需要可读器官名，同时保持和模型输出类别一致。

    Content: 将 organ_id 映射到默认名称 organ_1...organ_k。
    Input: num_classes（含背景类）。
    Output: organ_id -> organ_name 字典（不含背景0）。
    """
    organ_map = {}
    for organ_id in range(1, num_classes):
        organ_map[organ_id] = f"organ_{organ_id}"
    return organ_map


def write_csv(path, fieldnames, rows):
    """Why: 结果需要稳定表结构，便于后续 token 化与融合。

    Content: 按字段顺序写 CSV。
    Input: path、fieldnames、rows。
    Output: CSV 写入完成。
    """
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class DoubleConv(nn.Module):
    """Why: 与训练时 U-Net 结构保持一致，才能正确加载权重。

    Content: 两层卷积+BN+ReLU。
    Input: 2D feature map。
    Output: 同分辨率增强特征。
    """

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
    """Why: 推理网络必须和 seg_model.py 训练网络同构。

    Content: 轻量 2D U-Net，输出多类别分割 logits。
    Input: x [B,1,H,W]。
    Output: logits [B,C,H,W]。
    """

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

        # Keep this layer for state_dict compatibility with the training model.
        self.token_head = nn.Conv2d(c3, token_dim, kernel_size=1)

    def forward(self, x):
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
        return logits


def load_model(model_path, device):
    """Why: 6.2 的核心是把 6.1 训练出的模型用于 NSCLC CT 推理。

    Content: 读取 checkpoint，构建模型并加载权重。
    Input: model_path、device。
    Output: 已加载模型与 checkpoint 元信息字典。
    """
    ckpt = torch.load(str(model_path), map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        num_classes = int(ckpt.get("num_classes", 2))
        base_channels = int(ckpt.get("base_channels", 16))
        token_dim = int(ckpt.get("token_dim", 64))
        image_size = int(ckpt.get("image_size", 256))
    else:
        state_dict = ckpt
        num_classes = 2
        base_channels = 16
        token_dim = 64
        image_size = 256

    model = SmallUNet(
        in_channels=1,
        num_classes=num_classes,
        base_channels=base_channels,
        token_dim=token_dim,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    meta = {
        "num_classes": num_classes,
        "base_channels": base_channels,
        "token_dim": token_dim,
        "image_size": image_size,
    }
    return model, meta


def resize_slice_for_model(slice_2d, image_size):
    """Why: 训练输入是固定尺寸，推理也要统一尺寸。

    Content: 将单张切片线性插值到 image_size x image_size。
    Input: slice_2d、image_size。
    Output: resized_slice 与 zoom 因子。
    """
    h = slice_2d.shape[0]
    w = slice_2d.shape[1]
    zoom_h = image_size / h
    zoom_w = image_size / w
    resized = ndimage.zoom(slice_2d, zoom=(zoom_h, zoom_w), order=1)
    resized = np.asarray(resized, dtype=np.float32)
    return resized


def resize_mask_back(mask_2d, target_h, target_w):
    """Why: 模型输出是固定尺寸，必须回到原始 CT 尺寸对齐。

    Content: 最近邻缩放预测 mask 到目标高宽。
    Input: mask_2d、target_h、target_w。
    Output: 对齐后的整型 mask。
    """
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


def infer_volume_multilabel_mask(model, ct_volume, image_size, batch_slices, device):
    """Why: 6.2 需要对每个病人的 3D CT 推理器官分割结果。

    Content: 对每个轴向切片做 2D 分割，并堆叠成 3D 多类别 mask。
    Input: model、ct_volume、image_size、batch_slices、device。
    Output: pred_mask_3d（int16）。
    """
    depth = ct_volume.shape[0]
    h = ct_volume.shape[1]
    w = ct_volume.shape[2]
    pred_mask = np.zeros((depth, h, w), dtype=np.int16)

    resized_slices = []
    for z in range(depth):
        sl = np.asarray(ct_volume[z], dtype=np.float32)
        sl = np.clip(sl, 0.0, 1.0)
        resized = resize_slice_for_model(sl, image_size)
        resized_slices.append(resized)

    start = 0
    while start < depth:
        end = min(depth, start + batch_slices)
        batch_np = np.stack(resized_slices[start:end], axis=0)[:, None, :, :]
        batch_t = torch.from_numpy(batch_np).to(device)
        with torch.no_grad():
            logits = model(batch_t)
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int16)

        for i in range(pred.shape[0]):
            z = start + i
            pred_mask[z] = resize_mask_back(pred[i], h, w)
        start = end

    return pred_mask


def build_missing_flags(pred_mask, organ_map, min_organ_voxels):
    """Why: project.md 6.2 要求 missing_img_organ，用于后续缺失模态处理。

    Content: 按器官统计体素数，低于阈值记为 missing。
    Input: pred_mask、organ_map、min_organ_voxels。
    Output: (missing_dict, voxel_dict)。
    """
    missing = {}
    voxels = {}
    for organ_id, organ_name in organ_map.items():
        count = int((pred_mask == organ_id).sum())
        voxels[organ_name] = count
        missing[organ_name] = 1 if count < min_organ_voxels else 0
    return missing, voxels


def save_patient_mask_npz(path, pred_mask, organ_map, source_ct_npz):
    """Why: 下游 token 抽取会重复使用器官 mask，需持久化。

    Content: 保存多类别 mask 和器官映射元信息到压缩 npz。
    Input: path、pred_mask、organ_map、source_ct_npz。
    Output: 文件写入完成。
    """
    organ_ids = np.asarray(list(organ_map.keys()), dtype=np.int16)
    organ_names = np.asarray([organ_map[i] for i in organ_ids], dtype=object)
    np.savez_compressed(
        path,
        multi_label_mask=pred_mask.astype(np.int16),
        organ_ids=organ_ids,
        organ_names=organ_names,
        source_ct_npz=str(source_ct_npz),
    )


def parse_args():
    """Why: 让你一次脚本就能在调试/全量两种模式间切换。

    Content: 解析 6.2 推理需要的参数。
    Input: 命令行参数。
    Output: 参数对象。
    """
    parser = argparse.ArgumentParser(
        description="Infer organ masks on NSCLC Radiogenomics CT (project.md 6.2).",
        allow_abbrev=False,
    )
    parser.add_argument("--ct-dir", type=str, default=str(DEFAULT_CT_DIR))
    parser.add_argument("--model-path", type=str, default="", help="Optional. If empty, auto-resolve from run_tag.")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--run-tag", type=str, default="search_base24")
    parser.add_argument("--max-cases", type=int, default=None, help="None means run all CT cases.")
    parser.add_argument("--batch-slices", type=int, default=8)
    parser.add_argument("--min-organ-voxels", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args, _ = parser.parse_known_args()
    return args


def main():
    """Why: 一条命令执行 project.md 6.2 全流程。

    Content: 加载模型、批量推理 CT、保存 mask、输出 missing_img_organ 清单。
    Input: 命令行参数。
    Output: 6.2 所需 mask 与 CSV 结果文件。
    """
    args = parse_args()
    run_tag = args.run_tag.strip()
    if not run_tag:
        raise SystemExit("--run-tag must not be empty")

    ct_dir = Path(args.ct_dir)
    output_root = Path(args.output_root)
    model_path = resolve_model_path(args.model_path, output_root, run_tag)
    infer_paths = resolve_infer_paths(output_root, run_tag)
    infer_root = infer_paths["infer_root"]
    mask_dir = infer_paths["mask_dir"]
    manifest_csv = infer_paths["manifest_csv"]
    long_csv = infer_paths["long_csv"]

    if not ct_dir.exists():
        raise SystemExit(f"ct_dir not found: {ct_dir}")
    if not model_path.exists():
        raise SystemExit(f"model_path not found: {model_path}")
    if args.max_cases is not None and args.max_cases <= 0:
        raise SystemExit("--max-cases must be >= 1")
    if args.batch_slices <= 0:
        raise SystemExit("--batch-slices must be >= 1")

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
        f"[start] max_cases={args.max_cases} "
        f"batch_slices={args.batch_slices} "
        f"min_organ_voxels={args.min_organ_voxels}",
        flush=True,
    )
    print(f"[start] device={device}", flush=True)

    model, meta = load_model(model_path, device)
    num_classes = meta["num_classes"]
    image_size = meta["image_size"]
    organ_map = parse_organ_map(num_classes)
    print(
        f"[model] num_classes={num_classes} image_size={image_size} "
        f"base_channels={meta['base_channels']}",
        flush=True,
    )

    ct_paths = get_ct_npz_paths(ct_dir)
    if len(ct_paths) == 0:
        raise SystemExit(f"no ct npz found in: {ct_dir}")
    if args.max_cases is not None:
        ct_paths = ct_paths[: min(args.max_cases, len(ct_paths))]
    print(f"[start] selected_cases={len(ct_paths)}", flush=True)

    manifest_rows = []
    long_rows = []
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

            pred_mask = infer_volume_multilabel_mask(
                model=model,
                ct_volume=ct_volume,
                image_size=image_size,
                batch_slices=args.batch_slices,
                device=device,
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

    print(f"wrote: {manifest_csv}", flush=True)
    print(f"wrote: {long_csv}", flush=True)
    print(f"infer_root: {infer_root}", flush=True)
    print(f"wrote_mask_dir: {mask_dir}", flush=True)
    print(f"processed_cases: {len(ct_paths)}", flush=True)
    print(f"ok_cases: {ok_count}", flush=True)
    print(f"failed_cases: {fail_count}", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
