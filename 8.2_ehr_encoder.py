"""
作用：
- 实现 project.md Stage 8.2：EHR encoder（MLP）生成 g_ehr。
- 读取 Stage 8.1 的 x_ehr 特征，训练轻量 MLP 编码器（MVP）。
- 当前 MLP 版本要求 Stage 8.1 使用 one-hot 类别编码，避免把类别索引误当连续值输入。
- 导出 g_ehr、模型权重、训练日志与元信息。

输入：
- output/stage8/8.1_clinical_feature_engineering/x_ehr_features.npz（优先）
- output/stage8/8.1_clinical_feature_engineering_smoke/x_ehr_features.npz（回退）

输出：
- output/stage8/8.2_ehr_encoder/model/ehr_encoder.pt
- output/stage8/8.2_ehr_encoder/train/ehr_encoder_training_summary.csv
- output/stage8/8.2_ehr_encoder/train/ehr_encoder_meta.json
- output/stage8/8.2_ehr_encoder/tokens/g_ehr.csv
- output/stage8/8.2_ehr_encoder/tokens/ehr_encoder_outputs.npz
"""
import argparse
import csv
import json
import random
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

if nn is None:
    class _NNPlaceholder:
        """Why: 缺失 torch 时保持脚本可导入，运行时再统一报依赖。"""

        Module = object

    nn = _NNPlaceholder()


PRIMARY_STAGE81_NPZ = Path("output/stage8/8.1_clinical_feature_engineering/x_ehr_features.npz")
FALLBACK_STAGE81_NPZ = Path("output/stage8/8.1_clinical_feature_engineering_smoke/x_ehr_features.npz")
DEFAULT_OUTPUT_ROOT = Path("output/stage8/8.2_ehr_encoder")


def check_dependencies():
    """Why: Stage 8.2 的训练依赖 numpy/torch，需提前检查。

    Content: 检查依赖是否可用。
    Input: 无。
    Output: 缺失依赖名称列表。
    """
    missing = []
    if np is None:
        missing.append("numpy")
    if torch is None or DataLoader is None or TensorDataset is None:
        missing.append("torch")
    return missing


def resolve_stage81_npz(path_arg):
    """Why: 8.2 必须读取 8.1 的 x_ehr，路径需稳定解析。

    Content: 优先命令行路径，否则按默认与回退路径查找。
    Input: path_arg。
    Output: 可用 NPZ 路径。
    """
    if path_arg.strip():
        p = Path(path_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"stage8.1 npz not found: {p}")
    if PRIMARY_STAGE81_NPZ.exists():
        return PRIMARY_STAGE81_NPZ
    if FALLBACK_STAGE81_NPZ.exists():
        return FALLBACK_STAGE81_NPZ
    raise FileNotFoundError(
        "stage8.1 npz not found. expected one of: "
        + f"{PRIMARY_STAGE81_NPZ}, {FALLBACK_STAGE81_NPZ}"
    )


def resolve_output_paths(output_root):
    """Why: 模型/训练日志/token 需要分目录保存，避免结果混杂。

    Content: 组装 Stage 8.2 输出路径集合。
    Input: output_root。
    Output: 路径字典。
    """
    root = Path(output_root)
    train_dir = root / "train"
    model_dir = root / "model"
    token_dir = root / "tokens"
    return {
        "root": root,
        "train_dir": train_dir,
        "model_dir": model_dir,
        "token_dir": token_dir,
        "model_path": model_dir / "ehr_encoder.pt",
        "train_csv": train_dir / "ehr_encoder_training_summary.csv",
        "meta_json": train_dir / "ehr_encoder_meta.json",
        "g_ehr_csv": token_dir / "g_ehr.csv",
        "output_npz": token_dir / "ehr_encoder_outputs.npz",
    }


def ensure_output_dirs(paths):
    """Why: 输出前先创建目录，避免中途写文件失败。

    Content: 创建 train/model/tokens 目录。
    Input: paths。
    Output: 目录创建完成。
    """
    paths["train_dir"].mkdir(parents=True, exist_ok=True)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    paths["token_dir"].mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    """Why: 训练涉及随机性，固定 seed 能保证可复现。

    Content: 固定 python/numpy/torch 随机种子。
    Input: seed。
    Output: 随机状态被固定。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_stage81_npz(npz_path):
    """Why: Stage 8.2 训练要读取 x_ehr 与病人顺序，需统一校验输入结构。

    Content: 读取 x_ehr/patient_ids/feature_names，并返回结构化字典。
    Input: npz_path。
    Output: 包含 x_ehr、patient_ids、feature_names 的字典。
    """
    with np.load(npz_path, allow_pickle=True) as z:
        for key in ["x_ehr", "patient_ids", "feature_names"]:
            if key not in z:
                raise RuntimeError(f"stage8.1 npz missing key: {key}")
        x_ehr = np.asarray(z["x_ehr"], dtype=np.float32)
        patient_ids = np.asarray(z["patient_ids"]).astype(str)
        feature_names = np.asarray(z["feature_names"]).astype(str)
        categorical_encoding = ""
        if "categorical_encoding" in z:
            raw = np.asarray(z["categorical_encoding"]).astype(str)
            if raw.size > 0:
                categorical_encoding = str(raw.flatten()[0])

    if x_ehr.ndim != 2:
        raise RuntimeError(f"x_ehr must be 2D, got shape={x_ehr.shape}")
    if x_ehr.shape[0] != len(patient_ids):
        raise RuntimeError("x_ehr row count mismatches patient_ids")
    if x_ehr.shape[1] != len(feature_names):
        raise RuntimeError("x_ehr col count mismatches feature_names")

    return {
        "x_ehr": x_ehr,
        "patient_ids": patient_ids,
        "feature_names": feature_names,
        "categorical_encoding": categorical_encoding,
    }


def apply_patient_limit(x_ehr, patient_ids, max_patients):
    """Why: 支持快速调试前 N 个病人，但默认应支持全量。

    Content: 根据 max_patients 截断样本；0 表示全量。
    Input: x_ehr、patient_ids、max_patients。
    Output: 截断后的 x_ehr 与 patient_ids。
    """
    if max_patients < 0:
        raise RuntimeError("max_patients must be >= 0")
    if max_patients == 0:
        return x_ehr, patient_ids
    n = min(max_patients, len(patient_ids))
    return x_ehr[:n], patient_ids[:n]


def split_train_val_indices(n_samples, val_ratio, seed):
    """Why: 训练需要验证集来监控过拟合并支持早停。

    Content: 随机划分 train/val；样本过少时回退为纯训练。
    Input: n_samples、val_ratio、seed。
    Output: train_idx、val_idx。
    """
    if val_ratio < 0 or val_ratio >= 1:
        raise RuntimeError("val_ratio must be in [0,1)")

    idx = np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    if n_samples < 4 or val_ratio == 0:
        return idx.tolist(), []

    val_count = int(round(n_samples * val_ratio))
    val_count = max(1, min(val_count, n_samples - 2))
    val_idx = idx[:val_count].tolist()
    train_idx = idx[val_count:].tolist()
    return train_idx, val_idx


def build_dataloaders(x_ehr, train_idx, val_idx, batch_size):
    """Why: 训练和验证都需要 DataLoader 统一批处理流程。

    Content: 基于索引构建 train/val DataLoader。
    Input: x_ehr、train_idx、val_idx、batch_size。
    Output: train_loader、val_loader。
    """
    if batch_size <= 0:
        raise RuntimeError("batch_size must be >= 1")

    train_x = torch.from_numpy(x_ehr[train_idx]).float()
    train_loader = DataLoader(TensorDataset(train_x), batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_idx:
        val_x = torch.from_numpy(x_ehr[val_idx]).float()
        val_loader = DataLoader(TensorDataset(val_x), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class EHREncoderMLP(nn.Module):
    """Why: Stage 8.2 需要 MLP 把 x_ehr 编码为 g_ehr（MVP 方案）。

    Content: 编码器输出 g_ehr，解码器重建 x_ehr 作为训练约束。
    Input: x [B,p]。
    Output: recon [B,p], g [B,d]。
    """

    def __init__(self, input_dim, g_dim, hidden_dim, dropout):
        super().__init__()
        self.input_dim = int(input_dim)
        self.g_dim = int(g_dim)
        self.hidden_dim = int(hidden_dim)

        self.enc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.enc2 = nn.Linear(self.hidden_dim, self.g_dim)
        self.drop = nn.Dropout(float(dropout))

        self.dec1 = nn.Linear(self.g_dim, self.hidden_dim)
        self.dec2 = nn.Linear(self.hidden_dim, self.input_dim)

    def encode(self, x):
        h = torch.relu(self.enc1(x))
        h = self.drop(h)
        g = torch.relu(self.enc2(h))
        return g

    def decode(self, g):
        h = torch.relu(self.dec1(g))
        recon = self.dec2(h)
        return recon

    def forward(self, x):
        g = self.encode(x)
        recon = self.decode(g)
        return recon, g


def eval_loss(model, loader, device):
    """Why: 早停和最佳模型选择需要稳定的验证损失计算。

    Content: 在 loader 上计算平均 MSE。
    Input: model、loader、device。
    Output: 平均 loss 或 None。
    """
    if loader is None:
        return None

    mse = nn.MSELoss(reduction="mean")
    model.eval()
    loss_sum = 0.0
    step_count = 0
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, _g = model(x)
            loss = mse(recon, x)
            loss_sum += float(loss.item())
            step_count += 1
    if step_count == 0:
        return None
    return loss_sum / max(step_count, 1)


def train_model(model, train_loader, val_loader, device, epochs, lr, weight_decay, early_stop_patience, early_stop_min_delta):
    """Why: 需要训练 MLP 参数，才能得到稳定的 g_ehr 表征。

    Content: 训练重建任务并按 val/train loss 早停。
    Input: model、train_loader、val_loader、device、epochs、lr、weight_decay、early_stop_patience、early_stop_min_delta。
    Output: 训练结果字典（best epoch/metric/history）。
    """
    if epochs <= 0:
        raise RuntimeError("epochs must be >= 1")
    if lr <= 0:
        raise RuntimeError("lr must be > 0")
    if early_stop_patience <= 0:
        raise RuntimeError("early_stop_patience must be >= 1")
    if early_stop_min_delta < 0:
        raise RuntimeError("early_stop_min_delta must be >= 0")

    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    mse = nn.MSELoss(reduction="mean")

    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_state = None
    stopped_early = False
    history_rows = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_step_count = 0

        for (x,) in train_loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            recon, _g = model(x)
            loss = mse(recon, x)
            loss.backward()
            opt.step()

            train_loss_sum += float(loss.item())
            train_step_count += 1

        train_loss = train_loss_sum / max(train_step_count, 1)
        val_loss = eval_loss(model, val_loader, device)
        monitor = train_loss if val_loss is None else val_loss

        improved = monitor < (best_metric - float(early_stop_min_delta))
        if improved:
            best_metric = float(monitor)
            best_epoch = int(epoch)
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": "" if val_loss is None else float(val_loss),
                "monitor_loss": float(monitor),
                "is_best_epoch": 1 if improved else 0,
                "bad_epochs": int(bad_epochs),
            }
        )

        val_text = "NA" if val_loss is None else f"{val_loss:.6f}"
        print(
            f"[train] epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_text} "
            f"best_loss={best_metric:.6f} best_epoch={best_epoch}"
        )

        if bad_epochs >= int(early_stop_patience):
            stopped_early = True
            print(f"[early-stop] stop at epoch={epoch} patience={early_stop_patience}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "stopped_early": bool(stopped_early),
        "history_rows": history_rows,
    }


def infer_g_ehr(model, x_ehr, device, infer_batch_size):
    """Why: 训练后需为所有病人导出 g_ehr。

    Content: 批量前向推理，仅取编码器输出 g_ehr。
    Input: model、x_ehr、device、infer_batch_size。
    Output: g_ehr 矩阵。
    """
    if infer_batch_size <= 0:
        raise RuntimeError("infer_batch_size must be >= 1")

    ds = TensorDataset(torch.from_numpy(x_ehr).float())
    loader = DataLoader(ds, batch_size=infer_batch_size, shuffle=False)

    model.eval()
    chunks = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            _recon, g = model(x)
            chunks.append(g.detach().cpu().numpy())

    g_ehr = np.concatenate(chunks, axis=0).astype(np.float32)
    return g_ehr


def l2_normalize_rows(mat):
    """Why: g_ehr 向量尺度需稳定，便于后续融合。

    Content: 对每行做 L2 归一化，零向量保持零。
    Input: mat。
    Output: 归一化矩阵。
    """
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norm > 1e-8, norm, 1.0)
    out = mat / safe
    out[norm.flatten() <= 1e-8] = 0.0
    return out.astype(np.float32)


def write_csv(path, fieldnames, rows):
    """Why: 训练日志与 token 结果都需固定结构，便于后续读取。

    Content: 按字段顺序写 CSV。
    Input: path、fieldnames、rows。
    Output: CSV 文件。
    """
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_g_ehr_csv(path, patient_ids, g_ehr):
    """Why: 下游 tabular 融合时 CSV 最直接，便于人工检查。

    Content: 写 patient_id + g_ehr 向量列。
    Input: path、patient_ids、g_ehr。
    Output: g_ehr.csv。
    """
    fields = ["patient_id"] + [f"g_{i:03d}" for i in range(g_ehr.shape[1])]
    rows = []
    for i, pid in enumerate(patient_ids):
        row = {"patient_id": str(pid)}
        for j in range(g_ehr.shape[1]):
            row[f"g_{j:03d}"] = float(g_ehr[i, j])
        rows.append(row)
    write_csv(path, fields, rows)


def write_meta_json(path, meta):
    """Why: 训练配置和关键指标需落盘，便于复现和审计。

    Content: 写 JSON。
    Input: path、meta。
    Output: meta.json。
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args():
    """Why: Stage 8.2 需要可调超参，支持快速试验与稳定复现。

    Content: 解析参数并忽略 IDE 注入未知参数。
    Input: 命令行参数。
    Output: 参数对象。
    """
    parser = argparse.ArgumentParser(
        description="Stage 8.2 EHR encoder: MLP -> g_ehr.",
        allow_abbrev=False,
    )
    parser.add_argument("--stage81-npz", type=str, default="")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="0 means all patients; >0 means use first N patients.",
    )
    parser.add_argument("--g-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--infer-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"ignore_unknown_args: {unknown}")
    return args


def main():
    """Why: 一条命令跑完 Stage 8.2 全流程并导出 g_ehr。

    Content: 读 8.1、训练 EHR encoder、导出 g_ehr 与训练记录。
    Input: 命令行参数。
    Output: model/csv/npz/json 文件。
    """
    args = parse_args()
    missing = check_dependencies()
    if missing:
        raise SystemExit(
            "missing dependency: "
            + ",".join(missing)
            + ". install example: .venv/bin/pip install numpy torch"
        )

    if args.max_patients < 0:
        raise SystemExit("--max-patients must be >= 0 (0 means all patients)")
    if args.g_dim <= 0:
        raise SystemExit("--g-dim must be >= 1")
    if args.hidden_dim <= 0:
        raise SystemExit("--hidden-dim must be >= 1")
    if args.dropout < 0 or args.dropout >= 1:
        raise SystemExit("--dropout must be in [0,1)")

    npz_path = resolve_stage81_npz(args.stage81_npz)
    paths = resolve_output_paths(args.output_root)
    ensure_output_dirs(paths)

    set_seed(args.seed)

    data = load_stage81_npz(npz_path)
    x_ehr = data["x_ehr"]
    patient_ids = data["patient_ids"]
    feature_names = data["feature_names"]
    categorical_encoding = data["categorical_encoding"]

    x_ehr, patient_ids = apply_patient_limit(x_ehr, patient_ids, args.max_patients)
    total_patients = len(data["patient_ids"])

    train_idx, val_idx = split_train_val_indices(
        n_samples=x_ehr.shape[0],
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_loader, val_loader = build_dataloaders(
        x_ehr=x_ehr,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=args.batch_size,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("requested cuda but cuda is not available")
        device = torch.device(args.device)

    if categorical_encoding and categorical_encoding != "onehot":
        raise SystemExit(
            "stage8.2 MLP expects Stage 8.1 categorical_encoding=onehot, "
            + f"got '{categorical_encoding}'. "
            + "rerun stage8.1 with --categorical-encoding onehot, or add an embedding-based encoder first."
        )

    print(f"[start] stage81_npz={npz_path}")
    print(f"[start] output_root={paths['root']}")
    print(
        f"[start] selected_patients={len(patient_ids)} total_patients={total_patients} "
        f"full_patients={1 if args.max_patients == 0 else 0}"
    )
    print(
        f"[start] input_dim={x_ehr.shape[1]} g_dim={args.g_dim} hidden_dim={args.hidden_dim} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} val_ratio={args.val_ratio}"
    )
    print(
        f"[start] categorical_encoding_from_stage81={categorical_encoding if categorical_encoding else 'unknown'}"
    )
    print(
        f"[deps] torch={torch.__version__} cuda_available={torch.cuda.is_available()} numpy={np.__version__} device={device}"
    )

    model = EHREncoderMLP(
        input_dim=x_ehr.shape[1],
        g_dim=args.g_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    print("[stage] train ehr encoder")
    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x_ehr.shape[1]),
            "g_dim": int(args.g_dim),
            "hidden_dim": int(args.hidden_dim),
            "feature_names": feature_names.astype(str),
            "stage81_npz": str(npz_path),
            "categorical_encoding": str(categorical_encoding),
        },
        str(paths["model_path"]),
    )

    write_csv(
        paths["train_csv"],
        ["epoch", "train_loss", "val_loss", "monitor_loss", "is_best_epoch", "bad_epochs"],
        train_result["history_rows"],
    )

    print("[stage] infer g_ehr")
    g_ehr = infer_g_ehr(
        model=model,
        x_ehr=x_ehr,
        device=device,
        infer_batch_size=args.infer_batch_size,
    )
    g_ehr = l2_normalize_rows(g_ehr)

    write_g_ehr_csv(paths["g_ehr_csv"], patient_ids, g_ehr)
    np.savez_compressed(
        paths["output_npz"],
        patient_ids=patient_ids.astype(str),
        feature_names=feature_names.astype(str),
        x_ehr=x_ehr.astype(np.float32),
        g_ehr=g_ehr.astype(np.float32),
    )

    meta = {
        "stage81_npz": str(npz_path),
        "output_root": str(paths["root"]),
        "selected_patients": int(len(patient_ids)),
        "input_dim": int(x_ehr.shape[1]),
        "g_dim": int(args.g_dim),
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "val_ratio": float(args.val_ratio),
        "best_epoch": int(train_result["best_epoch"]),
        "best_monitor_loss": float(train_result["best_metric"]),
        "stopped_early": bool(train_result["stopped_early"]),
        "categorical_encoding_from_stage81": str(categorical_encoding),
    }
    write_meta_json(paths["meta_json"], meta)

    print(f"wrote_model: {paths['model_path']}")
    print(f"wrote: {paths['train_csv']}")
    print(f"wrote: {paths['meta_json']}")
    print(f"wrote: {paths['g_ehr_csv']}")
    print(f"wrote: {paths['output_npz']}")
    print(f"best_epoch: {train_result['best_epoch']}")
    print(f"best_monitor_loss: {train_result['best_metric']}")
    print(f"stopped_early: {train_result['stopped_early']}")
    print(f"g_ehr_shape: {tuple(g_ehr.shape)}")
    print("complete")


if __name__ == "__main__":
    main()
