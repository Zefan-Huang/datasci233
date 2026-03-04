"""
作用：
- 实现 project.md Stage 7.3：Immune / microenvironment token（derive from RNA first）。
- 先基于 RNA 计算 immune signatures（marker sets），再用 MLP 生成 t_imm。

输入：
- output/stage7/7.1_rna_alignment/x_rna_log1p_zscore.npz（优先）

输出：
- output/stage7/7.3_immune_token/model/immune_token_mlp.pt
- output/stage7/7.3_immune_token/train/immune_token_training_summary.csv
- output/stage7/7.3_immune_token/train/immune_token_meta.json
- output/stage7/7.3_immune_token/tokens/immune_signatures.csv
- output/stage7/7.3_immune_token/tokens/t_imm.csv
- output/stage7/7.3_immune_token/tokens/t_imm_outputs.npz
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


PRIMARY_STAGE71_NPZ = Path("output/stage7/7.1_rna_alignment/x_rna_log1p_zscore.npz")
FALLBACK_STAGE71_NPZ = Path("output/stage7/7.1_rna_alignment_smoke/x_rna_log1p_zscore.npz")
DEFAULT_OUTPUT_ROOT = Path("output/stage7/7.3_immune_token")

# marker sets 使用 Entrez GeneID（和当前 RNA 文件第一列一致）。
IMMUNE_MARKER_SETS = {
    "t_cell_core": ["915", "916", "914", "940"],  # CD3D/CD3E/CD2/CD27
    "cd8_cytotoxic": ["925", "926", "3002", "5551", "4818"],  # CD8A/B, GZMB, PRF1, NKG7
    "nk_cell": ["4818", "5551", "3002", "9437", "6402"],  # NKG7, PRF1, GZMB, NCR1, SELPLG
    "b_cell": ["931", "973", "974", "933", "930"],  # MS4A1, CD79A/B, CD22, CD19
    "antigen_presentation": ["3122", "3113", "3117", "3105", "3119"],  # HLA-DRA/DPA1/DPB1/DRB1/DQA1
    "myeloid": ["3684", "2214", "7940", "958", "929"],  # ITGAM, FCGR3A, LST1, CD14, CD163
    "macrophage": ["968", "929", "4057", "366", "4123"],  # CD68, CD163, CCL2, AIF1, MRC1
    "neutrophil": ["6279", "6280", "2215", "1003", "3688"],  # S100A8/A9, FCGR3B, CD24, ITGAX
    "ifn_gamma_response": ["3458", "3627", "4283", "6772", "3659"],  # IFNG, CXCL10, CXCL9, STAT1, IRF1
    "checkpoint_exhaustion": ["5133", "1493", "3902", "84868", "201633"],  # PDCD1, CTLA4, LAG3, HAVCR2, TIGIT
    "treg": ["50943", "3559", "22807", "100506742", "941"],  # FOXP3, IL2RA, IKZF2, CTLA4-AS? + CD28
    "stromal_tgf_beta": ["7040", "7422", "1277", "1278", "1281", "2191", "59"],  # TGFB1, VEGFA, collagens, FAP, ACTA2
    "proliferation": ["4288", "983", "7153", "10232", "4171"],  # MKI67, CDK1, TOP2A, MCM7, MCM2
}


def check_dependencies():
    """Why: Stage 7.3 训练与矩阵运算依赖 numpy/torch，需提前检查。

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


def resolve_stage71_npz(path_arg):
    """Why: Stage 7.3 必须读取 7.1 的 RNA 矩阵，需稳定解析输入路径。

    Content: 优先命令行路径，否则按默认与回退路径查找。
    Input: path_arg。
    Output: 可用 NPZ 路径。
    """
    if path_arg.strip():
        p = Path(path_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"stage7.1 npz not found: {p}")
    if PRIMARY_STAGE71_NPZ.exists():
        return PRIMARY_STAGE71_NPZ
    if FALLBACK_STAGE71_NPZ.exists():
        return FALLBACK_STAGE71_NPZ
    raise FileNotFoundError(
        "stage7.1 npz not found. expected one of: "
        + f"{PRIMARY_STAGE71_NPZ}, {FALLBACK_STAGE71_NPZ}"
    )


def resolve_output_paths(output_root):
    """Why: 训练日志、模型、token 需要分目录保存，便于管理。

    Content: 组装 Stage 7.3 输出路径。
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
        "model_path": model_dir / "immune_token_mlp.pt",
        "train_csv": train_dir / "immune_token_training_summary.csv",
        "meta_json": train_dir / "immune_token_meta.json",
        "signature_csv": token_dir / "immune_signatures.csv",
        "t_imm_csv": token_dir / "t_imm.csv",
        "output_npz": token_dir / "t_imm_outputs.npz",
    }


def ensure_output_dirs(paths):
    """Why: 输出前必须确保目录已存在，避免写文件失败。

    Content: 创建 train/model/tokens 目录。
    Input: paths。
    Output: 目录创建完成。
    """
    paths["train_dir"].mkdir(parents=True, exist_ok=True)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    paths["token_dir"].mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    """Why: 固定随机种子可减少训练波动，便于复现。

    Content: 设置 python/numpy/torch 随机状态。
    Input: seed。
    Output: 随机状态已设置。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_gene_id(raw):
    """Why: RNA gene_id 可能有小数或空白，需统一成可匹配格式。

    Content: 去空白并去掉 .0 后缀。
    Input: raw。
    Output: 规范化 gene_id 字符串。
    """
    txt = str(raw).strip()
    if txt.endswith(".0"):
        txt = txt[:-2]
    return txt


def load_stage71_npz(npz_path):
    """Why: 7.3 需要 x_rna、patient_ids、gene_ids 作为输入。

    Content: 读取并校验 7.1 NPZ 的关键字段。
    Input: npz_path。
    Output: 数据字典。
    """
    with np.load(npz_path, allow_pickle=True) as z:
        required = ["x_rna", "patient_ids", "gene_ids"]
        for k in required:
            if k not in z:
                raise RuntimeError(f"stage7.1 npz missing key: {k}")

        x_rna = np.asarray(z["x_rna"], dtype=np.float32)
        patient_ids = np.asarray(z["patient_ids"]).astype(str)
        gene_ids_raw = np.asarray(z["gene_ids"]).astype(str)

    if x_rna.ndim != 2:
        raise RuntimeError(f"x_rna must be 2D, got shape={x_rna.shape}")
    if x_rna.shape[0] != len(patient_ids):
        raise RuntimeError("x_rna row count mismatches patient_ids")
    if x_rna.shape[1] != len(gene_ids_raw):
        raise RuntimeError("x_rna col count mismatches gene_ids")

    gene_ids = np.asarray([normalize_gene_id(g) for g in gene_ids_raw])
    return {
        "x_rna": x_rna,
        "patient_ids": patient_ids,
        "gene_ids": gene_ids,
    }


def apply_patient_limit(x_rna, patient_ids, max_patients):
    """Why: 便于快速调试 N 例，但默认应支持全量。

    Content: 根据 max_patients 截断样本；0 表示全量。
    Input: x_rna、patient_ids、max_patients。
    Output: 截断后的 x_rna 与 patient_ids。
    """
    if max_patients < 0:
        raise RuntimeError("max_patients must be >= 0")
    if max_patients == 0:
        return x_rna, patient_ids
    n = min(max_patients, len(patient_ids))
    return x_rna[:n], patient_ids[:n]


def build_gene_index(gene_ids):
    """Why: signature 计算需要快速把 marker gene 映射到列索引。

    Content: 构建 gene_id -> index 字典。
    Input: gene_ids。
    Output: 映射字典。
    """
    out = {}
    for i, gid in enumerate(gene_ids.tolist()):
        if gid and gid not in out:
            out[gid] = int(i)
    return out


def compute_immune_signatures(x_rna, gene_ids, marker_sets):
    """Why: Stage 7.3 第一步是从 RNA 计算 immune signatures。

    Content: 对每个 marker set 取对应基因的均值，得到 sample x signature 矩阵。
    Input: x_rna、gene_ids、marker_sets。
    Output: signatures、signature_names、signature_meta_rows。
    """
    gene_index = build_gene_index(gene_ids)
    names = list(marker_sets.keys())
    n = x_rna.shape[0]
    s = len(names)
    sig_mat = np.zeros((n, s), dtype=np.float32)
    meta_rows = []

    for j, name in enumerate(names):
        markers = [normalize_gene_id(x) for x in marker_sets[name]]
        idx = [gene_index[g] for g in markers if g in gene_index]

        if len(idx) > 0:
            sig_mat[:, j] = x_rna[:, idx].mean(axis=1).astype(np.float32)
        else:
            sig_mat[:, j] = 0.0

        meta_rows.append(
            {
                "signature_name": name,
                "marker_count_total": int(len(markers)),
                "marker_count_used": int(len(idx)),
                "marker_gene_ids_used": json.dumps([gene_ids[k] for k in idx]),
            }
        )

    return sig_mat, names, meta_rows


def zscore_columns(mat):
    """Why: MLP 输入尺度需要统一，避免某些 signature 主导训练。

    Content: 对每列做 z-score；常量列置 0。
    Input: mat。
    Output: (z_mat, col_mean, col_std)。
    """
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    safe = np.where(std > 1e-8, std, 1.0)
    z = (mat - mean) / safe
    const_mask = std <= 1e-8
    if const_mask.any():
        z[:, const_mask] = 0.0
    return z.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def split_train_val_indices(n_samples, val_ratio, seed):
    """Why: 训练需要验证集监控过拟合并支持早停。

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


def build_dataloaders(sig_z, train_idx, val_idx, batch_size):
    """Why: 训练和验证都要统一批处理流程。

    Content: 基于索引构建 DataLoader。
    Input: sig_z、train_idx、val_idx、batch_size。
    Output: train_loader、val_loader。
    """
    if batch_size <= 0:
        raise RuntimeError("batch_size must be >= 1")

    train_x = torch.from_numpy(sig_z[train_idx]).float()
    train_loader = DataLoader(TensorDataset(train_x), batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_idx:
        val_x = torch.from_numpy(sig_z[val_idx]).float()
        val_loader = DataLoader(TensorDataset(val_x), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class ImmuneTokenMLP(nn.Module):
    """Why: Stage 7.3 需要 MLP 把 signatures 编码成 t_imm token。

    Content: 编码器输出 t_imm，解码器重建 signature 作为训练约束。
    Input: signature 向量 [B,S]。
    Output: recon [B,S], t_imm [B,d]。
    """

    def __init__(self, input_dim, token_dim, hidden_dim, dropout):
        super().__init__()
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)

        self.enc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.enc2 = nn.Linear(self.hidden_dim, self.token_dim)
        self.drop = nn.Dropout(float(dropout))

        self.dec1 = nn.Linear(self.token_dim, self.hidden_dim)
        self.dec2 = nn.Linear(self.hidden_dim, self.input_dim)

    def encode(self, x):
        h = torch.relu(self.enc1(x))
        h = self.drop(h)
        t = torch.relu(self.enc2(h))
        return t

    def decode(self, t):
        h = torch.relu(self.dec1(t))
        recon = self.dec2(h)
        return recon

    def forward(self, x):
        t = self.encode(x)
        recon = self.decode(t)
        return recon, t


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
    s = 0.0
    c = 0
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, _t = model(x)
            loss = mse(recon, x)
            s += float(loss.item())
            c += 1
    if c == 0:
        return None
    return s / max(c, 1)


def train_model(model, train_loader, val_loader, device, epochs, lr, weight_decay, early_stop_patience, early_stop_min_delta):
    """Why: 需要训练 MLP 参数，才能得到稳定的 t_imm 表征。

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
    rows = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        s = 0.0
        c = 0
        for (x,) in train_loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            recon, _t = model(x)
            loss = mse(recon, x)
            loss.backward()
            opt.step()
            s += float(loss.item())
            c += 1

        train_loss = s / max(c, 1)
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

        rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": "" if val_loss is None else float(val_loss),
                "monitor_loss": float(monitor),
                "is_best_epoch": 1 if improved else 0,
                "bad_epochs": int(bad_epochs),
            }
        )

        val_txt = "NA" if val_loss is None else f"{val_loss:.6f}"
        print(
            f"[train] epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_txt} "
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
        "history_rows": rows,
    }


def l2_normalize_rows(mat):
    """Why: token 向量尺度要稳定，便于后续融合。

    Content: 对每行做 L2 归一化，零向量保持零。
    Input: mat。
    Output: 归一化矩阵。
    """
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norm > 1e-8, norm, 1.0)
    out = mat / safe
    out[norm.flatten() <= 1e-8] = 0.0
    return out.astype(np.float32)


def infer_t_imm(model, sig_z, device, infer_batch_size):
    """Why: 训练后需为所有病人导出最终 t_imm token。

    Content: 批量前向只取 encoder 输出。
    Input: model、sig_z、device、infer_batch_size。
    Output: t_imm 矩阵。
    """
    if infer_batch_size <= 0:
        raise RuntimeError("infer_batch_size must be >= 1")

    ds = TensorDataset(torch.from_numpy(sig_z).float())
    loader = DataLoader(ds, batch_size=infer_batch_size, shuffle=False)

    model.eval()
    chunks = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            _recon, t = model(x)
            chunks.append(t.detach().cpu().numpy())

    t_imm = np.concatenate(chunks, axis=0).astype(np.float32)
    return t_imm


def write_csv(path, fieldnames, rows):
    """Why: 训练日志和 token 表需要固定结构，便于后续读取。

    Content: 按字段顺序写 CSV。
    Input: path、fieldnames、rows。
    Output: CSV 文件。
    """
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_signature_csv(path, patient_ids, sig_names, sig_raw, sig_z):
    """Why: Stage 7.3 需要保留 signature 结果，便于解释 t_imm 来源。

    Content: 写 patient x signature（raw/z）到 CSV。
    Input: path、patient_ids、sig_names、sig_raw、sig_z。
    Output: immune_signatures.csv。
    """
    fields = ["patient_id"]
    for name in sig_names:
        fields.append(f"raw_{name}")
    for name in sig_names:
        fields.append(f"z_{name}")

    rows = []
    for i, pid in enumerate(patient_ids):
        r = {"patient_id": str(pid)}
        for j, name in enumerate(sig_names):
            r[f"raw_{name}"] = float(sig_raw[i, j])
        for j, name in enumerate(sig_names):
            r[f"z_{name}"] = float(sig_z[i, j])
        rows.append(r)

    write_csv(path, fields, rows)


def write_t_imm_csv(path, patient_ids, t_imm):
    """Why: 下游融合常用 CSV，需直接提供病人级 token 表。

    Content: 写 patient_id + t_imm 各维度。
    Input: path、patient_ids、t_imm。
    Output: t_imm.csv。
    """
    fields = ["patient_id"] + [f"t_imm_{i:03d}" for i in range(t_imm.shape[1])]
    rows = []
    for i, pid in enumerate(patient_ids):
        r = {"patient_id": str(pid)}
        for j in range(t_imm.shape[1]):
            r[f"t_imm_{j:03d}"] = float(t_imm[i, j])
        rows.append(r)
    write_csv(path, fields, rows)


def write_meta_json(path, meta):
    """Why: 训练配置和关键统计需要落盘，方便复现和审计。

    Content: 写 JSON。
    Input: path、meta。
    Output: meta.json。
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args():
    """Why: 需要可调超参，便于后续实验扩展。

    Content: 解析参数并忽略 IDE 注入未知参数。
    Input: 命令行参数。
    Output: 参数对象。
    """
    parser = argparse.ArgumentParser(
        description="Stage 7.3 immune token: marker-set signatures + MLP token encoder.",
        allow_abbrev=False,
    )
    parser.add_argument("--stage71-npz", type=str, default="")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="0 means all patients; >0 means use first N patients.",
    )
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--infer-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"ignore_unknown_args: {unknown}")
    return args


def main():
    """Why: 一条命令跑完 Stage 7.3，直接产出 t_imm。

    Content: 读 7.1 -> 算 signatures -> 训练 MLP -> 导出 t_imm 与日志。
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
    if args.token_dim <= 0:
        raise SystemExit("--token-dim must be >= 1")
    if args.hidden_dim <= 0:
        raise SystemExit("--hidden-dim must be >= 1")
    if args.dropout < 0 or args.dropout >= 1:
        raise SystemExit("--dropout must be in [0,1)")

    npz_path = resolve_stage71_npz(args.stage71_npz)
    paths = resolve_output_paths(args.output_root)
    ensure_output_dirs(paths)

    set_seed(args.seed)

    data = load_stage71_npz(npz_path)
    x_rna = data["x_rna"]
    patient_ids = data["patient_ids"]
    gene_ids = data["gene_ids"]

    total_patients = len(patient_ids)
    x_rna, patient_ids = apply_patient_limit(x_rna, patient_ids, args.max_patients)

    sig_raw, sig_names, sig_meta_rows = compute_immune_signatures(
        x_rna=x_rna,
        gene_ids=gene_ids,
        marker_sets=IMMUNE_MARKER_SETS,
    )
    sig_z, sig_mean, sig_std = zscore_columns(sig_raw)

    train_idx, val_idx = split_train_val_indices(
        n_samples=sig_z.shape[0],
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_loader, val_loader = build_dataloaders(
        sig_z=sig_z,
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

    print(f"[start] stage71_npz={npz_path}")
    print(f"[start] output_root={paths['root']}")
    print(
        f"[start] selected_patients={len(patient_ids)} total_patients={total_patients} "
        f"full_patients={1 if args.max_patients == 0 else 0}"
    )
    print(
        f"[start] signature_count={len(sig_names)} token_dim={args.token_dim} hidden_dim={args.hidden_dim} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} val_ratio={args.val_ratio}"
    )
    print(
        f"[deps] torch={torch.__version__} cuda_available={torch.cuda.is_available()} numpy={np.__version__} device={device}"
    )

    model = ImmuneTokenMLP(
        input_dim=sig_z.shape[1],
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    print("[stage] train immune token mlp")
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
            "input_dim": int(sig_z.shape[1]),
            "token_dim": int(args.token_dim),
            "hidden_dim": int(args.hidden_dim),
            "signature_names": np.asarray(sig_names),
            "signature_mean": sig_mean,
            "signature_std": sig_std,
            "marker_sets": IMMUNE_MARKER_SETS,
            "stage71_npz": str(npz_path),
        },
        str(paths["model_path"]),
    )

    write_csv(
        paths["train_csv"],
        ["epoch", "train_loss", "val_loss", "monitor_loss", "is_best_epoch", "bad_epochs"],
        train_result["history_rows"],
    )

    t_imm = infer_t_imm(
        model=model,
        sig_z=sig_z,
        device=device,
        infer_batch_size=args.infer_batch_size,
    )
    t_imm = l2_normalize_rows(t_imm)

    write_signature_csv(
        paths["signature_csv"],
        patient_ids=patient_ids,
        sig_names=sig_names,
        sig_raw=sig_raw,
        sig_z=sig_z,
    )
    write_t_imm_csv(paths["t_imm_csv"], patient_ids, t_imm)

    np.savez_compressed(
        paths["output_npz"],
        patient_ids=patient_ids.astype(str),
        signature_names=np.asarray(sig_names),
        immune_signatures_raw=sig_raw.astype(np.float32),
        immune_signatures_z=sig_z.astype(np.float32),
        t_imm=t_imm.astype(np.float32),
        signature_mean=sig_mean.astype(np.float32),
        signature_std=sig_std.astype(np.float32),
    )

    marker_used_min = min(r["marker_count_used"] for r in sig_meta_rows) if sig_meta_rows else 0
    marker_used_max = max(r["marker_count_used"] for r in sig_meta_rows) if sig_meta_rows else 0

    meta = {
        "stage71_npz": str(npz_path),
        "output_root": str(paths["root"]),
        "selected_patients": int(len(patient_ids)),
        "signature_count": int(len(sig_names)),
        "token_dim": int(args.token_dim),
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "val_ratio": float(args.val_ratio),
        "best_epoch": int(train_result["best_epoch"]),
        "best_monitor_loss": float(train_result["best_metric"]),
        "stopped_early": bool(train_result["stopped_early"]),
        "marker_count_used_min": int(marker_used_min),
        "marker_count_used_max": int(marker_used_max),
        "signature_meta": sig_meta_rows,
    }
    write_meta_json(paths["meta_json"], meta)

    print(f"wrote_model: {paths['model_path']}")
    print(f"wrote: {paths['train_csv']}")
    print(f"wrote: {paths['meta_json']}")
    print(f"wrote: {paths['signature_csv']}")
    print(f"wrote: {paths['t_imm_csv']}")
    print(f"wrote: {paths['output_npz']}")
    print(f"best_epoch: {train_result['best_epoch']}")
    print(f"best_monitor_loss: {train_result['best_metric']}")
    print(f"stopped_early: {train_result['stopped_early']}")
    print(f"t_imm_shape: {tuple(t_imm.shape)}")
    print("complete")


if __name__ == "__main__":
    main()
