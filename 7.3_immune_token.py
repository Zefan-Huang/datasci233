import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

## same thing here and using different dataset.

PRIMARY_STAGE71_NPZ = Path("output/stage7/7.1_rna_alignment/x_rna_log1p_zscore.npz")
FALLBACK_STAGE71_NPZ = Path("output/stage7/7.1_rna_alignment_smoke/x_rna_log1p_zscore.npz")
DEFAULT_OUTPUT_ROOT = Path("output/stage7/7.3_immune_token")

IMMUNE_MARKER_SETS = {
    "t_cell_core": ["915", "916", "914", "940"],
    "cd8_cytotoxic": ["925", "926", "3002", "5551", "4818"],
    "nk_cell": ["4818", "5551", "3002", "9437", "6402"],
    "b_cell": ["931", "973", "974", "933", "930"],
    "antigen_presentation": ["3122", "3113", "3117", "3105", "3119"],
    "myeloid": ["3684", "2214", "7940", "958", "929"],
    "macrophage": ["968", "929", "4057", "366", "4123"],
    "neutrophil": ["6279", "6280", "2215", "1003", "3688"],
    "ifn_gamma_response": ["3458", "3627", "4283", "6772", "3659"],
    "checkpoint_exhaustion": ["5133", "1493", "3902", "84868", "201633"],
    "treg": ["50943", "3559", "22807", "100506742", "941"],
    "stromal_tgf_beta": ["7040", "7422", "1277", "1278", "1281", "2191", "59"],
    "proliferation": ["4288", "983", "7153", "10232", "4171"],
}



def resolve_stage71_npz(path_arg):
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
    paths["train_dir"].mkdir(parents=True, exist_ok=True)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    paths["token_dir"].mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_gene_id(raw):
    txt = str(raw).strip()
    if txt.endswith(".0"):
        txt = txt[:-2]
    return txt


def load_stage71_npz(npz_path):
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
    if max_patients < 0:
        raise RuntimeError("max_patients must be >= 0")
    if max_patients == 0:
        return x_rna, patient_ids
    n = min(max_patients, len(patient_ids))
    return x_rna[:n], patient_ids[:n]


def build_gene_index(gene_ids):
    out = {}
    for i, gid in enumerate(gene_ids.tolist()):
        if gid and gid not in out:
            out[gid] = int(i)
    return out


def compute_immune_signatures(x_rna, gene_ids, marker_sets):
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
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    safe = np.where(std > 1e-8, std, 1.0)
    z = (mat - mean) / safe
    const_mask = std <= 1e-8
    if const_mask.any():
        z[:, const_mask] = 0.0
    return z.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def split_train_val_indices(n_samples, val_ratio, seed):
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
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norm > 1e-8, norm, 1.0)
    out = mat / safe
    out[norm.flatten() <= 1e-8] = 0.0
    return out.astype(np.float32)


def infer_t_imm(model, sig_z, device, infer_batch_size):
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_signature_csv(path, patient_ids, sig_names, sig_raw, sig_z):
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
    fields = ["patient_id"] + [f"t_imm_{i:03d}" for i in range(t_imm.shape[1])]
    rows = []
    for i, pid in enumerate(patient_ids):
        r = {"patient_id": str(pid)}
        for j in range(t_imm.shape[1]):
            r[f"t_imm_{j:03d}"] = float(t_imm[i, j])
        rows.append(r)
    write_csv(path, fields, rows)


def write_meta_json(path, meta):
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args():
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
    args = parse_args()
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
