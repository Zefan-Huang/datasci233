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

## using MLP to do encode RNA info

PRIMARY_STAGE71_NPZ = Path("output/stage7/7.1_rna_alignment/x_rna_log1p_zscore.npz")
FALLBACK_STAGE71_NPZ = Path("output/stage7/7.1_rna_alignment_smoke/x_rna_log1p_zscore.npz")
DEFAULT_OUTPUT_ROOT = Path("output/stage7/7.2_rna_encoder")



def resolve_input_npz(path_arg):
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
        "model_path": model_dir / "rna_encoder.pt",
        "train_csv": train_dir / "rna_encoder_training_summary.csv",
        "meta_json": train_dir / "rna_encoder_meta.json",
        "g_rna_csv": token_dir / "g_rna.csv",
        "t_rna_csv": token_dir / "t_rna_tokens.csv",
        "gene_sel_csv": token_dir / "rna_gene_selection.csv",
        "output_npz": token_dir / "rna_encoder_outputs.npz",
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


def load_stage71_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as z:
        required = ["x_rna", "patient_ids", "gene_ids"]
        for key in required:
            if key not in z:
                raise RuntimeError(f"stage7.1 npz missing key: {key}")

        x_rna = np.asarray(z["x_rna"], dtype=np.float32)
        patient_ids = np.asarray(z["patient_ids"]).astype(str)
        gene_ids = np.asarray(z["gene_ids"]).astype(str)

        gene_std_log1p = None
        if "gene_std_log1p" in z:
            gene_std_log1p = np.asarray(z["gene_std_log1p"], dtype=np.float32)

    if x_rna.ndim != 2:
        raise RuntimeError(f"x_rna must be 2D, got shape={x_rna.shape}")
    if x_rna.shape[0] != len(patient_ids):
        raise RuntimeError("x_rna row count mismatches patient_ids")
    if x_rna.shape[1] != len(gene_ids):
        raise RuntimeError("x_rna col count mismatches gene_ids")

    return {
        "x_rna": x_rna,
        "patient_ids": patient_ids,
        "gene_ids": gene_ids,
        "gene_std_log1p": gene_std_log1p,
    }


def apply_patient_limit(x_rna, patient_ids, max_patients):
    if max_patients < 0:
        raise RuntimeError("max_patients must be >= 0")
    if max_patients == 0:
        return x_rna, patient_ids

    n = min(max_patients, len(patient_ids))
    return x_rna[:n], patient_ids[:n]


def select_top_variable_genes(x_rna, gene_ids, gene_std_log1p, top_genes):
    gene_count = x_rna.shape[1]
    if top_genes < 0:
        raise RuntimeError("top_genes must be >= 0")

    if gene_std_log1p is not None and len(gene_std_log1p) == gene_count:
        score = np.asarray(gene_std_log1p, dtype=np.float64) ** 2.0
    else:
        score = np.var(x_rna, axis=0).astype(np.float64)

    if top_genes == 0 or top_genes >= gene_count:
        selected_idx = np.arange(gene_count, dtype=np.int64)
    else:
        order = np.argsort(-score)
        selected_idx = np.asarray(order[:top_genes], dtype=np.int64)

    selected_gene_ids = gene_ids[selected_idx]
    selected_score = score[selected_idx]
    x_selected = x_rna[:, selected_idx].astype(np.float32)
    return selected_idx, selected_gene_ids, x_selected, selected_score


def split_train_val_indices(n_samples, val_ratio, seed):
    if val_ratio < 0 or val_ratio >= 1:
        raise RuntimeError("val_ratio must be in [0,1)")

    indices = np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    if n_samples < 4 or val_ratio == 0:
        return indices.tolist(), []

    val_count = int(round(n_samples * val_ratio))
    val_count = max(1, min(val_count, n_samples - 2))

    val_idx = indices[:val_count].tolist()
    train_idx = indices[val_count:].tolist()
    return train_idx, val_idx


def build_dataloaders(x_selected, train_idx, val_idx, batch_size):
    if batch_size <= 0:
        raise RuntimeError("batch_size must be >= 1")

    train_x = torch.from_numpy(x_selected[train_idx]).float()
    train_ds = TensorDataset(train_x)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_idx:
        val_x = torch.from_numpy(x_selected[val_idx]).float()
        val_ds = TensorDataset(val_x)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class RNAEncoderMLP(nn.Module):
    def __init__(self, input_dim, g_dim, num_tokens, token_dim, dropout):
        super().__init__()
        token_hidden_dim = int(num_tokens) * int(token_dim)
        if token_hidden_dim <= 0:
            token_hidden_dim = max(int(g_dim) * 2, 64)

        self.input_dim = int(input_dim)
        self.g_dim = int(g_dim)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.token_hidden_dim = int(token_hidden_dim)

        self.enc_fc1 = nn.Linear(self.input_dim, self.token_hidden_dim)
        self.enc_fc2 = nn.Linear(self.token_hidden_dim, self.g_dim)
        self.drop = nn.Dropout(float(dropout))

        self.dec_fc1 = nn.Linear(self.g_dim, self.token_hidden_dim)
        self.dec_fc2 = nn.Linear(self.token_hidden_dim, self.input_dim)

    def encode(self, x):
        token_flat = torch.relu(self.enc_fc1(x))
        token_flat = self.drop(token_flat)
        g = torch.relu(self.enc_fc2(token_flat))
        return g, token_flat

    def decode(self, g):
        h = torch.relu(self.dec_fc1(g))
        recon = self.dec_fc2(h)
        return recon

    def forward(self, x):
        g, token_flat = self.encode(x)
        recon = self.decode(g)
        return recon, g, token_flat


def evaluate_loader_loss(model, loader, device):
    if loader is None:
        return None

    mse = nn.MSELoss(reduction="mean")
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, _g, _t = model(x)
            loss = mse(recon, x)
            loss_sum += float(loss.item())
            count += 1
    if count == 0:
        return None
    return loss_sum / max(count, 1)


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_rna_encoder(model, train_loader, val_loader, device, epochs, lr, weight_decay, early_stop_patience, early_stop_min_delta):
    if epochs <= 0:
        raise RuntimeError("epochs must be >= 1")
    if lr <= 0:
        raise RuntimeError("lr must be > 0")
    if early_stop_patience <= 0:
        raise RuntimeError("early_stop_patience must be >= 1")
    if early_stop_min_delta < 0:
        raise RuntimeError("early_stop_min_delta must be >= 0")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    mse = nn.MSELoss(reduction="mean")

    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_state = None
    history_rows = []
    stopped_early = False

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, _g, _t = model(x)
            loss = mse(recon, x)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_steps += 1

        train_loss = train_loss_sum / max(train_steps, 1)
        val_loss = evaluate_loader_loss(model, val_loader, device)
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
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": "" if val_loss is None else val_loss,
                "monitor_loss": monitor,
                "is_best_epoch": 1 if improved else 0,
                "bad_epochs": bad_epochs,
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
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "stopped_early": stopped_early,
        "history_rows": history_rows,
    }


def infer_embeddings(model, x_selected, device, infer_batch_size):
    if infer_batch_size <= 0:
        raise RuntimeError("infer_batch_size must be >= 1")

    model.eval()
    ds = TensorDataset(torch.from_numpy(x_selected).float())
    loader = DataLoader(ds, batch_size=infer_batch_size, shuffle=False)

    g_chunks = []
    t_chunks = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            _recon, g, token_flat = model(x)
            g_chunks.append(g.detach().cpu().numpy())
            t_chunks.append(token_flat.detach().cpu().numpy())

    g_rna = np.concatenate(g_chunks, axis=0).astype(np.float32)
    token_flat = np.concatenate(t_chunks, axis=0).astype(np.float32)
    return g_rna, token_flat


def l2_normalize_rows(mat):
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norm > 1e-8, norm, 1.0)
    out = mat / safe
    out[norm.flatten() <= 1e-8] = 0.0
    return out.astype(np.float32)


def build_t_rna_tokens(token_flat, num_tokens, token_dim):
    if num_tokens <= 0 or token_dim <= 0:
        return np.zeros((token_flat.shape[0], 0, 0), dtype=np.float32)

    expected = int(num_tokens) * int(token_dim)
    if token_flat.shape[1] != expected:
        raise RuntimeError(
            f"token_flat dim mismatch, got {token_flat.shape[1]}, expected {expected}"
        )

    t_rna = token_flat.reshape(token_flat.shape[0], int(num_tokens), int(token_dim))
    return t_rna.astype(np.float32)


def write_g_rna_csv(path, patient_ids, g_rna):
    fieldnames = ["patient_id"] + [f"g_{i:03d}" for i in range(g_rna.shape[1])]
    rows = []
    for i, patient_id in enumerate(patient_ids):
        row = {"patient_id": str(patient_id)}
        for j in range(g_rna.shape[1]):
            row[f"g_{j:03d}"] = float(g_rna[i, j])
        rows.append(row)
    write_csv(path, fieldnames, rows)


def write_t_rna_csv(path, patient_ids, t_rna):
    fieldnames = ["patient_id", "token_index", "token_dim", "token_json"]
    rows = []
    for i, patient_id in enumerate(patient_ids):
        for t_idx in range(t_rna.shape[1]):
            rows.append(
                {
                    "patient_id": str(patient_id),
                    "token_index": int(t_idx),
                    "token_dim": int(t_rna.shape[2]),
                    "token_json": json.dumps([float(x) for x in t_rna[i, t_idx].tolist()]),
                }
            )
    write_csv(path, fieldnames, rows)


def write_gene_selection_csv(path, selected_idx, selected_gene_ids, selected_score):
    fieldnames = ["rank", "gene_index", "gene_id", "variance_score"]
    rows = []
    for rank, (idx, gid, sc) in enumerate(zip(selected_idx, selected_gene_ids, selected_score), start=1):
        rows.append(
            {
                "rank": int(rank),
                "gene_index": int(idx),
                "gene_id": str(gid),
                "variance_score": float(sc),
            }
        )
    write_csv(path, fieldnames, rows)


def write_meta_json(path, meta):
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 7.2 RNA encoder: top-variance genes + MLP encoder (g_rna, optional T_rna).",
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
    parser.add_argument(
        "--top-genes",
        type=int,
        default=5000,
        help="0 means use all genes; >0 means keep top-N variable genes.",
    )
    parser.add_argument("--g-dim", type=int, default=128)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=64)
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
    args = parse_args()
    if args.max_patients < 0:
        raise SystemExit("--max-patients must be >= 0 (0 means all patients)")
    if args.top_genes < 0:
        raise SystemExit("--top-genes must be >= 0 (0 means all genes)")
    if args.g_dim <= 0:
        raise SystemExit("--g-dim must be >= 1")
    if args.num_tokens < 0:
        raise SystemExit("--num-tokens must be >= 0")
    if args.token_dim < 0:
        raise SystemExit("--token-dim must be >= 0")
    if args.dropout < 0 or args.dropout >= 1:
        raise SystemExit("--dropout must be in [0,1)")

    npz_path = resolve_input_npz(args.stage71_npz)
    paths = resolve_output_paths(args.output_root)
    ensure_output_dirs(paths)

    set_seed(args.seed)

    data = load_stage71_npz(npz_path)
    x_rna = data["x_rna"]
    patient_ids = data["patient_ids"]
    gene_ids = data["gene_ids"]

    x_rna, patient_ids = apply_patient_limit(x_rna, patient_ids, args.max_patients)
    total_patients = len(data["patient_ids"])

    selected_idx, selected_gene_ids, x_selected, selected_score = select_top_variable_genes(
        x_rna=x_rna,
        gene_ids=gene_ids,
        gene_std_log1p=data["gene_std_log1p"],
        top_genes=args.top_genes,
    )

    train_idx, val_idx = split_train_val_indices(
        n_samples=x_selected.shape[0],
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_loader, val_loader = build_dataloaders(
        x_selected=x_selected,
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
        f"[start] selected_genes={x_selected.shape[1]} total_genes={x_rna.shape[1]} "
        f"full_genes={1 if args.top_genes == 0 else 0}"
    )
    print(
        f"[start] g_dim={args.g_dim} num_tokens={args.num_tokens} token_dim={args.token_dim} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} val_ratio={args.val_ratio}"
    )
    print(
        f"[deps] torch={torch.__version__} cuda_available={torch.cuda.is_available()} numpy={np.__version__} device={device}"
    )

    model = RNAEncoderMLP(
        input_dim=x_selected.shape[1],
        g_dim=args.g_dim,
        num_tokens=args.num_tokens,
        token_dim=args.token_dim,
        dropout=args.dropout,
    ).to(device)

    print("[stage] train rna encoder")
    train_result = train_rna_encoder(
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
            "input_dim": int(x_selected.shape[1]),
            "g_dim": int(args.g_dim),
            "num_tokens": int(args.num_tokens),
            "token_dim": int(args.token_dim),
            "selected_gene_indices": selected_idx.astype(np.int64),
            "selected_gene_ids": selected_gene_ids.astype(str),
            "stage71_npz": str(npz_path),
        },
        str(paths["model_path"]),
    )

    history_rows = train_result["history_rows"]
    write_csv(
        paths["train_csv"],
        ["epoch", "train_loss", "val_loss", "monitor_loss", "is_best_epoch", "bad_epochs"],
        history_rows,
    )

    print("[stage] infer embeddings")
    g_rna, token_flat = infer_embeddings(
        model=model,
        x_selected=x_selected,
        device=device,
        infer_batch_size=args.infer_batch_size,
    )

    g_rna = l2_normalize_rows(g_rna)
    t_rna = build_t_rna_tokens(token_flat=token_flat, num_tokens=args.num_tokens, token_dim=args.token_dim)
    if t_rna.shape[1] > 0 and t_rna.shape[2] > 0:
        flat = t_rna.reshape(t_rna.shape[0] * t_rna.shape[1], t_rna.shape[2])
        flat = l2_normalize_rows(flat)
        t_rna = flat.reshape(t_rna.shape[0], t_rna.shape[1], t_rna.shape[2])

    write_g_rna_csv(paths["g_rna_csv"], patient_ids, g_rna)
    write_t_rna_csv(paths["t_rna_csv"], patient_ids, t_rna)
    write_gene_selection_csv(paths["gene_sel_csv"], selected_idx, selected_gene_ids, selected_score)

    np.savez_compressed(
        paths["output_npz"],
        patient_ids=patient_ids.astype(str),
        selected_gene_indices=selected_idx.astype(np.int64),
        selected_gene_ids=selected_gene_ids.astype(str),
        x_rna_selected=x_selected.astype(np.float32),
        g_rna=g_rna.astype(np.float32),
        t_rna=t_rna.astype(np.float32),
    )

    meta = {
        "stage71_npz": str(npz_path),
        "output_root": str(paths["root"]),
        "selected_patients": int(len(patient_ids)),
        "selected_genes": int(x_selected.shape[1]),
        "g_dim": int(args.g_dim),
        "num_tokens": int(args.num_tokens),
        "token_dim": int(args.token_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "val_ratio": float(args.val_ratio),
        "best_epoch": int(train_result["best_epoch"]),
        "best_monitor_loss": float(train_result["best_metric"]),
        "stopped_early": bool(train_result["stopped_early"]),
    }
    write_meta_json(paths["meta_json"], meta)

    print(f"wrote_model: {paths['model_path']}")
    print(f"wrote: {paths['train_csv']}")
    print(f"wrote: {paths['meta_json']}")
    print(f"wrote: {paths['g_rna_csv']}")
    print(f"wrote: {paths['t_rna_csv']}")
    print(f"wrote: {paths['gene_sel_csv']}")
    print(f"wrote: {paths['output_npz']}")
    print(f"best_epoch: {train_result['best_epoch']}")
    print(f"best_monitor_loss: {train_result['best_metric']}")
    print(f"stopped_early: {train_result['stopped_early']}")
    print(f"g_rna_shape: {tuple(g_rna.shape)}")
    print(f"t_rna_shape: {tuple(t_rna.shape)}")
    print("complete")


if __name__ == "__main__":
    main()
