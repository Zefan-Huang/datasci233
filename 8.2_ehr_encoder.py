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

## again same thing, but different dataset it's EHR here, also MLP.

PRIMARY_STAGE81_NPZ = Path("output/stage8/8.1_clinical_feature_engineering/x_ehr_features.npz")
FALLBACK_STAGE81_NPZ = Path("output/stage8/8.1_clinical_feature_engineering_smoke/x_ehr_features.npz")
DEFAULT_OUTPUT_ROOT = Path("output/stage8/8.2_ehr_encoder")


def resolve_stage81_npz(path_arg):
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
    paths["train_dir"].mkdir(parents=True, exist_ok=True)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    paths["token_dir"].mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_stage81_npz(npz_path):
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
    if max_patients < 0:
        raise RuntimeError("max_patients must be >= 0")
    if max_patients == 0:
        return x_ehr, patient_ids
    n = min(max_patients, len(patient_ids))
    return x_ehr[:n], patient_ids[:n]


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


def build_dataloaders(x_ehr, train_idx, val_idx, batch_size):
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
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norm > 1e-8, norm, 1.0)
    out = mat / safe
    out[norm.flatten() <= 1e-8] = 0.0
    return out.astype(np.float32)


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_g_ehr_csv(path, patient_ids, g_ehr):
    fields = ["patient_id"] + [f"g_{i:03d}" for i in range(g_ehr.shape[1])]
    rows = []
    for i, pid in enumerate(patient_ids):
        row = {"patient_id": str(pid)}
        for j in range(g_ehr.shape[1]):
            row[f"g_{j:03d}"] = float(g_ehr[i, j])
        rows.append(row)
    write_csv(path, fields, rows)


def write_meta_json(path, meta):
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args():
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
    args = parse_args()
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
