import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_STAGE11_PACK = Path("output/stage11/11.2_graph_reasoning/graph_reasoning_pack.npz")
DEFAULT_LABELS_CSV = Path("output/labels_time_zero.csv")
FALLBACK_LABELS_CSV = Path("output/patient_manifest.csv")
DEFAULT_OUTPUT_ROOT = Path("output/stage12/12.1_primary_outputs")
DEFAULT_RECURRENCE_CLASSES = ["local", "regional", "distant"]



def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def resolve_required_path(path_arg, default_path, label):
    path = Path(path_arg) if str(path_arg).strip() else Path(default_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_labels_path(path_arg):
    if str(path_arg).strip():
        return Path(path_arg)
    if DEFAULT_LABELS_CSV.exists():
        return DEFAULT_LABELS_CSV
    if FALLBACK_LABELS_CSV.exists():
        return FALLBACK_LABELS_CSV
    raise FileNotFoundError(
        f"labels csv not found: {DEFAULT_LABELS_CSV} | {FALLBACK_LABELS_CSV}"
    )


def ensure_output_dirs(output_root):
    root = Path(output_root)
    train_dir = root / "train"
    model_dir = root / "model"
    pred_dir = root / "pred"
    train_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "train_dir": train_dir,
        "model_dir": model_dir,
        "pred_dir": pred_dir,
        "model_path": model_dir / "primary_heads.pt",
        "train_csv": train_dir / "primary_training_summary.csv",
        "meta_json": train_dir / "primary_meta.json",
        "pred_csv": pred_dir / "patient_primary_predictions.csv",
        "pred_npz": pred_dir / "primary_output_pack.npz",
    }


def normalize_missing_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"", "na", "n/a", "null", "not collected", "not recorded in database"}:
        return ""
    return text


def parse_optional_int(value):
    text = normalize_missing_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def parse_optional_float(value):
    text = normalize_missing_text(value)
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def normalize_recurrence_location(value):
    text = normalize_missing_text(value).lower()
    if not text:
        return ""
    return text


def load_stage11_pack(npz_path):
    with np.load(npz_path, allow_pickle=True) as z:
        out = {}
        for key in z.files:
            value = z[key]
            if value.dtype.kind in {"U", "O"}:
                out[key] = value.astype(str)
            else:
                out[key] = value
    required = {"patient_ids", "organ_node_names", "Z_prime"}
    missing = required - set(out.keys())
    if missing:
        raise RuntimeError(f"stage11 pack missing required arrays: {sorted(missing)}")
    return out


def load_label_rows(labels_csv_path):
    with Path(labels_csv_path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "patient_id" not in fieldnames:
            raise RuntimeError(f"labels csv missing patient_id column: {labels_csv_path}")
        rows = {}
        for row in reader:
            patient_id = normalize_missing_text(row.get("patient_id"))
            if not patient_id:
                continue
            rows[patient_id] = row
    return rows


def resolve_recurrence_classes(labels_by_patient, classes_arg):
    if str(classes_arg).strip():
        classes = [x.strip().lower() for x in str(classes_arg).split(",") if x.strip()]
        if not classes:
            raise RuntimeError("--recurrence-classes must not be empty when provided")
        return classes

    observed = []
    seen = set()
    for name in DEFAULT_RECURRENCE_CLASSES:
        if name not in seen:
            seen.add(name)
            observed.append(name)
    for row in labels_by_patient.values():
        label = normalize_recurrence_location(row.get("rec_location_class"))
        if label and label not in seen:
            seen.add(label)
            observed.append(label)
    return observed


def build_supervision_arrays(patient_ids, labels_by_patient, recurrence_classes):
    class_to_index = {name: idx for idx, name in enumerate(recurrence_classes)}

    event_os = np.zeros((len(patient_ids),), dtype=np.float32)
    time_os_days = np.zeros((len(patient_ids),), dtype=np.float32)
    os_label_known = np.zeros((len(patient_ids),), dtype=np.uint8)

    event_rec = np.zeros((len(patient_ids),), dtype=np.float32)
    time_rec_days = np.zeros((len(patient_ids),), dtype=np.float32)
    rec_label_known = np.zeros((len(patient_ids),), dtype=np.uint8)
    rec_location_index = np.full((len(patient_ids),), -1, dtype=np.int64)
    rec_location_known = np.zeros((len(patient_ids),), dtype=np.uint8)

    missing_patients = []
    for idx, patient_id in enumerate(patient_ids):
        row = labels_by_patient.get(patient_id)
        if row is None:
            missing_patients.append(patient_id)
            continue

        time_os = parse_optional_float(row.get("time_os_days", row.get("time_os")))
        event_os_value = parse_optional_int(row.get("event_os"))
        os_known_value = parse_optional_int(row.get("os_label_known"))
        if os_known_value is None:
            os_known_value = 1 if (time_os is not None and event_os_value is not None) else 0
        if os_known_value == 1 and time_os is not None and event_os_value is not None:
            os_label_known[idx] = 1
            time_os_days[idx] = float(time_os)
            event_os[idx] = float(event_os_value)

        time_rec = parse_optional_float(row.get("time_rec_days", row.get("time_rec")))
        event_rec_value = parse_optional_int(row.get("event_rec"))
        rec_known_value = parse_optional_int(row.get("rec_label_known"))
        if rec_known_value is None:
            rec_known_value = 1 if event_rec_value is not None else 0
        if rec_known_value == 1 and event_rec_value is not None and time_rec is not None:
            rec_label_known[idx] = 1
            event_rec[idx] = float(event_rec_value)
            time_rec_days[idx] = float(time_rec)

        location = normalize_recurrence_location(row.get("rec_location_class"))
        if rec_label_known[idx] == 1 and int(event_rec[idx]) == 1 and location in class_to_index:
            rec_location_index[idx] = int(class_to_index[location])
            rec_location_known[idx] = 1

    if missing_patients:
        raise RuntimeError(
            "labels missing for stage11 patients: "
            + ",".join(missing_patients[:10])
            + ("..." if len(missing_patients) > 10 else "")
        )

    return {
        "event_os": event_os,
        "time_os_days": time_os_days,
        "os_label_known": os_label_known,
        "event_rec": event_rec,
        "time_rec_days": time_rec_days,
        "rec_label_known": rec_label_known,
        "rec_location_index": rec_location_index,
        "rec_location_known": rec_location_known,
    }


def stratified_split_indices(strata, val_ratio, seed):
    if not 0.0 <= float(val_ratio) < 1.0:
        raise RuntimeError("val_ratio must be in [0,1)")

    rng = np.random.RandomState(int(seed))
    groups = defaultdict(list)
    for idx, key in enumerate(strata):
        groups[str(key)].append(idx)

    train_indices = []
    val_indices = []
    for key in sorted(groups.keys()):
        idxs = groups[key]
        idxs = list(np.asarray(idxs)[rng.permutation(len(idxs))].tolist())
        if float(val_ratio) <= 0.0 or len(idxs) <= 1:
            train_indices.extend(idxs)
            continue
        n_val = int(round(len(idxs) * float(val_ratio)))
        n_val = max(1, n_val)
        n_val = min(n_val, len(idxs) - 1)
        val_indices.extend(idxs[:n_val])
        train_indices.extend(idxs[n_val:])

    train_indices = np.asarray(sorted(train_indices), dtype=np.int64)
    val_indices = np.asarray(sorted(val_indices), dtype=np.int64)
    return train_indices, val_indices


def parse_seed_list(seed_arg, default_seed):
    text = str(seed_arg).strip()
    if not text:
        return [int(default_seed)]
    out = []
    seen = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if not out:
        out.append(int(default_seed))
    return out


def stratified_kfold_splits(strata, num_folds, seed):
    if int(num_folds) < 2:
        raise RuntimeError("num_folds must be >= 2")

    rng = np.random.RandomState(int(seed))
    groups = defaultdict(list)
    for idx, key in enumerate(strata):
        groups[str(key)].append(idx)

    fold_values = [[] for _ in range(int(num_folds))]
    for key in sorted(groups.keys()):
        idxs = np.asarray(groups[key], dtype=np.int64)
        idxs = idxs[rng.permutation(len(idxs))]
        for pos, idx in enumerate(idxs.tolist()):
            fold_values[pos % int(num_folds)].append(int(idx))

    all_indices = np.arange(len(strata), dtype=np.int64)
    splits = []
    for fold_idx in range(int(num_folds)):
        val_indices = np.asarray(sorted(fold_values[fold_idx]), dtype=np.int64)
        val_mask = np.zeros((len(strata),), dtype=np.uint8)
        val_mask[val_indices] = 1
        train_indices = all_indices[val_mask == 0]
        if train_indices.size == 0 or val_indices.size == 0:
            raise RuntimeError(
                f"invalid k-fold split for fold={fold_idx}: "
                f"train={train_indices.size} val={val_indices.size}"
            )
        splits.append((train_indices, val_indices))
    return splits


def build_split_strata(supervision):
    strata = []
    for os_known, event_os, rec_known, event_rec, loc_known, loc_idx in zip(
        supervision["os_label_known"].tolist(),
        supervision["event_os"].tolist(),
        supervision["rec_label_known"].tolist(),
        supervision["event_rec"].tolist(),
        supervision["rec_location_known"].tolist(),
        supervision["rec_location_index"].tolist(),
    ):
        os_part = "u" if int(os_known) == 0 else str(int(event_os))
        rec_part = "u" if int(rec_known) == 0 else str(int(event_rec))
        loc_part = "u" if int(loc_known) == 0 else str(int(loc_idx))
        strata.append(f"os{os_part}_rec{rec_part}_loc{loc_part}")
    return strata


def build_time_bin_edges(train_time_days, train_event_os, num_bins):
    if int(num_bins) <= 0:
        raise RuntimeError("num_bins must be > 0 for discrete survival")
    valid_time = train_time_days[np.isfinite(train_time_days) & (train_time_days > 0)]
    if valid_time.size == 0:
        raise RuntimeError("no valid train_time_days available for discrete survival")
    event_time = train_time_days[
        np.isfinite(train_time_days) & (train_time_days > 0) & (train_event_os > 0.5)
    ]
    source = event_time if event_time.size >= max(4, int(num_bins)) else valid_time
    edges = np.quantile(source, np.linspace(0.0, 1.0, int(num_bins) + 1)).astype(np.float32)
    edges[0] = 0.0
    edges[-1] = max(float(edges[-1]), float(valid_time.max()), 1.0)
    edges = np.maximum.accumulate(edges)
    for idx in range(1, len(edges)):
        if float(edges[idx]) <= float(edges[idx - 1]):
            edges[idx] = float(edges[idx - 1]) + 1.0
    return edges.astype(np.float32)


def choose_device(device_arg):
    value = str(device_arg).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def snapshot_state_dict(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def restore_state_dict(module, state_dict):
    module.load_state_dict(state_dict)


def compute_binary_pos_weight(target, known_mask):
    target_known = target[known_mask]
    positives = float((target_known > 0.5).sum().item())
    negatives = float((target_known <= 0.5).sum().item())
    if positives <= 0.0:
        return 1.0
    return max(negatives / positives, 1.0)


def compute_multiclass_weights(target, known_mask, num_classes):
    weights = torch.ones((int(num_classes),), dtype=torch.float32, device=target.device)
    known_target = target[known_mask]
    if known_target.numel() == 0:
        return weights
    counts = torch.bincount(known_target, minlength=int(num_classes)).float()
    nonzero = counts > 0
    if torch.any(nonzero):
        weights[nonzero] = counts[nonzero].sum() / (counts[nonzero] * float(nonzero.sum()))
    return weights


def cox_partial_log_likelihood(log_risk, time_days, event_os, known_mask):
    valid = known_mask & torch.isfinite(time_days) & (time_days > 0)
    if int(valid.sum().item()) == 0:
        return log_risk.sum() * 0.0

    risk = log_risk[valid]
    time = time_days[valid]
    event = event_os[valid] > 0.5
    if int(event.sum().item()) == 0:
        return risk.sum() * 0.0

    unique_event_times = torch.unique(time[event])
    total = risk.new_tensor(0.0)
    num_events = 0
    for event_time in unique_event_times:
        event_mask = (time == event_time) & event
        if int(event_mask.sum().item()) == 0:
            continue
        risk_set_mask = time >= event_time
        denom = torch.exp(risk[risk_set_mask]).sum().clamp_min(1e-12)
        total = total + risk[event_mask].sum() - float(event_mask.sum().item()) * torch.log(denom)
        num_events += int(event_mask.sum().item())
    if num_events <= 0:
        return risk.sum() * 0.0
    return -total / float(num_events)


def assign_time_bins_torch(time_days, bin_edges):
    boundaries = bin_edges[1:]
    idx = torch.bucketize(time_days, boundaries=boundaries, right=False)
    return idx.clamp(max=int(boundaries.numel()) - 1)


def discrete_time_nll(hazard_logits, time_days, event_os, known_mask, bin_edges):
    valid = known_mask & torch.isfinite(time_days) & (time_days > 0)
    if int(valid.sum().item()) == 0:
        return hazard_logits.sum() * 0.0

    hazard_logits = hazard_logits[valid]
    time_days = time_days[valid]
    event_os = event_os[valid]

    hazard = torch.sigmoid(hazard_logits).clamp(min=1e-6, max=1.0 - 1e-6)
    log_hazard = torch.log(hazard)
    log_survival = torch.log1p(-hazard)

    bin_index = assign_time_bins_torch(time_days, bin_edges)
    arange_bins = torch.arange(hazard.shape[1], device=hazard.device).unsqueeze(0)
    before_mask = arange_bins < bin_index.unsqueeze(1)
    at_mask = arange_bins == bin_index.unsqueeze(1)

    event_mask = event_os > 0.5
    loss = hazard.new_zeros((hazard.shape[0],), dtype=hazard.dtype)
    if int(event_mask.sum().item()) > 0:
        loss[event_mask] = (
            -(log_survival[event_mask] * before_mask[event_mask].float()).sum(dim=1)
            - (log_hazard[event_mask] * at_mask[event_mask].float()).sum(dim=1)
        )
    cens_mask = ~event_mask
    if int(cens_mask.sum().item()) > 0:
        loss[cens_mask] = -(log_survival[cens_mask] * before_mask[cens_mask].float()).sum(dim=1)
    return loss.mean()


def masked_recurrence_bce(rec_logit, event_rec, known_mask, pos_weight):
    if int(known_mask.sum().item()) == 0:
        return rec_logit.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(
        rec_logit[known_mask],
        event_rec[known_mask],
        pos_weight=rec_logit.new_tensor(float(pos_weight)),
        reduction="mean",
    )
    return loss


def masked_recurrence_location_ce(loc_logits, loc_target, known_mask, class_weights):
    if int(known_mask.sum().item()) == 0:
        return loc_logits.sum() * 0.0
    return F.cross_entropy(
        loc_logits[known_mask],
        loc_target[known_mask],
        weight=class_weights,
        reduction="mean",
    )


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def concordance_index(time_days, event_os, risk_score, known_mask):
    valid_indices = np.where(
        (known_mask.astype(np.uint8) == 1)
        & np.isfinite(time_days)
        & (time_days > 0)
    )[0]
    if valid_indices.size <= 1:
        return None

    concordant = 0.0
    comparable = 0.0
    for i_pos in range(valid_indices.size):
        i = int(valid_indices[i_pos])
        for j_pos in range(i_pos + 1, valid_indices.size):
            j = int(valid_indices[j_pos])
            if int(event_os[i]) == 1 and float(time_days[i]) < float(time_days[j]):
                comparable += 1.0
                if float(risk_score[i]) > float(risk_score[j]):
                    concordant += 1.0
                elif float(risk_score[i]) == float(risk_score[j]):
                    concordant += 0.5
            elif int(event_os[j]) == 1 and float(time_days[j]) < float(time_days[i]):
                comparable += 1.0
                if float(risk_score[j]) > float(risk_score[i]):
                    concordant += 1.0
                elif float(risk_score[j]) == float(risk_score[i]):
                    concordant += 0.5
    if comparable <= 0.0:
        return None
    return float(concordant / comparable)


def binary_auc_score(y_true, y_score):
    positives = [float(s) for y, s in zip(y_true, y_score) if int(y) == 1]
    negatives = [float(s) for y, s in zip(y_true, y_score) if int(y) == 0]
    if not positives or not negatives:
        return None
    total = 0.0
    count = 0
    for pos in positives:
        for neg in negatives:
            count += 1
            if pos > neg:
                total += 1.0
            elif pos == neg:
                total += 0.5
    if count <= 0:
        return None
    return float(total / count)


def multiclass_accuracy_score(target_index, pred_prob):
    if len(target_index) == 0:
        return None
    pred_index = np.asarray(pred_prob).argmax(axis=1)
    target_index = np.asarray(target_index, dtype=np.int64)
    return float((pred_index == target_index).mean())


def safe_mean_std(values):
    valid = []
    for value in values:
        if value is None:
            continue
        try:
            value_float = float(value)
        except Exception:
            continue
        if math.isnan(value_float):
            continue
        valid.append(value_float)
    if not valid:
        return None, None
    arr = np.asarray(valid, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=0))


def compute_risk_score(outputs, survival_mode):
    if str(survival_mode) == "cox":
        return np.asarray(outputs["os_log_risk"], dtype=np.float32)
    survival_curve = np.asarray(outputs["survival_curve"], dtype=np.float32)
    if survival_curve.ndim == 2 and survival_curve.shape[1] > 0:
        return (-survival_curve.sum(axis=1)).astype(np.float32)
    hazard_prob = np.asarray(outputs["hazard_prob"], dtype=np.float32)
    return hazard_prob.sum(axis=1).astype(np.float32)


def evaluate_split_metrics(supervision, split_indices_np, outputs, survival_mode):
    split_indices_np = np.asarray(split_indices_np, dtype=np.int64)
    risk_score = compute_risk_score(outputs, survival_mode=survival_mode)

    c_index = concordance_index(
        time_days=supervision["time_os_days"][split_indices_np],
        event_os=supervision["event_os"][split_indices_np],
        risk_score=risk_score[split_indices_np],
        known_mask=supervision["os_label_known"][split_indices_np],
    )

    rec_known_mask = supervision["rec_label_known"][split_indices_np] == 1
    rec_auc = binary_auc_score(
        y_true=supervision["event_rec"][split_indices_np][rec_known_mask].astype(np.int64).tolist(),
        y_score=outputs["rec_prob"][split_indices_np][rec_known_mask].tolist(),
    )

    loc_known_mask = supervision["rec_location_known"][split_indices_np] == 1
    loc_acc = multiclass_accuracy_score(
        target_index=supervision["rec_location_index"][split_indices_np][loc_known_mask],
        pred_prob=outputs["rec_location_prob"][split_indices_np][loc_known_mask],
    )

    return {
        "val_patient_count": int(split_indices_np.shape[0]),
        "val_os_known_count": int(supervision["os_label_known"][split_indices_np].sum()),
        "val_rec_known_count": int(supervision["rec_label_known"][split_indices_np].sum()),
        "val_loc_known_count": int(supervision["rec_location_known"][split_indices_np].sum()),
        "val_c_index": c_index,
        "val_rec_auc": rec_auc,
        "val_loc_acc": loc_acc,
    }


class AttentionPool(nn.Module):
    def __init__(self, d_model=128, hidden_dim=128):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(int(d_model), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, z_prime):
        scores = self.score_mlp(z_prime).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        pooled = torch.sum(alpha.unsqueeze(-1) * z_prime, dim=1)
        return pooled, alpha


class WeightedSumPool(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.alpha_logits = nn.Parameter(torch.zeros(int(num_nodes)))

    def forward(self, z_prime):
        alpha = torch.softmax(self.alpha_logits, dim=0)
        pooled = torch.sum(alpha.view(1, -1, 1) * z_prime, dim=1)
        alpha_batch = alpha.view(1, -1).expand(z_prime.shape[0], -1)
        return pooled, alpha_batch


class PrimaryTaskHeads(nn.Module):
    def __init__(
        self,
        d_model=128,
        num_nodes=6,
        pool_mode="attention",
        pool_hidden_dim=128,
        trunk_hidden_dim=128,
        dropout=0.1,
        survival_mode="cox",
        num_time_bins=8,
        num_recurrence_classes=3,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_nodes = int(num_nodes)
        self.pool_mode = str(pool_mode)
        self.survival_mode = str(survival_mode)
        self.num_time_bins = int(num_time_bins)
        self.num_recurrence_classes = int(num_recurrence_classes)

        if self.pool_mode == "attention":
            self.pool = AttentionPool(d_model=self.d_model, hidden_dim=pool_hidden_dim)
        elif self.pool_mode == "weighted_sum":
            self.pool = WeightedSumPool(num_nodes=self.num_nodes)
        else:
            raise RuntimeError(f"unsupported pool_mode: {self.pool_mode}")

        self.trunk = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, int(trunk_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        trunk_dim = int(trunk_hidden_dim)

        if self.survival_mode == "cox":
            self.os_head = nn.Linear(trunk_dim, 1)
        elif self.survival_mode == "discrete":
            self.os_head = nn.Linear(trunk_dim, self.num_time_bins)
        else:
            raise RuntimeError(f"unsupported survival_mode: {self.survival_mode}")

        self.recurrence_head = nn.Linear(trunk_dim, 1)
        self.recurrence_location_head = nn.Linear(trunk_dim, self.num_recurrence_classes)

    def forward(self, z_prime):
        if z_prime.ndim != 3:
            raise RuntimeError(f"z_prime must be [B,N,D], got shape={tuple(z_prime.shape)}")
        if int(z_prime.shape[1]) != self.num_nodes or int(z_prime.shape[2]) != self.d_model:
            raise RuntimeError(
                f"z_prime shape mismatch: expected [B,{self.num_nodes},{self.d_model}] "
                f"got {tuple(z_prime.shape)}"
            )

        pooled, pool_weights = self.pool(z_prime)
        trunk = self.trunk(pooled)
        recurrence_logit = self.recurrence_head(trunk).squeeze(-1)
        recurrence_location_logits = self.recurrence_location_head(trunk)

        out = {
            "u": pooled,
            "trunk": trunk,
            "pool_weights": pool_weights,
            "recurrence_logit": recurrence_logit,
            "recurrence_location_logits": recurrence_location_logits,
        }
        if self.survival_mode == "cox":
            out["os_log_risk"] = self.os_head(trunk).squeeze(-1)
        else:
            out["hazard_logits"] = self.os_head(trunk)
        return out


def compute_total_loss(
    outputs,
    supervision_torch,
    indices,
    survival_mode,
    bin_edges_torch,
    recurrence_pos_weight,
    recurrence_location_class_weights,
    loss_weight_os,
    loss_weight_rec,
    loss_weight_loc,
):
    idx = indices
    os_known_mask = supervision_torch["os_label_known"][idx] > 0
    rec_known_mask = supervision_torch["rec_label_known"][idx] > 0
    loc_known_mask = supervision_torch["rec_location_known"][idx] > 0

    if str(survival_mode) == "cox":
        os_loss = cox_partial_log_likelihood(
            log_risk=outputs["os_log_risk"][idx],
            time_days=supervision_torch["time_os_days"][idx],
            event_os=supervision_torch["event_os"][idx],
            known_mask=os_known_mask,
        )
    else:
        os_loss = discrete_time_nll(
            hazard_logits=outputs["hazard_logits"][idx],
            time_days=supervision_torch["time_os_days"][idx],
            event_os=supervision_torch["event_os"][idx],
            known_mask=os_known_mask,
            bin_edges=bin_edges_torch,
        )

    rec_loss = masked_recurrence_bce(
        rec_logit=outputs["recurrence_logit"][idx],
        event_rec=supervision_torch["event_rec"][idx],
        known_mask=rec_known_mask,
        pos_weight=recurrence_pos_weight,
    )
    loc_loss = masked_recurrence_location_ce(
        loc_logits=outputs["recurrence_location_logits"][idx],
        loc_target=supervision_torch["rec_location_index"][idx],
        known_mask=loc_known_mask,
        class_weights=recurrence_location_class_weights,
    )
    total_loss = (
        float(loss_weight_os) * os_loss
        + float(loss_weight_rec) * rec_loss
        + float(loss_weight_loc) * loc_loss
    )
    return {
        "total_loss": total_loss,
        "os_loss": os_loss,
        "rec_loss": rec_loss,
        "loc_loss": loc_loss,
        "os_known_count": int(os_known_mask.sum().item()),
        "rec_known_count": int(rec_known_mask.sum().item()),
        "loc_known_count": int(loc_known_mask.sum().item()),
    }


def tensorize_supervision(supervision, device):
    return {
        "event_os": torch.from_numpy(supervision["event_os"]).float().to(device),
        "time_os_days": torch.from_numpy(supervision["time_os_days"]).float().to(device),
        "os_label_known": torch.from_numpy(supervision["os_label_known"]).to(device),
        "event_rec": torch.from_numpy(supervision["event_rec"]).float().to(device),
        "time_rec_days": torch.from_numpy(supervision["time_rec_days"]).float().to(device),
        "rec_label_known": torch.from_numpy(supervision["rec_label_known"]).to(device),
        "rec_location_index": torch.from_numpy(supervision["rec_location_index"]).long().to(device),
        "rec_location_known": torch.from_numpy(supervision["rec_location_known"]).to(device),
    }


def train_primary_heads(
    z_prime,
    supervision,
    recurrence_classes,
    output_paths,
    pool_mode="attention",
    survival_mode="cox",
    num_time_bins=8,
    pool_hidden_dim=128,
    trunk_hidden_dim=128,
    dropout=0.1,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    val_ratio=0.2,
    early_stop_patience=30,
    early_stop_min_delta=1e-5,
    loss_weight_os=1.0,
    loss_weight_rec=1.0,
    loss_weight_loc=1.0,
    seed=2026,
    device="auto",
    train_indices_np=None,
    val_indices_np=None,
):
    device = choose_device(device)
    set_seed(seed)

    if train_indices_np is None or val_indices_np is None:
        strata = build_split_strata(supervision)
        train_indices_np, val_indices_np = stratified_split_indices(strata, val_ratio=val_ratio, seed=seed)
    else:
        train_indices_np = np.asarray(train_indices_np, dtype=np.int64)
        val_indices_np = np.asarray(val_indices_np, dtype=np.int64)
        if np.intersect1d(train_indices_np, val_indices_np).size > 0:
            raise RuntimeError("train_indices_np and val_indices_np must be disjoint")
    if train_indices_np.size == 0:
        raise RuntimeError("empty train split")

    z_prime_torch = torch.from_numpy(z_prime).float().to(device)
    supervision_torch = tensorize_supervision(supervision, device=device)

    train_indices = torch.from_numpy(train_indices_np).long().to(device)
    val_indices = torch.from_numpy(val_indices_np).long().to(device)

    bin_edges = None
    bin_edges_torch = None
    if str(survival_mode) == "discrete":
        bin_edges = build_time_bin_edges(
            train_time_days=supervision["time_os_days"][train_indices_np],
            train_event_os=supervision["event_os"][train_indices_np],
            num_bins=num_time_bins,
        )
        bin_edges_torch = torch.from_numpy(bin_edges).float().to(device)
    else:
        bin_edges = np.asarray([], dtype=np.float32)

    model = PrimaryTaskHeads(
        d_model=int(z_prime.shape[-1]),
        num_nodes=int(z_prime.shape[1]),
        pool_mode=pool_mode,
        pool_hidden_dim=pool_hidden_dim,
        trunk_hidden_dim=trunk_hidden_dim,
        dropout=dropout,
        survival_mode=survival_mode,
        num_time_bins=num_time_bins,
        num_recurrence_classes=len(recurrence_classes),
    ).to(device)

    recurrence_pos_weight = compute_binary_pos_weight(
        target=supervision_torch["event_rec"][train_indices],
        known_mask=supervision_torch["rec_label_known"][train_indices] > 0,
    )
    recurrence_location_class_weights = compute_multiclass_weights(
        target=supervision_torch["rec_location_index"][train_indices],
        known_mask=supervision_torch["rec_location_known"][train_indices] > 0,
        num_classes=len(recurrence_classes),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    history_rows = []
    best_state = snapshot_state_dict(model)
    best_epoch = 0
    best_monitor_loss = None
    bad_epochs = 0
    stopped_early = False

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(z_prime_torch)
        train_loss_dict = compute_total_loss(
            outputs=train_outputs,
            supervision_torch=supervision_torch,
            indices=train_indices,
            survival_mode=survival_mode,
            bin_edges_torch=bin_edges_torch,
            recurrence_pos_weight=recurrence_pos_weight,
            recurrence_location_class_weights=recurrence_location_class_weights,
            loss_weight_os=loss_weight_os,
            loss_weight_rec=loss_weight_rec,
            loss_weight_loc=loss_weight_loc,
        )
        train_loss_dict["total_loss"].backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(z_prime_torch)
            full_train_loss_dict = compute_total_loss(
                outputs=outputs,
                supervision_torch=supervision_torch,
                indices=train_indices,
                survival_mode=survival_mode,
                bin_edges_torch=bin_edges_torch,
                recurrence_pos_weight=recurrence_pos_weight,
                recurrence_location_class_weights=recurrence_location_class_weights,
                loss_weight_os=loss_weight_os,
                loss_weight_rec=loss_weight_rec,
                loss_weight_loc=loss_weight_loc,
            )
            if val_indices.numel() > 0:
                val_loss_dict = compute_total_loss(
                    outputs=outputs,
                    supervision_torch=supervision_torch,
                    indices=val_indices,
                    survival_mode=survival_mode,
                    bin_edges_torch=bin_edges_torch,
                    recurrence_pos_weight=recurrence_pos_weight,
                    recurrence_location_class_weights=recurrence_location_class_weights,
                    loss_weight_os=loss_weight_os,
                    loss_weight_rec=loss_weight_rec,
                    loss_weight_loc=loss_weight_loc,
                )
                monitor_loss = float(val_loss_dict["total_loss"].item())
            else:
                val_loss_dict = full_train_loss_dict
                monitor_loss = float(full_train_loss_dict["total_loss"].item())

        improved = False
        if best_monitor_loss is None or monitor_loss < (best_monitor_loss - float(early_stop_min_delta)):
            best_monitor_loss = monitor_loss
            best_epoch = int(epoch)
            best_state = snapshot_state_dict(model)
            bad_epochs = 0
            improved = True
        else:
            bad_epochs += 1

        if survival_mode == "cox":
            risk_scores = outputs["os_log_risk"].detach().cpu().numpy()
        else:
            hazard_prob = torch.sigmoid(outputs["hazard_logits"]).detach().cpu().numpy()
            survival_curve = np.cumprod(1.0 - hazard_prob, axis=1)
            risk_scores = -survival_curve.sum(axis=1)

        train_c_index = concordance_index(
            time_days=supervision["time_os_days"][train_indices_np],
            event_os=supervision["event_os"][train_indices_np],
            risk_score=risk_scores[train_indices_np],
            known_mask=supervision["os_label_known"][train_indices_np],
        )
        val_c_index = None
        if val_indices_np.size > 0:
            val_c_index = concordance_index(
                time_days=supervision["time_os_days"][val_indices_np],
                event_os=supervision["event_os"][val_indices_np],
                risk_score=risk_scores[val_indices_np],
                known_mask=supervision["os_label_known"][val_indices_np],
            )

        history_rows.append(
            {
                "epoch": int(epoch),
                "train_total_loss": float(full_train_loss_dict["total_loss"].item()),
                "train_os_loss": float(full_train_loss_dict["os_loss"].item()),
                "train_rec_loss": float(full_train_loss_dict["rec_loss"].item()),
                "train_loc_loss": float(full_train_loss_dict["loc_loss"].item()),
                "train_c_index": "" if train_c_index is None else float(train_c_index),
                "val_total_loss": float(val_loss_dict["total_loss"].item()),
                "val_os_loss": float(val_loss_dict["os_loss"].item()),
                "val_rec_loss": float(val_loss_dict["rec_loss"].item()),
                "val_loc_loss": float(val_loss_dict["loc_loss"].item()),
                "val_c_index": "" if val_c_index is None else float(val_c_index),
                "monitor_loss": float(monitor_loss),
                "is_best_epoch": 1 if improved else 0,
                "bad_epochs": int(bad_epochs),
            }
        )

        if int(bad_epochs) >= int(early_stop_patience):
            stopped_early = True
            break

    restore_state_dict(model, best_state)
    model.eval()

    with torch.no_grad():
        outputs = model(z_prime_torch)
        rec_prob = torch.sigmoid(outputs["recurrence_logit"]).detach().cpu().numpy().astype(np.float32)
        rec_location_prob = torch.softmax(
            outputs["recurrence_location_logits"], dim=1
        ).detach().cpu().numpy().astype(np.float32)
        pool_weights = outputs["pool_weights"].detach().cpu().numpy().astype(np.float32)
        pooled_u = outputs["u"].detach().cpu().numpy().astype(np.float32)
        trunk = outputs["trunk"].detach().cpu().numpy().astype(np.float32)

        if survival_mode == "cox":
            os_log_risk = outputs["os_log_risk"].detach().cpu().numpy().astype(np.float32)
            hazard_prob = np.zeros((z_prime.shape[0], 0), dtype=np.float32)
            survival_curve = np.zeros((z_prime.shape[0], 0), dtype=np.float32)
        else:
            os_log_risk = np.zeros((z_prime.shape[0],), dtype=np.float32)
            hazard_prob = torch.sigmoid(outputs["hazard_logits"]).detach().cpu().numpy().astype(np.float32)
            survival_curve = np.cumprod(1.0 - hazard_prob, axis=1).astype(np.float32)

    split_name = np.asarray(["train"] * z_prime.shape[0], dtype=object)
    split_name[val_indices_np] = "val"

    model_payload = {
        "state_dict": model.state_dict(),
        "config": {
            "d_model": int(z_prime.shape[-1]),
            "num_nodes": int(z_prime.shape[1]),
            "pool_mode": str(pool_mode),
            "pool_hidden_dim": int(pool_hidden_dim),
            "trunk_hidden_dim": int(trunk_hidden_dim),
            "dropout": float(dropout),
            "survival_mode": str(survival_mode),
            "num_time_bins": int(num_time_bins),
            "recurrence_classes": list(recurrence_classes),
            "bin_edges": bin_edges.tolist(),
        },
    }
    torch.save(model_payload, output_paths["model_path"])

    write_csv(
        output_paths["train_csv"],
        [
            "epoch",
            "train_total_loss",
            "train_os_loss",
            "train_rec_loss",
            "train_loc_loss",
            "train_c_index",
            "val_total_loss",
            "val_os_loss",
            "val_rec_loss",
            "val_loc_loss",
            "val_c_index",
            "monitor_loss",
            "is_best_epoch",
            "bad_epochs",
        ],
        history_rows,
    )

    return {
        "model": model,
        "history_rows": history_rows,
        "best_epoch": int(best_epoch),
        "best_monitor_loss": None if best_monitor_loss is None else float(best_monitor_loss),
        "stopped_early": bool(stopped_early),
        "train_indices": train_indices_np,
        "val_indices": val_indices_np,
        "recurrence_pos_weight": float(recurrence_pos_weight),
        "recurrence_location_class_weights": recurrence_location_class_weights.detach().cpu().numpy().astype(np.float32),
        "bin_edges": bin_edges.astype(np.float32),
        "split_name": split_name,
        "pool_weights": pool_weights,
        "pooled_u": pooled_u,
        "trunk": trunk,
        "rec_prob": rec_prob,
        "rec_location_prob": rec_location_prob,
        "os_log_risk": os_log_risk,
        "hazard_prob": hazard_prob,
        "survival_curve": survival_curve,
    }


def find_history_row_by_epoch(history_rows, epoch):
    for row in history_rows:
        if int(row["epoch"]) == int(epoch):
            return row
    if not history_rows:
        raise RuntimeError("history_rows is empty")
    return history_rows[-1]


def run_cross_validation(
    stage11_pack,
    supervision,
    recurrence_classes,
    output_root,
    pool_mode="attention",
    survival_mode="discrete",
    num_time_bins=8,
    pool_hidden_dim=128,
    trunk_hidden_dim=128,
    dropout=0.1,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    num_folds=3,
    cv_seeds=None,
    early_stop_patience=30,
    early_stop_min_delta=1e-5,
    loss_weight_os=0.5,
    loss_weight_rec=1.0,
    loss_weight_loc=1.0,
    device="auto",
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs_root = output_root / "cv_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    patient_ids = np.asarray(stage11_pack["patient_ids"]).astype(str)
    z_prime = stage11_pack["Z_prime"].astype(np.float32)
    strata = build_split_strata(supervision)
    seeds = [int(x) for x in (cv_seeds or [])]
    if not seeds:
        raise RuntimeError("cv_seeds must not be empty for cross-validation")

    fold_metric_rows = []
    seed_summary_rows = []

    for cv_seed in seeds:
        fold_splits = stratified_kfold_splits(strata=strata, num_folds=num_folds, seed=cv_seed)

        oof_risk = np.full((len(patient_ids),), np.nan, dtype=np.float32)
        oof_rec_prob = np.full((len(patient_ids),), np.nan, dtype=np.float32)
        oof_rec_loc_prob = np.full((len(patient_ids), len(recurrence_classes)), np.nan, dtype=np.float32)
        oof_mask = np.zeros((len(patient_ids),), dtype=np.uint8)

        seed_fold_c_index = []
        seed_fold_rec_auc = []
        seed_fold_loc_acc = []
        seed_fold_monitor = []

        for fold_idx, (train_indices_np, val_indices_np) in enumerate(fold_splits, start=1):
            run_root = runs_root / f"seed_{cv_seed}" / f"fold_{fold_idx:02d}"
            output_paths = ensure_output_dirs(run_root)
            run_seed = int(cv_seed) * 100 + int(fold_idx)

            train_result = train_primary_heads(
                z_prime=z_prime,
                supervision=supervision,
                recurrence_classes=recurrence_classes,
                output_paths=output_paths,
                pool_mode=pool_mode,
                survival_mode=survival_mode,
                num_time_bins=num_time_bins,
                pool_hidden_dim=pool_hidden_dim,
                trunk_hidden_dim=trunk_hidden_dim,
                dropout=dropout,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                val_ratio=0.0,
                early_stop_patience=early_stop_patience,
                early_stop_min_delta=early_stop_min_delta,
                loss_weight_os=loss_weight_os,
                loss_weight_rec=loss_weight_rec,
                loss_weight_loc=loss_weight_loc,
                seed=run_seed,
                device=device,
                train_indices_np=train_indices_np,
                val_indices_np=val_indices_np,
            )

            write_prediction_csv(
                path=output_paths["pred_csv"],
                patient_ids=patient_ids,
                split_name=train_result["split_name"],
                supervision=supervision,
                recurrence_classes=recurrence_classes,
                survival_mode=survival_mode,
                outputs=train_result,
            )

            best_row = find_history_row_by_epoch(train_result["history_rows"], train_result["best_epoch"])
            split_metrics = evaluate_split_metrics(
                supervision=supervision,
                split_indices_np=val_indices_np,
                outputs=train_result,
                survival_mode=survival_mode,
            )
            risk_score = compute_risk_score(train_result, survival_mode=survival_mode)

            oof_risk[val_indices_np] = risk_score[val_indices_np]
            oof_rec_prob[val_indices_np] = train_result["rec_prob"][val_indices_np]
            oof_rec_loc_prob[val_indices_np] = train_result["rec_location_prob"][val_indices_np]
            oof_mask[val_indices_np] = 1

            seed_fold_c_index.append(split_metrics["val_c_index"])
            seed_fold_rec_auc.append(split_metrics["val_rec_auc"])
            seed_fold_loc_acc.append(split_metrics["val_loc_acc"])
            seed_fold_monitor.append(float(best_row["monitor_loss"]))

            fold_metric_rows.append(
                {
                    "cv_seed": int(cv_seed),
                    "fold_index": int(fold_idx),
                    "run_seed": int(run_seed),
                    "output_root": str(run_root),
                    "train_count": int(train_indices_np.shape[0]),
                    "val_count": int(val_indices_np.shape[0]),
                    "best_epoch": int(train_result["best_epoch"]),
                    "best_monitor_loss": float(train_result["best_monitor_loss"]),
                    "val_total_loss": float(best_row["val_total_loss"]),
                    "val_os_loss": float(best_row["val_os_loss"]),
                    "val_rec_loss": float(best_row["val_rec_loss"]),
                    "val_loc_loss": float(best_row["val_loc_loss"]),
                    "val_c_index": "" if split_metrics["val_c_index"] is None else float(split_metrics["val_c_index"]),
                    "val_rec_auc": "" if split_metrics["val_rec_auc"] is None else float(split_metrics["val_rec_auc"]),
                    "val_loc_acc": "" if split_metrics["val_loc_acc"] is None else float(split_metrics["val_loc_acc"]),
                    "val_os_known_count": int(split_metrics["val_os_known_count"]),
                    "val_rec_known_count": int(split_metrics["val_rec_known_count"]),
                    "val_loc_known_count": int(split_metrics["val_loc_known_count"]),
                }
            )

        oof_indices = np.where(oof_mask == 1)[0]
        if oof_indices.shape[0] != len(patient_ids):
            raise RuntimeError(
                f"OOF predictions incomplete for cv_seed={cv_seed}: "
                f"covered={oof_indices.shape[0]} total={len(patient_ids)}"
            )

        oof_c_index = concordance_index(
            time_days=supervision["time_os_days"][oof_indices],
            event_os=supervision["event_os"][oof_indices],
            risk_score=oof_risk[oof_indices],
            known_mask=supervision["os_label_known"][oof_indices],
        )
        rec_known_mask = supervision["rec_label_known"][oof_indices] == 1
        oof_rec_auc = binary_auc_score(
            y_true=supervision["event_rec"][oof_indices][rec_known_mask].astype(np.int64).tolist(),
            y_score=oof_rec_prob[oof_indices][rec_known_mask].tolist(),
        )
        loc_known_mask = supervision["rec_location_known"][oof_indices] == 1
        oof_loc_acc = multiclass_accuracy_score(
            target_index=supervision["rec_location_index"][oof_indices][loc_known_mask],
            pred_prob=oof_rec_loc_prob[oof_indices][loc_known_mask],
        )

        fold_c_mean, fold_c_std = safe_mean_std(seed_fold_c_index)
        fold_auc_mean, fold_auc_std = safe_mean_std(seed_fold_rec_auc)
        fold_loc_mean, fold_loc_std = safe_mean_std(seed_fold_loc_acc)
        fold_monitor_mean, fold_monitor_std = safe_mean_std(seed_fold_monitor)

        seed_summary_rows.append(
            {
                "cv_seed": int(cv_seed),
                "num_folds": int(num_folds),
                "fold_monitor_loss_mean": "" if fold_monitor_mean is None else fold_monitor_mean,
                "fold_monitor_loss_std": "" if fold_monitor_std is None else fold_monitor_std,
                "fold_val_c_index_mean": "" if fold_c_mean is None else fold_c_mean,
                "fold_val_c_index_std": "" if fold_c_std is None else fold_c_std,
                "fold_val_rec_auc_mean": "" if fold_auc_mean is None else fold_auc_mean,
                "fold_val_rec_auc_std": "" if fold_auc_std is None else fold_auc_std,
                "fold_val_loc_acc_mean": "" if fold_loc_mean is None else fold_loc_mean,
                "fold_val_loc_acc_std": "" if fold_loc_std is None else fold_loc_std,
                "oof_c_index": "" if oof_c_index is None else float(oof_c_index),
                "oof_rec_auc": "" if oof_rec_auc is None else float(oof_rec_auc),
                "oof_loc_acc": "" if oof_loc_acc is None else float(oof_loc_acc),
            }
        )

    fold_fieldnames = [
        "cv_seed",
        "fold_index",
        "run_seed",
        "output_root",
        "train_count",
        "val_count",
        "best_epoch",
        "best_monitor_loss",
        "val_total_loss",
        "val_os_loss",
        "val_rec_loss",
        "val_loc_loss",
        "val_c_index",
        "val_rec_auc",
        "val_loc_acc",
        "val_os_known_count",
        "val_rec_known_count",
        "val_loc_known_count",
    ]
    seed_fieldnames = [
        "cv_seed",
        "num_folds",
        "fold_monitor_loss_mean",
        "fold_monitor_loss_std",
        "fold_val_c_index_mean",
        "fold_val_c_index_std",
        "fold_val_rec_auc_mean",
        "fold_val_rec_auc_std",
        "fold_val_loc_acc_mean",
        "fold_val_loc_acc_std",
        "oof_c_index",
        "oof_rec_auc",
        "oof_loc_acc",
    ]
    write_csv(output_root / "cv_fold_metrics.csv", fold_fieldnames, fold_metric_rows)
    write_csv(output_root / "cv_seed_summary.csv", seed_fieldnames, seed_summary_rows)

    oof_c_mean, oof_c_std = safe_mean_std([row["oof_c_index"] for row in seed_summary_rows])
    oof_auc_mean, oof_auc_std = safe_mean_std([row["oof_rec_auc"] for row in seed_summary_rows])
    oof_loc_mean, oof_loc_std = safe_mean_std([row["oof_loc_acc"] for row in seed_summary_rows])

    summary = {
        "patient_count": int(len(patient_ids)),
        "num_folds": int(num_folds),
        "cv_seeds": [int(x) for x in seeds],
        "num_runs": int(len(fold_metric_rows)),
        "pool_mode": str(pool_mode),
        "survival_mode": str(survival_mode),
        "num_time_bins": int(num_time_bins),
        "loss_weight_os": float(loss_weight_os),
        "loss_weight_rec": float(loss_weight_rec),
        "loss_weight_loc": float(loss_weight_loc),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "epochs": int(epochs),
        "fold_metric_mean": {
            "val_c_index": safe_mean_std([row["val_c_index"] for row in fold_metric_rows])[0],
            "val_rec_auc": safe_mean_std([row["val_rec_auc"] for row in fold_metric_rows])[0],
            "val_loc_acc": safe_mean_std([row["val_loc_acc"] for row in fold_metric_rows])[0],
            "best_monitor_loss": safe_mean_std([row["best_monitor_loss"] for row in fold_metric_rows])[0],
        },
        "fold_metric_std": {
            "val_c_index": safe_mean_std([row["val_c_index"] for row in fold_metric_rows])[1],
            "val_rec_auc": safe_mean_std([row["val_rec_auc"] for row in fold_metric_rows])[1],
            "val_loc_acc": safe_mean_std([row["val_loc_acc"] for row in fold_metric_rows])[1],
            "best_monitor_loss": safe_mean_std([row["best_monitor_loss"] for row in fold_metric_rows])[1],
        },
        "oof_metric_mean": {
            "oof_c_index": oof_c_mean,
            "oof_rec_auc": oof_auc_mean,
            "oof_loc_acc": oof_loc_mean,
        },
        "oof_metric_std": {
            "oof_c_index": oof_c_std,
            "oof_rec_auc": oof_auc_std,
            "oof_loc_acc": oof_loc_std,
        },
    }
    write_json(output_root / "cv_summary.json", summary)
    return summary


def write_prediction_csv(
    path,
    patient_ids,
    split_name,
    supervision,
    recurrence_classes,
    survival_mode,
    outputs,
):
    fieldnames = [
        "patient_id",
        "split",
        "os_label_known",
        "time_os_days",
        "event_os",
        "rec_label_known",
        "time_rec_days",
        "event_rec",
        "rec_location_known",
        "rec_location_target",
        "recurrence_probability",
    ]
    if str(survival_mode) == "cox":
        fieldnames.append("os_log_risk")
    else:
        fieldnames.append("hazard_prob_json")
        fieldnames.append("survival_curve_json")
    for name in recurrence_classes:
        fieldnames.append(f"rec_location_prob__{name}")

    rows = []
    for idx, patient_id in enumerate(patient_ids.tolist()):
        row = {
            "patient_id": patient_id,
            "split": str(split_name[idx]),
            "os_label_known": int(supervision["os_label_known"][idx]),
            "time_os_days": "" if int(supervision["os_label_known"][idx]) == 0 else float(supervision["time_os_days"][idx]),
            "event_os": "" if int(supervision["os_label_known"][idx]) == 0 else int(supervision["event_os"][idx]),
            "rec_label_known": int(supervision["rec_label_known"][idx]),
            "time_rec_days": "" if int(supervision["rec_label_known"][idx]) == 0 else float(supervision["time_rec_days"][idx]),
            "event_rec": "" if int(supervision["rec_label_known"][idx]) == 0 else int(supervision["event_rec"][idx]),
            "rec_location_known": int(supervision["rec_location_known"][idx]),
            "rec_location_target": (
                ""
                if int(supervision["rec_location_known"][idx]) == 0
                else recurrence_classes[int(supervision["rec_location_index"][idx])]
            ),
            "recurrence_probability": float(outputs["rec_prob"][idx]),
        }
        if str(survival_mode) == "cox":
            row["os_log_risk"] = float(outputs["os_log_risk"][idx])
        else:
            row["hazard_prob_json"] = json.dumps(outputs["hazard_prob"][idx].tolist())
            row["survival_curve_json"] = json.dumps(outputs["survival_curve"][idx].tolist())
        for class_idx, name in enumerate(recurrence_classes):
            row[f"rec_location_prob__{name}"] = float(outputs["rec_location_prob"][idx, class_idx])
        rows.append(row)
    write_csv(path, fieldnames, rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 12.1 primary supervised outputs: OS survival head + recurrence head",
        allow_abbrev=False,
    )
    parser.add_argument("--stage11-pack", type=str, default=str(DEFAULT_STAGE11_PACK))
    parser.add_argument("--labels-csv", type=str, default="")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--pool-mode", type=str, default="attention", choices=["attention", "weighted_sum"])
    parser.add_argument("--survival-mode", type=str, default="discrete", choices=["cox", "discrete"])
    parser.add_argument("--num-time-bins", type=int, default=8)
    parser.add_argument("--recurrence-classes", type=str, default="")
    parser.add_argument("--pool-hidden-dim", type=int, default=128)
    parser.add_argument("--trunk-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5)
    parser.add_argument("--loss-weight-os", type=float, default=0.5)
    parser.add_argument("--loss-weight-rec", type=float, default=1.0)
    parser.add_argument("--loss-weight-loc", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cv-num-folds", type=int, default=0, help="0/1 disables CV; >=2 runs k-fold evaluation.")
    parser.add_argument(
        "--cv-seeds",
        type=str,
        default="",
        help="Comma-separated seeds for CV, e.g. 2024,2025,2026. Empty means use --seed only.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    if args.num_time_bins <= 0:
        raise SystemExit("--num-time-bins must be > 0")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if args.val_ratio < 0.0 or args.val_ratio >= 1.0:
        raise SystemExit("--val-ratio must be in [0, 1)")
    if args.cv_num_folds < 0:
        raise SystemExit("--cv-num-folds must be >= 0")

    stage11_pack_path = resolve_required_path(args.stage11_pack, DEFAULT_STAGE11_PACK, "stage11 pack")
    labels_csv_path = resolve_labels_path(args.labels_csv)

    print(f"[start] stage11_pack={stage11_pack_path}")
    print(f"[start] labels_csv={labels_csv_path}")
    print(f"[start] output_root={Path(args.output_root)}")
    print(
        f"[start] pool_mode={args.pool_mode} survival_mode={args.survival_mode} "
        f"num_time_bins={args.num_time_bins}"
    )
    print(
        f"[start] epochs={args.epochs} lr={args.lr} weight_decay={args.weight_decay} "
        f"val_ratio={args.val_ratio}"
    )

    stage11_pack = load_stage11_pack(stage11_pack_path)
    labels_by_patient = load_label_rows(labels_csv_path)
    recurrence_classes = resolve_recurrence_classes(labels_by_patient, args.recurrence_classes)
    supervision = build_supervision_arrays(
        patient_ids=stage11_pack["patient_ids"],
        labels_by_patient=labels_by_patient,
        recurrence_classes=recurrence_classes,
    )

    print(
        f"[data] patient_count={len(stage11_pack['patient_ids'])} "
        f"num_nodes={int(stage11_pack['Z_prime'].shape[1])} d_model={int(stage11_pack['Z_prime'].shape[2])}"
    )
    print(
        f"[labels] os_known={int(supervision['os_label_known'].sum())} "
        f"rec_known={int(supervision['rec_label_known'].sum())} "
        f"rec_location_known={int(supervision['rec_location_known'].sum())}"
    )
    print(f"[labels] recurrence_classes={recurrence_classes}")

    if int(args.cv_num_folds) >= 2:
        cv_seeds = parse_seed_list(args.cv_seeds, default_seed=args.seed)
        print(f"[cv] num_folds={args.cv_num_folds} cv_seeds={cv_seeds}")
        cv_summary = run_cross_validation(
            stage11_pack=stage11_pack,
            supervision=supervision,
            recurrence_classes=recurrence_classes,
            output_root=args.output_root,
            pool_mode=args.pool_mode,
            survival_mode=args.survival_mode,
            num_time_bins=args.num_time_bins,
            pool_hidden_dim=args.pool_hidden_dim,
            trunk_hidden_dim=args.trunk_hidden_dim,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_folds=args.cv_num_folds,
            cv_seeds=cv_seeds,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            loss_weight_os=args.loss_weight_os,
            loss_weight_rec=args.loss_weight_rec,
            loss_weight_loc=args.loss_weight_loc,
            device=args.device,
        )
        print(f"wrote: {Path(args.output_root) / 'cv_fold_metrics.csv'}")
        print(f"wrote: {Path(args.output_root) / 'cv_seed_summary.csv'}")
        print(f"wrote: {Path(args.output_root) / 'cv_summary.json'}")
        print(f"cv_oof_metric_mean: {cv_summary['oof_metric_mean']}")
        print("complete")
        return

    output_paths = ensure_output_dirs(args.output_root)

    train_result = train_primary_heads(
        z_prime=stage11_pack["Z_prime"].astype(np.float32),
        supervision=supervision,
        recurrence_classes=recurrence_classes,
        output_paths=output_paths,
        pool_mode=args.pool_mode,
        survival_mode=args.survival_mode,
        num_time_bins=args.num_time_bins,
        pool_hidden_dim=args.pool_hidden_dim,
        trunk_hidden_dim=args.trunk_hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        loss_weight_os=args.loss_weight_os,
        loss_weight_rec=args.loss_weight_rec,
        loss_weight_loc=args.loss_weight_loc,
        seed=args.seed,
        device=args.device,
    )

    write_prediction_csv(
        path=output_paths["pred_csv"],
        patient_ids=stage11_pack["patient_ids"],
        split_name=train_result["split_name"],
        supervision=supervision,
        recurrence_classes=recurrence_classes,
        survival_mode=args.survival_mode,
        outputs=train_result,
    )

    np.savez_compressed(
        output_paths["pred_npz"],
        patient_ids=stage11_pack["patient_ids"].astype(str),
        organ_node_names=stage11_pack["organ_node_names"].astype(str),
        split_name=train_result["split_name"].astype(str),
        recurrence_classes=np.asarray(recurrence_classes).astype(str),
        pool_weights=train_result["pool_weights"].astype(np.float32),
        pooled_u=train_result["pooled_u"].astype(np.float32),
        trunk=train_result["trunk"].astype(np.float32),
        os_log_risk=train_result["os_log_risk"].astype(np.float32),
        hazard_prob=train_result["hazard_prob"].astype(np.float32),
        survival_curve=train_result["survival_curve"].astype(np.float32),
        recurrence_probability=train_result["rec_prob"].astype(np.float32),
        recurrence_location_probability=train_result["rec_location_prob"].astype(np.float32),
        event_os=supervision["event_os"].astype(np.float32),
        time_os_days=supervision["time_os_days"].astype(np.float32),
        os_label_known=supervision["os_label_known"].astype(np.uint8),
        event_rec=supervision["event_rec"].astype(np.float32),
        time_rec_days=supervision["time_rec_days"].astype(np.float32),
        rec_label_known=supervision["rec_label_known"].astype(np.uint8),
        rec_location_index=supervision["rec_location_index"].astype(np.int64),
        rec_location_known=supervision["rec_location_known"].astype(np.uint8),
        time_bin_edges=train_result["bin_edges"].astype(np.float32),
    )

    meta = {
        "stage11_pack_path": str(stage11_pack_path),
        "labels_csv_path": str(labels_csv_path),
        "output_root": str(output_paths["root"]),
        "patient_count": int(len(stage11_pack["patient_ids"])),
        "num_nodes": int(stage11_pack["Z_prime"].shape[1]),
        "d_model": int(stage11_pack["Z_prime"].shape[2]),
        "pool_mode": str(args.pool_mode),
        "survival_mode": str(args.survival_mode),
        "num_time_bins": int(args.num_time_bins),
        "time_bin_edges": train_result["bin_edges"].tolist(),
        "recurrence_classes": list(recurrence_classes),
        "train_count": int(train_result["train_indices"].shape[0]),
        "val_count": int(train_result["val_indices"].shape[0]),
        "os_label_known_count": int(supervision["os_label_known"].sum()),
        "rec_label_known_count": int(supervision["rec_label_known"].sum()),
        "rec_location_known_count": int(supervision["rec_location_known"].sum()),
        "recurrence_pos_weight": float(train_result["recurrence_pos_weight"]),
        "recurrence_location_class_weights": train_result["recurrence_location_class_weights"].tolist(),
        "pool_hidden_dim": int(args.pool_hidden_dim),
        "trunk_hidden_dim": int(args.trunk_hidden_dim),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "val_ratio": float(args.val_ratio),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "loss_weight_os": float(args.loss_weight_os),
        "loss_weight_rec": float(args.loss_weight_rec),
        "loss_weight_loc": float(args.loss_weight_loc),
        "seed": int(args.seed),
        "device": str(args.device),
        "best_epoch": int(train_result["best_epoch"]),
        "best_monitor_loss": (
            None
            if train_result["best_monitor_loss"] is None
            else float(train_result["best_monitor_loss"])
        ),
        "stopped_early": bool(train_result["stopped_early"]),
    }
    write_json(output_paths["meta_json"], meta)

    print(f"wrote: {output_paths['model_path']}")
    print(f"wrote: {output_paths['train_csv']}")
    print(f"wrote: {output_paths['meta_json']}")
    print(f"wrote: {output_paths['pred_csv']}")
    print(f"wrote: {output_paths['pred_npz']}")
    print(f"best_epoch: {train_result['best_epoch']}")
    print(f"best_monitor_loss: {train_result['best_monitor_loss']}")
    print("complete")


if __name__ == "__main__":
    main()
