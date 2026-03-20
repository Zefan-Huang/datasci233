import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


ROOT = resolve_root()
DEFAULT_STAGE11_PACK = ROOT / "output/stage11/11.2_graph_reasoning/graph_reasoning_pack.npz"
DEFAULT_OUTPUT_ROOT = ROOT / "output/stage12/12.2_explanation_training"


def load_local_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRIMARY_MOD = load_local_module(ROOT / "12.1_primary_outputs.py", "stage12_primary_outputs")



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
        "model_path": model_dir / "explanation_guided_model.pt",
        "train_csv": train_dir / "explanation_training_summary.csv",
        "meta_json": train_dir / "explanation_meta.json",
        "pred_csv": pred_dir / "patient_primary_predictions.csv",
        "primary_npz": pred_dir / "primary_output_pack.npz",
        "graph_npz": pred_dir / "graph_reasoning_pack.npz",
        "graph_summary_json": pred_dir / "graph_reasoning_summary.json",
    }


def filter_stage11_pack(stage11_pack, keep_patient_ids):
    keep_set = {str(x) for x in keep_patient_ids}
    patient_ids = [str(x) for x in stage11_pack["patient_ids"].tolist()]
    indices = [idx for idx, patient_id in enumerate(patient_ids) if patient_id in keep_set]
    if not indices:
        raise RuntimeError("no matching patients found for requested stage11 subset")
    subset = {}
    for key, value in stage11_pack.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(patient_ids):
            subset[key] = value[np.asarray(indices, dtype=np.int64)]
        else:
            subset[key] = value
    return subset


def load_stage11_pack(npz_path):
    with np.load(npz_path, allow_pickle=True) as z:
        out = {}
        for key in z.files:
            value = z[key]
            if value.dtype.kind in {"U", "O"}:
                out[key] = value.astype(str)
            else:
                out[key] = value
    required = {
        "patient_ids",
        "organ_node_names",
        "Z_prime",
        "edge_type_code",
        "prior_edge_mask",
        "candidate_edge_mask",
        "adjacency_logits",
        "adjacency_prob",
    }
    missing = required - set(out.keys())
    if missing:
        raise RuntimeError(f"stage11 pack missing required arrays: {sorted(missing)}")
    return out


def build_edge_feature_tensor(pack, device):
    edge_type_code = torch.from_numpy(pack["edge_type_code"]).long().to(device)
    edge_type_onehot = F.one_hot(edge_type_code, num_classes=4).float()
    adjacency_logits = torch.from_numpy(pack["adjacency_logits"]).float().to(device).unsqueeze(-1)
    adjacency_prob = torch.from_numpy(pack["adjacency_prob"]).float().to(device).unsqueeze(-1)
    prior_edge_mask = torch.from_numpy(pack["prior_edge_mask"]).float().to(device).unsqueeze(-1)
    candidate_edge_mask = torch.from_numpy(pack["candidate_edge_mask"]).float().to(device).unsqueeze(-1)
    return torch.cat(
        [
            edge_type_onehot,
            adjacency_logits,
            adjacency_prob,
            prior_edge_mask,
            candidate_edge_mask,
        ],
        dim=-1,
    )


def load_model_payload(model_path):
    payload = torch.load(str(model_path), map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"invalid model payload: {model_path}")
    return payload


def freeze_model_prefixes(model, freeze_prefixes):
    prefixes = [str(x).strip() for x in freeze_prefixes if str(x).strip()]
    if not prefixes:
        return []
    frozen = []
    for name, param in model.named_parameters():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            param.requires_grad = False
            frozen.append(name)
    return frozen


def build_location_bucket_matrix(organ_node_names, recurrence_classes):
    organ_names = [str(x) for x in organ_node_names]
    matrix = np.zeros((len(recurrence_classes), len(organ_names)), dtype=np.float32)

    def assign_weight(class_idx, names):
        indices = [organ_names.index(name) for name in names if name in organ_names]
        if not indices:
            indices = [idx for idx, name in enumerate(organ_names) if name != "Primary"]
        if not indices:
            indices = list(range(len(organ_names)))
        weight = 1.0 / float(len(indices))
        for idx in indices:
            matrix[class_idx, idx] = weight

    for class_idx, class_name in enumerate(recurrence_classes):
        label = str(class_name).strip().lower()
        if "local" in label or label == "primary":
            assign_weight(class_idx, ["Primary"])
        elif "regional" in label or "lymph" in label or "node" in label:
            assign_weight(class_idx, ["LymphNodeMediastinum"])
        elif "distant" in label or "met" in label:
            assign_weight(class_idx, ["Lung", "Bone", "Liver", "Brain"])
        else:
            assign_weight(class_idx, ["Lung", "Bone", "Liver", "LymphNodeMediastinum", "Brain"])
    return matrix.astype(np.float32)


class LatentDiffusionExplainer(nn.Module):
    def __init__(self, d_model=128, edge_feature_dim=8, hidden_dim=256):
        super().__init__()
        self.organ_head = nn.Sequential(
            nn.Linear(int(d_model), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(int(d_model) * 2 + int(edge_feature_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, z_prime, edge_features, candidate_edge_mask):
        organ_susceptibility = torch.sigmoid(self.organ_head(z_prime)).squeeze(-1)

        batch_size, num_nodes, _ = z_prime.shape
        zi = z_prime.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, -1)
        zj = z_prime.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, -1)
        pair_features = torch.cat(
            [
                zi,
                zj,
                edge_features.unsqueeze(0).expand(batch_size, -1, -1, -1),
            ],
            dim=-1,
        )
        edge_prob = torch.sigmoid(self.edge_head(pair_features)).squeeze(-1)
        edge_prob = edge_prob * candidate_edge_mask.unsqueeze(0).float()
        return organ_susceptibility, edge_prob


class ExplanationGuidedPrimaryModel(nn.Module):
    def __init__(
        self,
        d_model,
        num_nodes,
        organ_node_names,
        recurrence_classes,
        edge_feature_dim,
        pool_mode="attention",
        pool_hidden_dim=128,
        trunk_hidden_dim=128,
        explanation_hidden_dim=256,
        dropout=0.1,
        survival_mode="discrete",
        num_time_bins=8,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_nodes = int(num_nodes)
        self.survival_mode = str(survival_mode)
        self.num_time_bins = int(num_time_bins)
        self.num_recurrence_classes = int(len(recurrence_classes))
        self.organ_node_names = [str(x) for x in organ_node_names]
        if "Primary" not in self.organ_node_names:
            raise RuntimeError("organ_node_names must include Primary")
        self.primary_index = int(self.organ_node_names.index("Primary"))

        if str(pool_mode) == "attention":
            self.pool = PRIMARY_MOD.AttentionPool(d_model=self.d_model, hidden_dim=pool_hidden_dim)
        elif str(pool_mode) == "weighted_sum":
            self.pool = PRIMARY_MOD.WeightedSumPool(num_nodes=self.num_nodes)
        else:
            raise RuntimeError(f"unsupported pool_mode: {pool_mode}")

        self.base_trunk = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, int(trunk_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.explainer = LatentDiffusionExplainer(
            d_model=self.d_model,
            edge_feature_dim=int(edge_feature_dim),
            hidden_dim=int(explanation_hidden_dim),
        )

        fusion_input_dim = int(trunk_hidden_dim) + self.d_model + self.d_model + 4
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, int(trunk_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )

        if self.survival_mode == "cox":
            self.os_head = nn.Linear(int(trunk_hidden_dim), 1)
        elif self.survival_mode == "discrete":
            self.os_head = nn.Linear(int(trunk_hidden_dim), self.num_time_bins)
        else:
            raise RuntimeError(f"unsupported survival_mode: {self.survival_mode}")
        self.recurrence_head = nn.Linear(int(trunk_hidden_dim), 1)
        self.recurrence_location_head = nn.Linear(int(trunk_hidden_dim), self.num_recurrence_classes)
        self.explanation_recurrence_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        non_primary_mask = np.ones((self.num_nodes,), dtype=np.float32)
        non_primary_mask[self.primary_index] = 0.0
        location_bucket_matrix = build_location_bucket_matrix(
            organ_node_names=self.organ_node_names,
            recurrence_classes=recurrence_classes,
        )
        self.register_buffer("non_primary_mask", torch.from_numpy(non_primary_mask).float())
        self.register_buffer("location_bucket_matrix", torch.from_numpy(location_bucket_matrix).float())

    def forward(self, z_prime, edge_features, candidate_edge_mask):
        pooled, pool_weights = self.pool(z_prime)
        base_trunk = self.base_trunk(pooled)

        organ_susceptibility, edge_diffusion_prob = self.explainer(
            z_prime=z_prime,
            edge_features=edge_features,
            candidate_edge_mask=candidate_edge_mask,
        )

        non_primary_mask = self.non_primary_mask.view(1, -1)
        susceptibility_non_primary = organ_susceptibility * non_primary_mask
        susceptibility_norm = susceptibility_non_primary / susceptibility_non_primary.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        susceptibility_context = torch.sum(susceptibility_norm.unsqueeze(-1) * z_prime, dim=1)

        primary_out_edge = edge_diffusion_prob[:, self.primary_index, :] * non_primary_mask
        primary_out_norm = primary_out_edge / primary_out_edge.sum(dim=1, keepdim=True).clamp_min(1e-6)
        edge_context = torch.sum(primary_out_norm.unsqueeze(-1) * z_prime, dim=1)

        diffusion_features = torch.stack(
            [
                susceptibility_non_primary.mean(dim=1),
                susceptibility_non_primary.max(dim=1).values,
                primary_out_edge.mean(dim=1),
                primary_out_edge.max(dim=1).values,
            ],
            dim=1,
        )

        fused_input = torch.cat(
            [
                base_trunk,
                susceptibility_context,
                edge_context,
                diffusion_features,
            ],
            dim=1,
        )
        joint_trunk = self.fusion(fused_input)

        explanation_recurrence_logit = self.explanation_recurrence_head(diffusion_features).squeeze(-1)
        organ_bucket_scores = torch.matmul(organ_susceptibility, self.location_bucket_matrix.t())
        edge_bucket_scores = torch.matmul(primary_out_edge, self.location_bucket_matrix.t())
        explanation_location_scores = 0.5 * organ_bucket_scores + 0.5 * edge_bucket_scores
        explanation_location_logits = torch.logit(
            explanation_location_scores.clamp(min=1e-4, max=1.0 - 1e-4)
        )

        out = {
            "u": pooled,
            "base_trunk": base_trunk,
            "trunk": joint_trunk,
            "pool_weights": pool_weights,
            "organ_susceptibility": organ_susceptibility,
            "edge_diffusion_prob": edge_diffusion_prob,
            "susceptibility_context": susceptibility_context,
            "edge_context": edge_context,
            "diffusion_features": diffusion_features,
            "primary_out_edge": primary_out_edge,
            "explanation_recurrence_logit": explanation_recurrence_logit,
            "explanation_location_logits": explanation_location_logits,
            "recurrence_logit": self.recurrence_head(joint_trunk).squeeze(-1),
            "recurrence_location_logits": self.recurrence_location_head(joint_trunk),
        }
        if self.survival_mode == "cox":
            out["os_log_risk"] = self.os_head(joint_trunk).squeeze(-1)
        else:
            out["hazard_logits"] = self.os_head(joint_trunk)
        return out


def compute_edge_prior_loss(edge_prob, prior_edge_mask, candidate_edge_mask, indices):
    idx = indices
    pred = edge_prob[idx]
    target = prior_edge_mask.unsqueeze(0).expand(pred.shape[0], -1, -1).float()
    valid = candidate_edge_mask.unsqueeze(0).expand(pred.shape[0], -1, -1) > 0
    eye = torch.eye(pred.shape[1], device=pred.device, dtype=torch.bool).unsqueeze(0)
    valid = valid & (~eye)
    if int(valid.sum().item()) == 0:
        return pred.sum() * 0.0

    pred_valid = pred[valid].clamp(min=1e-6, max=1.0 - 1e-6)
    target_valid = target[valid]
    positives = float((target_valid > 0.5).sum().item())
    negatives = float((target_valid <= 0.5).sum().item())
    pos_weight = 1.0 if positives <= 0.0 else max(negatives / positives, 1.0)
    weights = torch.ones_like(pred_valid)
    weights[target_valid > 0.5] = float(pos_weight)
    return F.binary_cross_entropy(pred_valid, target_valid, weight=weights, reduction="mean")


def compute_joint_loss(
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
    loss_weight_expl_rec,
    loss_weight_expl_loc,
    loss_weight_edge_prior,
    prior_edge_mask,
    candidate_edge_mask,
):
    base = PRIMARY_MOD.compute_total_loss(
        outputs=outputs,
        supervision_torch=supervision_torch,
        indices=indices,
        survival_mode=survival_mode,
        bin_edges_torch=bin_edges_torch,
        recurrence_pos_weight=recurrence_pos_weight,
        recurrence_location_class_weights=recurrence_location_class_weights,
        loss_weight_os=loss_weight_os,
        loss_weight_rec=loss_weight_rec,
        loss_weight_loc=loss_weight_loc,
    )
    idx = indices
    rec_known_mask = supervision_torch["rec_label_known"][idx] > 0
    loc_known_mask = supervision_torch["rec_location_known"][idx] > 0
    expl_rec_loss = PRIMARY_MOD.masked_recurrence_bce(
        rec_logit=outputs["explanation_recurrence_logit"][idx],
        event_rec=supervision_torch["event_rec"][idx],
        known_mask=rec_known_mask,
        pos_weight=recurrence_pos_weight,
    )
    expl_loc_loss = PRIMARY_MOD.masked_recurrence_location_ce(
        loc_logits=outputs["explanation_location_logits"][idx],
        loc_target=supervision_torch["rec_location_index"][idx],
        known_mask=loc_known_mask,
        class_weights=recurrence_location_class_weights,
    )
    edge_prior_loss = compute_edge_prior_loss(
        edge_prob=outputs["edge_diffusion_prob"],
        prior_edge_mask=prior_edge_mask,
        candidate_edge_mask=candidate_edge_mask,
        indices=indices,
    )
    total_loss = (
        base["total_loss"]
        + float(loss_weight_expl_rec) * expl_rec_loss
        + float(loss_weight_expl_loc) * expl_loc_loss
        + float(loss_weight_edge_prior) * edge_prior_loss
    )
    base.update(
        {
            "total_loss": total_loss,
            "expl_rec_loss": expl_rec_loss,
            "expl_loc_loss": expl_loc_loss,
            "edge_prior_loss": edge_prior_loss,
        }
    )
    return base


def extract_numpy_outputs(outputs, survival_mode):
    rec_prob = torch.sigmoid(outputs["recurrence_logit"]).detach().cpu().numpy().astype(np.float32)
    rec_location_prob = torch.softmax(outputs["recurrence_location_logits"], dim=1).detach().cpu().numpy().astype(np.float32)
    expl_rec_prob = torch.sigmoid(outputs["explanation_recurrence_logit"]).detach().cpu().numpy().astype(np.float32)
    expl_loc_prob = torch.softmax(outputs["explanation_location_logits"], dim=1).detach().cpu().numpy().astype(np.float32)
    organ_susceptibility = outputs["organ_susceptibility"].detach().cpu().numpy().astype(np.float32)
    edge_diffusion_prob = outputs["edge_diffusion_prob"].detach().cpu().numpy().astype(np.float32)
    pool_weights = outputs["pool_weights"].detach().cpu().numpy().astype(np.float32)
    pooled_u = outputs["u"].detach().cpu().numpy().astype(np.float32)
    base_trunk = outputs["base_trunk"].detach().cpu().numpy().astype(np.float32)
    trunk = outputs["trunk"].detach().cpu().numpy().astype(np.float32)
    susceptibility_context = outputs["susceptibility_context"].detach().cpu().numpy().astype(np.float32)
    edge_context = outputs["edge_context"].detach().cpu().numpy().astype(np.float32)
    diffusion_features = outputs["diffusion_features"].detach().cpu().numpy().astype(np.float32)
    primary_out_edge = outputs["primary_out_edge"].detach().cpu().numpy().astype(np.float32)

    if str(survival_mode) == "cox":
        os_log_risk = outputs["os_log_risk"].detach().cpu().numpy().astype(np.float32)
        hazard_prob = np.zeros((trunk.shape[0], 0), dtype=np.float32)
        survival_curve = np.zeros((trunk.shape[0], 0), dtype=np.float32)
    else:
        os_log_risk = np.zeros((trunk.shape[0],), dtype=np.float32)
        hazard_prob = torch.sigmoid(outputs["hazard_logits"]).detach().cpu().numpy().astype(np.float32)
        survival_curve = np.cumprod(1.0 - hazard_prob, axis=1).astype(np.float32)

    return {
        "rec_prob": rec_prob,
        "rec_location_prob": rec_location_prob,
        "explanation_rec_prob": expl_rec_prob,
        "explanation_location_prob": expl_loc_prob,
        "organ_susceptibility": organ_susceptibility,
        "edge_diffusion_prob": edge_diffusion_prob,
        "pool_weights": pool_weights,
        "pooled_u": pooled_u,
        "base_trunk": base_trunk,
        "trunk": trunk,
        "susceptibility_context": susceptibility_context,
        "edge_context": edge_context,
        "diffusion_features": diffusion_features,
        "primary_out_edge": primary_out_edge,
        "os_log_risk": os_log_risk,
        "hazard_prob": hazard_prob,
        "survival_curve": survival_curve,
    }


def evaluate_explanation_metrics(supervision, split_indices_np, outputs):
    split_indices_np = np.asarray(split_indices_np, dtype=np.int64)
    rec_known_mask = supervision["rec_label_known"][split_indices_np] == 1
    loc_known_mask = supervision["rec_location_known"][split_indices_np] == 1
    rec_auc = PRIMARY_MOD.binary_auc_score(
        y_true=supervision["event_rec"][split_indices_np][rec_known_mask].astype(np.int64).tolist(),
        y_score=outputs["explanation_rec_prob"][split_indices_np][rec_known_mask].tolist(),
    )
    loc_acc = PRIMARY_MOD.multiclass_accuracy_score(
        target_index=supervision["rec_location_index"][split_indices_np][loc_known_mask],
        pred_prob=outputs["explanation_location_prob"][split_indices_np][loc_known_mask],
    )
    return {
        "expl_rec_auc": rec_auc,
        "expl_loc_acc": loc_acc,
    }


def train_explanation_guided_model(
    stage11_pack,
    supervision,
    recurrence_classes,
    output_paths,
    pool_mode="attention",
    survival_mode="discrete",
    num_time_bins=8,
    pool_hidden_dim=128,
    trunk_hidden_dim=128,
    explanation_hidden_dim=256,
    dropout=0.1,
    epochs=120,
    lr=1e-3,
    weight_decay=1e-4,
    val_ratio=0.2,
    early_stop_patience=30,
    early_stop_min_delta=1e-5,
    loss_weight_os=0.5,
    loss_weight_rec=1.0,
    loss_weight_loc=1.0,
    loss_weight_expl_rec=0.5,
    loss_weight_expl_loc=0.5,
    loss_weight_edge_prior=0.1,
    seed=2024,
    device="auto",
    init_model_path=None,
    freeze_prefixes=None,
):
    device = PRIMARY_MOD.choose_device(device)
    set_seed(seed)

    z_prime = stage11_pack["Z_prime"].astype(np.float32)
    train_indices_np, val_indices_np = PRIMARY_MOD.stratified_split_indices(
        PRIMARY_MOD.build_split_strata(supervision),
        val_ratio=val_ratio,
        seed=seed,
    )
    if train_indices_np.size == 0:
        raise RuntimeError("empty train split")

    z_prime_torch = torch.from_numpy(z_prime).float().to(device)
    edge_features_torch = build_edge_feature_tensor(stage11_pack, device=device)
    candidate_edge_mask_torch = torch.from_numpy(stage11_pack["candidate_edge_mask"]).to(device)
    prior_edge_mask_torch = torch.from_numpy(stage11_pack["prior_edge_mask"]).to(device)
    supervision_torch = PRIMARY_MOD.tensorize_supervision(supervision, device=device)
    train_indices = torch.from_numpy(train_indices_np).long().to(device)
    val_indices = torch.from_numpy(val_indices_np).long().to(device)

    if str(survival_mode) == "discrete":
        bin_edges = PRIMARY_MOD.build_time_bin_edges(
            train_time_days=supervision["time_os_days"][train_indices_np],
            train_event_os=supervision["event_os"][train_indices_np],
            num_bins=num_time_bins,
        )
        bin_edges_torch = torch.from_numpy(bin_edges).float().to(device)
    else:
        bin_edges = np.asarray([], dtype=np.float32)
        bin_edges_torch = None

    model = ExplanationGuidedPrimaryModel(
        d_model=int(z_prime.shape[-1]),
        num_nodes=int(z_prime.shape[1]),
        organ_node_names=stage11_pack["organ_node_names"].tolist(),
        recurrence_classes=recurrence_classes,
        edge_feature_dim=int(edge_features_torch.shape[-1]),
        pool_mode=pool_mode,
        pool_hidden_dim=pool_hidden_dim,
        trunk_hidden_dim=trunk_hidden_dim,
        explanation_hidden_dim=explanation_hidden_dim,
        dropout=dropout,
        survival_mode=survival_mode,
        num_time_bins=num_time_bins,
    ).to(device)
    loaded_init_model_path = None
    if init_model_path is not None:
        payload = load_model_payload(init_model_path)
        model.load_state_dict(payload["state_dict"], strict=False)
        loaded_init_model_path = str(init_model_path)
    frozen_parameter_names = freeze_model_prefixes(model, freeze_prefixes or [])

    recurrence_pos_weight = PRIMARY_MOD.compute_binary_pos_weight(
        target=supervision_torch["event_rec"][train_indices],
        known_mask=supervision_torch["rec_label_known"][train_indices] > 0,
    )
    recurrence_location_class_weights = PRIMARY_MOD.compute_multiclass_weights(
        target=supervision_torch["rec_location_index"][train_indices],
        known_mask=supervision_torch["rec_location_known"][train_indices] > 0,
        num_classes=len(recurrence_classes),
    )
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("all model parameters are frozen; nothing to optimize")
    optimizer = torch.optim.Adam(trainable_parameters, lr=float(lr), weight_decay=float(weight_decay))

    history_rows = []
    best_state = PRIMARY_MOD.snapshot_state_dict(model)
    best_epoch = 0
    best_monitor_loss = None
    bad_epochs = 0
    stopped_early = False

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(
            z_prime_torch,
            edge_features=edge_features_torch,
            candidate_edge_mask=candidate_edge_mask_torch,
        )
        train_loss_dict = compute_joint_loss(
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
            loss_weight_expl_rec=loss_weight_expl_rec,
            loss_weight_expl_loc=loss_weight_expl_loc,
            loss_weight_edge_prior=loss_weight_edge_prior,
            prior_edge_mask=prior_edge_mask_torch,
            candidate_edge_mask=candidate_edge_mask_torch,
        )
        train_loss_dict["total_loss"].backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(
                z_prime_torch,
                edge_features=edge_features_torch,
                candidate_edge_mask=candidate_edge_mask_torch,
            )
            full_train_loss_dict = compute_joint_loss(
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
                loss_weight_expl_rec=loss_weight_expl_rec,
                loss_weight_expl_loc=loss_weight_expl_loc,
                loss_weight_edge_prior=loss_weight_edge_prior,
                prior_edge_mask=prior_edge_mask_torch,
                candidate_edge_mask=candidate_edge_mask_torch,
            )
            if val_indices.numel() > 0:
                val_loss_dict = compute_joint_loss(
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
                    loss_weight_expl_rec=loss_weight_expl_rec,
                    loss_weight_expl_loc=loss_weight_expl_loc,
                    loss_weight_edge_prior=loss_weight_edge_prior,
                    prior_edge_mask=prior_edge_mask_torch,
                    candidate_edge_mask=candidate_edge_mask_torch,
                )
                monitor_loss = float(val_loss_dict["total_loss"].item())
            else:
                val_loss_dict = full_train_loss_dict
                monitor_loss = float(full_train_loss_dict["total_loss"].item())

            outputs_np = extract_numpy_outputs(outputs, survival_mode=survival_mode)
            train_metrics = PRIMARY_MOD.evaluate_split_metrics(
                supervision=supervision,
                split_indices_np=train_indices_np,
                outputs=outputs_np,
                survival_mode=survival_mode,
            )
            val_metrics = PRIMARY_MOD.evaluate_split_metrics(
                supervision=supervision,
                split_indices_np=val_indices_np,
                outputs=outputs_np,
                survival_mode=survival_mode,
            ) if val_indices_np.size > 0 else train_metrics
            train_expl_metrics = evaluate_explanation_metrics(
                supervision=supervision,
                split_indices_np=train_indices_np,
                outputs=outputs_np,
            )
            val_expl_metrics = evaluate_explanation_metrics(
                supervision=supervision,
                split_indices_np=val_indices_np,
                outputs=outputs_np,
            ) if val_indices_np.size > 0 else train_expl_metrics

        improved = False
        if best_monitor_loss is None or monitor_loss < (best_monitor_loss - float(early_stop_min_delta)):
            best_monitor_loss = monitor_loss
            best_epoch = int(epoch)
            best_state = PRIMARY_MOD.snapshot_state_dict(model)
            bad_epochs = 0
            improved = True
        else:
            bad_epochs += 1

        history_rows.append(
            {
                "epoch": int(epoch),
                "train_total_loss": float(full_train_loss_dict["total_loss"].item()),
                "train_os_loss": float(full_train_loss_dict["os_loss"].item()),
                "train_rec_loss": float(full_train_loss_dict["rec_loss"].item()),
                "train_loc_loss": float(full_train_loss_dict["loc_loss"].item()),
                "train_expl_rec_loss": float(full_train_loss_dict["expl_rec_loss"].item()),
                "train_expl_loc_loss": float(full_train_loss_dict["expl_loc_loss"].item()),
                "train_edge_prior_loss": float(full_train_loss_dict["edge_prior_loss"].item()),
                "train_c_index": "" if train_metrics["val_c_index"] is None else float(train_metrics["val_c_index"]),
                "train_rec_auc": "" if train_metrics["val_rec_auc"] is None else float(train_metrics["val_rec_auc"]),
                "train_loc_acc": "" if train_metrics["val_loc_acc"] is None else float(train_metrics["val_loc_acc"]),
                "train_expl_rec_auc": "" if train_expl_metrics["expl_rec_auc"] is None else float(train_expl_metrics["expl_rec_auc"]),
                "train_expl_loc_acc": "" if train_expl_metrics["expl_loc_acc"] is None else float(train_expl_metrics["expl_loc_acc"]),
                "val_total_loss": float(val_loss_dict["total_loss"].item()),
                "val_os_loss": float(val_loss_dict["os_loss"].item()),
                "val_rec_loss": float(val_loss_dict["rec_loss"].item()),
                "val_loc_loss": float(val_loss_dict["loc_loss"].item()),
                "val_expl_rec_loss": float(val_loss_dict["expl_rec_loss"].item()),
                "val_expl_loc_loss": float(val_loss_dict["expl_loc_loss"].item()),
                "val_edge_prior_loss": float(val_loss_dict["edge_prior_loss"].item()),
                "val_c_index": "" if val_metrics["val_c_index"] is None else float(val_metrics["val_c_index"]),
                "val_rec_auc": "" if val_metrics["val_rec_auc"] is None else float(val_metrics["val_rec_auc"]),
                "val_loc_acc": "" if val_metrics["val_loc_acc"] is None else float(val_metrics["val_loc_acc"]),
                "val_expl_rec_auc": "" if val_expl_metrics["expl_rec_auc"] is None else float(val_expl_metrics["expl_rec_auc"]),
                "val_expl_loc_acc": "" if val_expl_metrics["expl_loc_acc"] is None else float(val_expl_metrics["expl_loc_acc"]),
                "monitor_loss": float(monitor_loss),
                "is_best_epoch": 1 if improved else 0,
                "bad_epochs": int(bad_epochs),
            }
        )

        if int(bad_epochs) >= int(early_stop_patience):
            stopped_early = True
            break

    PRIMARY_MOD.restore_state_dict(model, best_state)
    model.eval()
    with torch.no_grad():
        final_outputs = model(
            z_prime_torch,
            edge_features=edge_features_torch,
            candidate_edge_mask=candidate_edge_mask_torch,
        )
        outputs_np = extract_numpy_outputs(final_outputs, survival_mode=survival_mode)

    split_name = np.asarray(["train"] * z_prime.shape[0], dtype=object)
    split_name[val_indices_np] = "val"

    model_payload = {
        "state_dict": model.state_dict(),
        "config": {
            "d_model": int(z_prime.shape[-1]),
            "num_nodes": int(z_prime.shape[1]),
            "organ_node_names": stage11_pack["organ_node_names"].astype(str).tolist(),
            "recurrence_classes": list(recurrence_classes),
            "pool_mode": str(pool_mode),
            "pool_hidden_dim": int(pool_hidden_dim),
            "trunk_hidden_dim": int(trunk_hidden_dim),
            "explanation_hidden_dim": int(explanation_hidden_dim),
            "dropout": float(dropout),
            "survival_mode": str(survival_mode),
            "num_time_bins": int(num_time_bins),
            "edge_feature_dim": int(edge_features_torch.shape[-1]),
            "bin_edges": bin_edges.tolist(),
        },
    }
    torch.save(model_payload, output_paths["model_path"])

    PRIMARY_MOD.write_csv(
        output_paths["train_csv"],
        [
            "epoch",
            "train_total_loss",
            "train_os_loss",
            "train_rec_loss",
            "train_loc_loss",
            "train_expl_rec_loss",
            "train_expl_loc_loss",
            "train_edge_prior_loss",
            "train_c_index",
            "train_rec_auc",
            "train_loc_acc",
            "train_expl_rec_auc",
            "train_expl_loc_acc",
            "val_total_loss",
            "val_os_loss",
            "val_rec_loss",
            "val_loc_loss",
            "val_expl_rec_loss",
            "val_expl_loc_loss",
            "val_edge_prior_loss",
            "val_c_index",
            "val_rec_auc",
            "val_loc_acc",
            "val_expl_rec_auc",
            "val_expl_loc_acc",
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
        "loaded_init_model_path": loaded_init_model_path,
        "frozen_parameter_names": frozen_parameter_names,
        "trainable_parameter_count": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
        "frozen_parameter_count": int(sum(param.numel() for param in model.parameters() if not param.requires_grad)),
        **outputs_np,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 12.2 latent diffusion explanation training",
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
    parser.add_argument("--explanation-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5)
    parser.add_argument("--loss-weight-os", type=float, default=0.5)
    parser.add_argument("--loss-weight-rec", type=float, default=1.0)
    parser.add_argument("--loss-weight-loc", type=float, default=1.0)
    parser.add_argument("--loss-weight-expl-rec", type=float, default=0.5)
    parser.add_argument("--loss-weight-expl-loc", type=float, default=0.5)
    parser.add_argument("--loss-weight-edge-prior", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--init-model-path", type=str, default="")
    parser.add_argument(
        "--freeze-prefixes",
        type=str,
        default="",
        help="Comma-separated module prefixes to freeze, e.g. pool,base_trunk",
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

    stage11_pack_path = resolve_required_path(args.stage11_pack, DEFAULT_STAGE11_PACK, "stage11 pack")
    labels_csv_path = PRIMARY_MOD.resolve_labels_path(args.labels_csv)
    output_paths = ensure_output_dirs(args.output_root)

    print(f"[start] stage11_pack={stage11_pack_path}")
    print(f"[start] labels_csv={labels_csv_path}")
    print(f"[start] output_root={output_paths['root']}")
    print(
        f"[start] pool_mode={args.pool_mode} survival_mode={args.survival_mode} "
        f"epochs={args.epochs} seed={args.seed}"
    )

    stage11_pack = load_stage11_pack(stage11_pack_path)
    labels_by_patient = PRIMARY_MOD.load_label_rows(labels_csv_path)
    recurrence_classes = PRIMARY_MOD.resolve_recurrence_classes(labels_by_patient, args.recurrence_classes)
    supervision = PRIMARY_MOD.build_supervision_arrays(
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

    train_result = train_explanation_guided_model(
        stage11_pack=stage11_pack,
        supervision=supervision,
        recurrence_classes=recurrence_classes,
        output_paths=output_paths,
        pool_mode=args.pool_mode,
        survival_mode=args.survival_mode,
        num_time_bins=args.num_time_bins,
        pool_hidden_dim=args.pool_hidden_dim,
        trunk_hidden_dim=args.trunk_hidden_dim,
        explanation_hidden_dim=args.explanation_hidden_dim,
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
        loss_weight_expl_rec=args.loss_weight_expl_rec,
        loss_weight_expl_loc=args.loss_weight_expl_loc,
        loss_weight_edge_prior=args.loss_weight_edge_prior,
        seed=args.seed,
        device=args.device,
        init_model_path=(Path(args.init_model_path) if str(args.init_model_path).strip() else None),
        freeze_prefixes=[x.strip() for x in str(args.freeze_prefixes).split(",") if x.strip()],
    )

    PRIMARY_MOD.write_prediction_csv(
        path=output_paths["pred_csv"],
        patient_ids=stage11_pack["patient_ids"],
        split_name=train_result["split_name"],
        supervision=supervision,
        recurrence_classes=recurrence_classes,
        survival_mode=args.survival_mode,
        outputs=train_result,
    )

    np.savez_compressed(
        output_paths["primary_npz"],
        patient_ids=stage11_pack["patient_ids"].astype(str),
        organ_node_names=stage11_pack["organ_node_names"].astype(str),
        split_name=train_result["split_name"].astype(str),
        recurrence_classes=np.asarray(recurrence_classes).astype(str),
        pool_weights=train_result["pool_weights"].astype(np.float32),
        pooled_u=train_result["pooled_u"].astype(np.float32),
        base_trunk=train_result["base_trunk"].astype(np.float32),
        trunk=train_result["trunk"].astype(np.float32),
        os_log_risk=train_result["os_log_risk"].astype(np.float32),
        hazard_prob=train_result["hazard_prob"].astype(np.float32),
        survival_curve=train_result["survival_curve"].astype(np.float32),
        recurrence_probability=train_result["rec_prob"].astype(np.float32),
        recurrence_location_probability=train_result["rec_location_prob"].astype(np.float32),
        explanation_recurrence_probability=train_result["explanation_rec_prob"].astype(np.float32),
        explanation_location_probability=train_result["explanation_location_prob"].astype(np.float32),
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

    np.savez_compressed(
        output_paths["graph_npz"],
        patient_ids=stage11_pack["patient_ids"].astype(str),
        organ_node_names=stage11_pack["organ_node_names"].astype(str),
        Z_prime=stage11_pack["Z_prime"].astype(np.float32),
        organ_susceptibility=train_result["organ_susceptibility"].astype(np.float32),
        edge_diffusion_prob=train_result["edge_diffusion_prob"].astype(np.float32),
        edge_type_code=stage11_pack["edge_type_code"].astype(np.uint8),
        prior_edge_mask=stage11_pack["prior_edge_mask"].astype(np.uint8),
        candidate_edge_mask=stage11_pack["candidate_edge_mask"].astype(np.uint8),
        adjacency_logits=stage11_pack["adjacency_logits"].astype(np.float32),
        adjacency_prob=stage11_pack["adjacency_prob"].astype(np.float32),
        split_name=train_result["split_name"].astype(str),
        explanation_recurrence_probability=train_result["explanation_rec_prob"].astype(np.float32),
        explanation_location_probability=train_result["explanation_location_prob"].astype(np.float32),
        diffusion_features=train_result["diffusion_features"].astype(np.float32),
        susceptibility_context=train_result["susceptibility_context"].astype(np.float32),
        edge_context=train_result["edge_context"].astype(np.float32),
        primary_out_edge=train_result["primary_out_edge"].astype(np.float32),
    )

    graph_summary = {
        "stage": "12.2_explanation_training",
        "stage11_pack_path": str(stage11_pack_path),
        "patient_count": int(len(stage11_pack["patient_ids"])),
        "num_nodes": int(stage11_pack["Z_prime"].shape[1]),
        "d_model": int(stage11_pack["Z_prime"].shape[2]),
        "pool_mode": str(args.pool_mode),
        "survival_mode": str(args.survival_mode),
        "num_time_bins": int(args.num_time_bins),
        "seed": int(args.seed),
        "random_init_only": False,
        "explanation_semantics": (
            "Latent diffusion explanation only; organ/path outputs are model-induced explanation layers "
            "constrained by OS/recurrence tasks and do not imply organ-level ground-truth supervision."
        ),
        "best_epoch": int(train_result["best_epoch"]),
        "best_monitor_loss": train_result["best_monitor_loss"],
        "ranges": {
            "organ_susceptibility_min": float(train_result["organ_susceptibility"].min()),
            "organ_susceptibility_max": float(train_result["organ_susceptibility"].max()),
            "edge_diffusion_prob_min": float(train_result["edge_diffusion_prob"].min()),
            "edge_diffusion_prob_max": float(train_result["edge_diffusion_prob"].max()),
        },
        "shapes": {
            "Z_prime": list(stage11_pack["Z_prime"].shape),
            "organ_susceptibility": list(train_result["organ_susceptibility"].shape),
            "edge_diffusion_prob": list(train_result["edge_diffusion_prob"].shape),
        },
    }
    PRIMARY_MOD.write_json(output_paths["graph_summary_json"], graph_summary)

    final_metrics = PRIMARY_MOD.evaluate_split_metrics(
        supervision=supervision,
        split_indices_np=train_result["val_indices"],
        outputs=train_result,
        survival_mode=args.survival_mode,
    )
    final_expl_metrics = evaluate_explanation_metrics(
        supervision=supervision,
        split_indices_np=train_result["val_indices"],
        outputs=train_result,
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
        "explanation_hidden_dim": int(args.explanation_hidden_dim),
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
        "loss_weight_expl_rec": float(args.loss_weight_expl_rec),
        "loss_weight_expl_loc": float(args.loss_weight_expl_loc),
        "loss_weight_edge_prior": float(args.loss_weight_edge_prior),
        "seed": int(args.seed),
        "device": str(args.device),
        "init_model_path": train_result["loaded_init_model_path"],
        "freeze_prefixes": [x.strip() for x in str(args.freeze_prefixes).split(",") if x.strip()],
        "trainable_parameter_count": int(train_result["trainable_parameter_count"]),
        "frozen_parameter_count": int(train_result["frozen_parameter_count"]),
        "best_epoch": int(train_result["best_epoch"]),
        "best_monitor_loss": train_result["best_monitor_loss"],
        "stopped_early": bool(train_result["stopped_early"]),
        "val_metrics": final_metrics,
        "val_explanation_metrics": final_expl_metrics,
        "explanation_semantics": graph_summary["explanation_semantics"],
    }
    PRIMARY_MOD.write_json(output_paths["meta_json"], meta)

    print(f"wrote: {output_paths['model_path']}")
    print(f"wrote: {output_paths['train_csv']}")
    print(f"wrote: {output_paths['meta_json']}")
    print(f"wrote: {output_paths['pred_csv']}")
    print(f"wrote: {output_paths['primary_npz']}")
    print(f"wrote: {output_paths['graph_npz']}")
    print(f"wrote: {output_paths['graph_summary_json']}")
    print(f"best_epoch: {train_result['best_epoch']}")
    print(f"val_metrics: {final_metrics}")
    print(f"val_explanation_metrics: {final_expl_metrics}")
    print("complete")


if __name__ == "__main__":
    main()
