import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

##

DEFAULT_STAGE11_PACK = Path("output/stage11/11.1_graph_construction/graph_construction_pack.npz")
DEFAULT_OUTPUT_ROOT = Path("output/stage11/11.2_graph_reasoning")
ORGAN_NODE_NAMES = [
    "Primary",
    "Lung",
    "Bone",
    "Liver",
    "LymphNodeMediastinum",
    "Brain",
]
EDGE_TYPE_NAME = {
    0: "none",
    1: "self",
    2: "strong",
    3: "weak",
}


def resolve_required_path(path_arg, default_path, label):
    path = Path(path_arg) if str(path_arg).strip() else Path(default_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def ensure_output_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def choose_device(device_arg):
    value = str(device_arg).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
        "Z",
        "edge_type_code",
        "prior_edge_mask",
        "candidate_edge_mask",
        "prior_edge_logits",
        "residual_logits",
        "adjacency_logits",
        "adjacency_prob",
    }
    missing = required - set(out.keys())
    if missing:
        raise RuntimeError(f"stage11 pack missing required arrays: {sorted(missing)}")
    return out


def validate_organ_order(organ_node_names):
    observed = [str(x) for x in organ_node_names.tolist()]
    expected = list(ORGAN_NODE_NAMES)
    if observed != expected:
        raise RuntimeError(f"unexpected organ node order: observed={observed} expected={expected}")


def build_edge_feature_tensor(pack, device):
    edge_type_code = torch.from_numpy(pack["edge_type_code"]).long().to(device)
    edge_type_onehot = F.one_hot(edge_type_code, num_classes=4).float()
    prior_edge_logits = torch.from_numpy(pack["prior_edge_logits"]).float().to(device).unsqueeze(-1)
    residual_logits = torch.from_numpy(pack["residual_logits"]).float().to(device).unsqueeze(-1)
    adjacency_logits = torch.from_numpy(pack["adjacency_logits"]).float().to(device).unsqueeze(-1)
    prior_edge_mask = torch.from_numpy(pack["prior_edge_mask"]).float().to(device).unsqueeze(-1)
    candidate_edge_mask = torch.from_numpy(pack["candidate_edge_mask"]).float().to(device).unsqueeze(-1)
    return torch.cat(
        [
            edge_type_onehot,
            prior_edge_logits,
            residual_logits,
            adjacency_logits,
            prior_edge_mask,
            candidate_edge_mask,
        ],
        dim=-1,
    )


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model=128, num_heads=8, ffn_hidden_dim=256, dropout=0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ffn_hidden_dim = int(ffn_hidden_dim)
        self.dropout = float(dropout)

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, got d_model={self.d_model} num_heads={self.num_heads}"
            )
        self.head_dim = self.d_model // self.num_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.ln_attn = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.ffn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.ffn_hidden_dim, self.d_model),
            nn.Dropout(self.dropout),
        )
        self.ln_ffn = nn.LayerNorm(self.d_model)

    def forward(self, x, adjacency_logits, candidate_edge_mask):
        if x.ndim != 3:
            raise RuntimeError(f"x must be [B,N,D], got shape={tuple(x.shape)}")
        batch_size, num_nodes, d_model = x.shape
        if d_model != self.d_model:
            raise RuntimeError(f"d_model mismatch: expected={self.d_model} got={d_model}")

        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        scores = scores + adjacency_logits.unsqueeze(0).unsqueeze(0)

        candidate_edge_mask = candidate_edge_mask.to(torch.bool)
        if candidate_edge_mask.ndim != 2 or tuple(candidate_edge_mask.shape) != (num_nodes, num_nodes):
            raise RuntimeError(
                f"candidate_edge_mask must be [N,N], got shape={tuple(candidate_edge_mask.shape)}"
            )
        scores = scores.masked_fill(~candidate_edge_mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(~candidate_edge_mask.unsqueeze(0).unsqueeze(0), 0.0)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.d_model)
        attn_out = self.o_proj(context)

        z_attn = self.ln_attn(x + attn_out)
        z_ffn = self.ffn(z_attn)
        out = self.ln_ffn(z_attn + z_ffn)
        return out, attn.mean(dim=1)


class OrganGraphReasoner(nn.Module):
    def __init__(self, d_model=128, num_heads=8, num_layers=2, ffn_hidden_dim=256, dropout=0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, z, adjacency_logits, candidate_edge_mask):
        attention_maps = []
        x = z
        for layer in self.layers:
            x, attn = layer(x, adjacency_logits=adjacency_logits, candidate_edge_mask=candidate_edge_mask)
            attention_maps.append(attn)
        return x, torch.stack(attention_maps, dim=1)


class LatentDiffusionExplainer(nn.Module):
    def __init__(self, d_model=128, edge_feature_dim=9, hidden_dim=256):
        super().__init__()
        self.d_model = int(d_model)
        self.edge_feature_dim = int(edge_feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.organ_head = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(self.d_model * 2 + self.edge_feature_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
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


def derive_topk_paths_for_patient(
    p_edge,
    organ_node_names,
    top_k=3,
    max_hops=3,
    beam_width=8,
    source_name="Primary",
):
    organ_node_names = list(organ_node_names)
    source_index = organ_node_names.index(source_name)

    beam = [([source_index], 0.0)]
    finished = []
    num_nodes = len(organ_node_names)

    for _hop in range(int(max_hops)):
        expansions = []
        for path, log_score in beam:
            src = path[-1]
            for dst in range(num_nodes):
                if dst == src or dst in path:
                    continue
                prob = float(p_edge[src, dst])
                if prob <= 0.0:
                    continue
                next_path = path + [dst]
                next_log_score = log_score + math.log(max(prob, 1e-12))
                finished.append((next_path, next_log_score))
                expansions.append((next_path, next_log_score))
        expansions.sort(key=lambda item: item[1], reverse=True)
        beam = expansions[: max(int(beam_width), int(top_k))]
        if not beam:
            break

    finished.sort(key=lambda item: item[1], reverse=True)
    dedup = []
    seen = set()
    for path, log_score in finished:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(
            {
                "path_indices": [int(x) for x in path],
                "path_names": [organ_node_names[x] for x in path],
                "num_hops": int(len(path) - 1),
                "score_log": float(log_score),
                "score_prob": float(math.exp(log_score)),
            }
        )
        if len(dedup) >= int(top_k):
            break
    return dedup


def write_organ_susceptibility_csv(path, patient_ids, organ_node_names, organ_susceptibility):
    fieldnames = ["patient_id", "organ_index", "organ_name", "susceptibility"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for patient_idx, patient_id in enumerate(patient_ids.tolist()):
            for organ_idx, organ_name in enumerate(organ_node_names.tolist()):
                writer.writerow(
                    {
                        "patient_id": patient_id,
                        "organ_index": organ_idx,
                        "organ_name": organ_name,
                        "susceptibility": float(organ_susceptibility[patient_idx, organ_idx]),
                    }
                )


def write_edge_diffusion_csv(path, patient_ids, organ_node_names, edge_type_code, prior_edge_mask, edge_prob):
    fieldnames = [
        "patient_id",
        "src_index",
        "src_name",
        "dst_index",
        "dst_name",
        "edge_type",
        "is_prior_edge",
        "edge_diffusion_prob",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for patient_idx, patient_id in enumerate(patient_ids.tolist()):
            for src_idx, src_name in enumerate(organ_node_names.tolist()):
                for dst_idx, dst_name in enumerate(organ_node_names.tolist()):
                    writer.writerow(
                        {
                            "patient_id": patient_id,
                            "src_index": src_idx,
                            "src_name": src_name,
                            "dst_index": dst_idx,
                            "dst_name": dst_name,
                            "edge_type": EDGE_TYPE_NAME[int(edge_type_code[src_idx, dst_idx])],
                            "is_prior_edge": int(prior_edge_mask[src_idx, dst_idx]),
                            "edge_diffusion_prob": float(edge_prob[patient_idx, src_idx, dst_idx]),
                        }
                    )


def run_stage11_graph_reasoning(
    stage11_pack_path,
    output_root,
    d_model=128,
    num_heads=8,
    num_layers=2,
    ffn_hidden_dim=256,
    dropout=0.1,
    explanation_hidden_dim=256,
    top_k=3,
    max_hops=3,
    beam_width=8,
    device="auto",
    seed=1337,
):
    stage11_pack_path = Path(stage11_pack_path)
    output_root = Path(output_root)
    ensure_output_dir(output_root)
    device = choose_device(device)
    set_seed(seed)

    pack = load_stage11_pack(stage11_pack_path)
    validate_organ_order(pack["organ_node_names"])

    z = torch.from_numpy(pack["Z"]).float().to(device)
    adjacency_logits = torch.from_numpy(pack["adjacency_logits"]).float().to(device)
    candidate_edge_mask = torch.from_numpy(pack["candidate_edge_mask"]).to(device)
    edge_features = build_edge_feature_tensor(pack, device=device)

    if z.shape[-1] != int(d_model):
        raise RuntimeError(f"d_model mismatch: requested={d_model} but pack Z dim={z.shape[-1]}")

    reasoner = OrganGraphReasoner(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout=dropout,
    ).to(device)
    explainer = LatentDiffusionExplainer(
        d_model=d_model,
        edge_feature_dim=int(edge_features.shape[-1]),
        hidden_dim=explanation_hidden_dim,
    ).to(device)
    reasoner.eval()
    explainer.eval()

    with torch.no_grad():
        z_prime, attention_maps = reasoner(
            z=z,
            adjacency_logits=adjacency_logits,
            candidate_edge_mask=candidate_edge_mask,
        )
        organ_susceptibility, edge_diffusion_prob = explainer(
            z_prime=z_prime,
            edge_features=edge_features,
            candidate_edge_mask=candidate_edge_mask,
        )

    z_prime_np = z_prime.detach().cpu().numpy().astype(np.float32)
    attention_maps_np = attention_maps.detach().cpu().numpy().astype(np.float32)
    organ_susceptibility_np = organ_susceptibility.detach().cpu().numpy().astype(np.float32)
    edge_diffusion_prob_np = edge_diffusion_prob.detach().cpu().numpy().astype(np.float32)

    output_npz = output_root / "graph_reasoning_pack.npz"
    np.savez_compressed(
        output_npz,
        patient_ids=pack["patient_ids"],
        organ_node_names=pack["organ_node_names"],
        Z_prime=z_prime_np,
        graph_attention_mean=attention_maps_np,
        organ_susceptibility=organ_susceptibility_np,
        edge_diffusion_prob=edge_diffusion_prob_np,
        edge_type_code=pack["edge_type_code"].astype(np.uint8),
        prior_edge_mask=pack["prior_edge_mask"].astype(np.uint8),
        candidate_edge_mask=pack["candidate_edge_mask"].astype(np.uint8),
        adjacency_logits=pack["adjacency_logits"].astype(np.float32),
        adjacency_prob=pack["adjacency_prob"].astype(np.float32),
    )

    organ_csv = output_root / "organ_susceptibility.csv"
    write_organ_susceptibility_csv(
        organ_csv,
        patient_ids=pack["patient_ids"],
        organ_node_names=pack["organ_node_names"],
        organ_susceptibility=organ_susceptibility_np,
    )

    edge_csv = output_root / "edge_diffusion_long.csv"
    write_edge_diffusion_csv(
        edge_csv,
        patient_ids=pack["patient_ids"],
        organ_node_names=pack["organ_node_names"],
        edge_type_code=pack["edge_type_code"],
        prior_edge_mask=pack["prior_edge_mask"],
        edge_prob=edge_diffusion_prob_np,
    )

    topk_paths = []
    organ_node_names = [str(x) for x in pack["organ_node_names"].tolist()]
    for patient_idx, patient_id in enumerate(pack["patient_ids"].tolist()):
        topk_paths.append(
            {
                "patient_id": str(patient_id),
                "paths": derive_topk_paths_for_patient(
                    p_edge=edge_diffusion_prob_np[patient_idx],
                    organ_node_names=organ_node_names,
                    top_k=top_k,
                    max_hops=max_hops,
                    beam_width=beam_width,
                    source_name="Primary",
                ),
            }
        )
    topk_json = output_root / "topk_paths.json"
    topk_json.write_text(json.dumps(topk_paths, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "stage": "11.2_graph_reasoning",
        "stage11_pack_path": str(stage11_pack_path),
        "patient_count": int(pack["Z"].shape[0]),
        "num_nodes": int(pack["Z"].shape[1]),
        "d_model": int(d_model),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "ffn_hidden_dim": int(ffn_hidden_dim),
        "dropout": float(dropout),
        "explanation_hidden_dim": int(explanation_hidden_dim),
        "top_k": int(top_k),
        "max_hops": int(max_hops),
        "beam_width": int(beam_width),
        "device": str(device),
        "seed": int(seed),
        "random_init_only": True,
        "explanation_semantics": (
            "Latent diffusion explanation only; do not claim organ-level ground-truth supervision."
        ),
        "shapes": {
            "Z_prime": list(z_prime_np.shape),
            "graph_attention_mean": list(attention_maps_np.shape),
            "organ_susceptibility": list(organ_susceptibility_np.shape),
            "edge_diffusion_prob": list(edge_diffusion_prob_np.shape),
        },
        "ranges": {
            "organ_susceptibility_min": float(organ_susceptibility_np.min()),
            "organ_susceptibility_max": float(organ_susceptibility_np.max()),
            "edge_diffusion_prob_min": float(edge_diffusion_prob_np.min()),
            "edge_diffusion_prob_max": float(edge_diffusion_prob_np.max()),
        },
    }
    summary_path = output_root / "graph_reasoning_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "pack_path": str(output_npz),
        "organ_csv_path": str(organ_csv),
        "edge_csv_path": str(edge_csv),
        "topk_json_path": str(topk_json),
        "summary_path": str(summary_path),
        "patient_count": int(pack["Z"].shape[0]),
        "Z_prime_shape": tuple(z_prime_np.shape),
        "edge_diffusion_shape": tuple(edge_diffusion_prob_np.shape),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Stage 11.2 graph reasoning",
        allow_abbrev=False,
    )
    parser.add_argument("--stage11-pack", default=str(DEFAULT_STAGE11_PACK))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--explanation-hidden-dim", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def parse_args():
    args, _unknown = build_arg_parser().parse_known_args()
    return args


def main():
    args = parse_args()
    stage11_pack = resolve_required_path(args.stage11_pack, DEFAULT_STAGE11_PACK, "stage11 pack")
    result = run_stage11_graph_reasoning(
        stage11_pack_path=stage11_pack,
        output_root=Path(args.output_root),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_hidden_dim=args.ffn_hidden_dim,
        dropout=args.dropout,
        explanation_hidden_dim=args.explanation_hidden_dim,
        top_k=args.top_k,
        max_hops=args.max_hops,
        beam_width=args.beam_width,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
