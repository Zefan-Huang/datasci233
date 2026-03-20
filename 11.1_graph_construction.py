import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

## GNN, important, making node and edges. I Only made 6 nodes: Primary; Lung; Bone; Liver; LymphNodeMediastinum; Brain
## this is kinda my priori. Maybe i will do more nodes in the future depending on how good the results.

DEFAULT_STAGE10_NPZ = Path("output/stage10/10.1_multimodal_fusion/fused_organ_tokens.npz")
DEFAULT_OUTPUT_ROOT = Path("output/stage11/11.1_graph_construction")

ORGAN_NODE_NAMES = [
    "Primary",
    "Lung",
    "Bone",
    "Liver",
    "LymphNodeMediastinum",
    "Brain",
]

EDGE_TYPE_NONE = 0
EDGE_TYPE_SELF = 1
EDGE_TYPE_STRONG = 2
EDGE_TYPE_WEAK = 3
EDGE_TYPE_NAME = {
    EDGE_TYPE_NONE: "none",
    EDGE_TYPE_SELF: "self",
    EDGE_TYPE_STRONG: "strong",
    EDGE_TYPE_WEAK: "weak",
}
SPARSE_PRIOR_EDGES = [
    ("Primary", "Lung", EDGE_TYPE_STRONG),
    ("Lung", "Primary", EDGE_TYPE_STRONG),
    ("Primary", "LymphNodeMediastinum", EDGE_TYPE_STRONG),
    ("LymphNodeMediastinum", "Primary", EDGE_TYPE_STRONG),
    ("Primary", "Bone", EDGE_TYPE_WEAK),
    ("Primary", "Liver", EDGE_TYPE_WEAK),
    ("Primary", "Brain", EDGE_TYPE_WEAK),
    ("LymphNodeMediastinum", "Bone", EDGE_TYPE_WEAK),
    ("LymphNodeMediastinum", "Liver", EDGE_TYPE_WEAK),
    ("LymphNodeMediastinum", "Brain", EDGE_TYPE_WEAK),
]


def resolve_required_path(path_arg, default_path, label):
    path = Path(path_arg) if str(path_arg).strip() else Path(default_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def ensure_output_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def load_stage10_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as z:
        out = {}
        for key in z.files:
            value = z[key]
            if value.dtype.kind in {"U", "O"}:
                out[key] = value.astype(str)
            else:
                out[key] = value
    required = {"patient_ids", "organ_node_names", "Z"}
    missing = required - set(out.keys())
    if missing:
        raise RuntimeError(f"stage10 npz missing required arrays: {sorted(missing)}")
    return out


def validate_organ_order(organ_node_names):
    observed = [str(x) for x in organ_node_names.tolist()]
    expected = list(ORGAN_NODE_NAMES)
    if observed != expected:
        raise RuntimeError(f"unexpected organ node order: observed={observed} expected={expected}")


def matrix_to_edge_index(mask):
    row, col = np.nonzero(mask.astype(np.uint8))
    return np.stack([row, col], axis=0).astype(np.int64)


class WeakPriorGraphConstructor(nn.Module):
    def __init__(
        self,
        organ_node_names=None,
        self_logit=3.0,
        strong_logit=1.5,
        weak_logit=-0.5,
        nonprior_logit=-2.0,
        allow_nonprior_residual=True,
    ):
        super().__init__()
        self.organ_node_names = list(organ_node_names or ORGAN_NODE_NAMES)
        self.num_nodes = len(self.organ_node_names)
        self.self_logit = float(self_logit)
        self.strong_logit = float(strong_logit)
        self.weak_logit = float(weak_logit)
        self.nonprior_logit = float(nonprior_logit)
        self.allow_nonprior_residual = bool(allow_nonprior_residual)

        node_to_index = {name: idx for idx, name in enumerate(self.organ_node_names)}

        edge_type_code = np.full((self.num_nodes, self.num_nodes), EDGE_TYPE_NONE, dtype=np.uint8)
        for idx in range(self.num_nodes):
            edge_type_code[idx, idx] = EDGE_TYPE_SELF
        for src_name, dst_name, edge_type in SPARSE_PRIOR_EDGES:
            src_idx = node_to_index[src_name]
            dst_idx = node_to_index[dst_name]
            edge_type_code[src_idx, dst_idx] = edge_type

        prior_edge_mask = edge_type_code != EDGE_TYPE_NONE
        prior_edge_logits = np.full(
            (self.num_nodes, self.num_nodes),
            self.nonprior_logit,
            dtype=np.float32,
        )
        prior_edge_logits[edge_type_code == EDGE_TYPE_SELF] = self.self_logit
        prior_edge_logits[edge_type_code == EDGE_TYPE_STRONG] = self.strong_logit
        prior_edge_logits[edge_type_code == EDGE_TYPE_WEAK] = self.weak_logit

        if self.allow_nonprior_residual:
            residual_mask = np.ones((self.num_nodes, self.num_nodes), dtype=np.float32)
        else:
            residual_mask = prior_edge_mask.astype(np.float32)

        self.register_buffer("edge_type_code", torch.from_numpy(edge_type_code))
        self.register_buffer("prior_edge_mask", torch.from_numpy(prior_edge_mask.astype(np.uint8)))
        self.register_buffer("prior_edge_logits", torch.from_numpy(prior_edge_logits))
        self.register_buffer("residual_mask", torch.from_numpy(residual_mask))
        self.residual_logits = nn.Parameter(torch.zeros(self.num_nodes, self.num_nodes))

    def forward(self, batch_size=None):
        residual_logits = self.residual_logits * self.residual_mask
        adjacency_logits = self.prior_edge_logits + residual_logits
        adjacency_prob = torch.sigmoid(adjacency_logits)
        candidate_edge_mask = (self.residual_mask > 0).to(torch.uint8)

        out = {
            "edge_type_code": self.edge_type_code,
            "prior_edge_mask": self.prior_edge_mask,
            "prior_edge_logits": self.prior_edge_logits,
            "candidate_edge_mask": candidate_edge_mask,
            "residual_logits": residual_logits,
            "adjacency_logits": adjacency_logits,
            "adjacency_prob": adjacency_prob,
        }
        if batch_size is not None:
            batch_size = int(batch_size)
            out["adjacency_logits_batch"] = adjacency_logits.unsqueeze(0).expand(batch_size, -1, -1)
            out["adjacency_prob_batch"] = adjacency_prob.unsqueeze(0).expand(batch_size, -1, -1)
        return out


def write_edge_manifest_csv(path, organ_node_names, graph_state):
    fieldnames = [
        "src_index",
        "src_name",
        "dst_index",
        "dst_name",
        "edge_type",
        "is_prior_edge",
        "is_candidate_edge",
        "prior_edge_logit",
        "residual_logit",
        "adjacency_logit",
        "adjacency_prob",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for src_idx, src_name in enumerate(organ_node_names):
            for dst_idx, dst_name in enumerate(organ_node_names):
                edge_type_code = int(graph_state["edge_type_code"][src_idx, dst_idx])
                writer.writerow(
                    {
                        "src_index": src_idx,
                        "src_name": src_name,
                        "dst_index": dst_idx,
                        "dst_name": dst_name,
                        "edge_type": EDGE_TYPE_NAME[edge_type_code],
                        "is_prior_edge": int(graph_state["prior_edge_mask"][src_idx, dst_idx]),
                        "is_candidate_edge": int(graph_state["candidate_edge_mask"][src_idx, dst_idx]),
                        "prior_edge_logit": float(graph_state["prior_edge_logits"][src_idx, dst_idx]),
                        "residual_logit": float(graph_state["residual_logits"][src_idx, dst_idx]),
                        "adjacency_logit": float(graph_state["adjacency_logits"][src_idx, dst_idx]),
                        "adjacency_prob": float(graph_state["adjacency_prob"][src_idx, dst_idx]),
                    }
                )


def run_stage11_graph_construction(
    stage10_npz_path,
    output_root,
    self_logit=3.0,
    strong_logit=1.5,
    weak_logit=-0.5,
    nonprior_logit=-2.0,
    allow_nonprior_residual=True,
):
    stage10_npz_path = Path(stage10_npz_path)
    output_root = Path(output_root)
    ensure_output_dir(output_root)

    stage10_pack = load_stage10_npz(stage10_npz_path)
    validate_organ_order(stage10_pack["organ_node_names"])

    graph_constructor = WeakPriorGraphConstructor(
        organ_node_names=stage10_pack["organ_node_names"].tolist(),
        self_logit=self_logit,
        strong_logit=strong_logit,
        weak_logit=weak_logit,
        nonprior_logit=nonprior_logit,
        allow_nonprior_residual=allow_nonprior_residual,
    )
    graph_constructor.eval()

    with torch.no_grad():
        graph_state_torch = graph_constructor(batch_size=int(stage10_pack["Z"].shape[0]))

    graph_state = {
        key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        for key, value in graph_state_torch.items()
    }

    prior_edge_index = matrix_to_edge_index(graph_state["prior_edge_mask"])
    candidate_edge_index = matrix_to_edge_index(graph_state["candidate_edge_mask"])

    output_npz = output_root / "graph_construction_pack.npz"
    np.savez_compressed(
        output_npz,
        patient_ids=stage10_pack["patient_ids"],
        organ_node_names=stage10_pack["organ_node_names"],
        Z=stage10_pack["Z"].astype(np.float32),
        prior_edge_index=prior_edge_index,
        candidate_edge_index=candidate_edge_index,
        edge_type_code=graph_state["edge_type_code"].astype(np.uint8),
        prior_edge_mask=graph_state["prior_edge_mask"].astype(np.uint8),
        candidate_edge_mask=graph_state["candidate_edge_mask"].astype(np.uint8),
        prior_edge_logits=graph_state["prior_edge_logits"].astype(np.float32),
        residual_logits=graph_state["residual_logits"].astype(np.float32),
        adjacency_logits=graph_state["adjacency_logits"].astype(np.float32),
        adjacency_prob=graph_state["adjacency_prob"].astype(np.float32),
    )

    edge_manifest_csv = output_root / "graph_edges.csv"
    write_edge_manifest_csv(
        edge_manifest_csv,
        organ_node_names=[str(x) for x in stage10_pack["organ_node_names"].tolist()],
        graph_state=graph_state,
    )

    summary = {
        "stage": "11.1_graph_construction",
        "stage10_npz_path": str(stage10_npz_path),
        "patient_count": int(stage10_pack["Z"].shape[0]),
        "num_nodes": int(stage10_pack["Z"].shape[1]),
        "node_feature_dim": int(stage10_pack["Z"].shape[2]),
        "self_logit": float(self_logit),
        "strong_logit": float(strong_logit),
        "weak_logit": float(weak_logit),
        "nonprior_logit": float(nonprior_logit),
        "allow_nonprior_residual": bool(allow_nonprior_residual),
        "residual_init_zero": True,
        "prior_edge_count": int(graph_state["prior_edge_mask"].sum()),
        "candidate_edge_count": int(graph_state["candidate_edge_mask"].sum()),
        "edge_type_counts": {
            "self": int((graph_state["edge_type_code"] == EDGE_TYPE_SELF).sum()),
            "strong": int((graph_state["edge_type_code"] == EDGE_TYPE_STRONG).sum()),
            "weak": int((graph_state["edge_type_code"] == EDGE_TYPE_WEAK).sum()),
            "none": int((graph_state["edge_type_code"] == EDGE_TYPE_NONE).sum()),
        },
        "shapes": {
            "Z": list(stage10_pack["Z"].shape),
            "prior_edge_index": list(prior_edge_index.shape),
            "candidate_edge_index": list(candidate_edge_index.shape),
            "adjacency_logits": list(graph_state["adjacency_logits"].shape),
            "adjacency_prob": list(graph_state["adjacency_prob"].shape),
        },
    }
    summary_path = output_root / "graph_construction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "pack_path": str(output_npz),
        "edge_manifest_path": str(edge_manifest_csv),
        "summary_path": str(summary_path),
        "patient_count": int(stage10_pack["Z"].shape[0]),
        "prior_edge_count": int(graph_state["prior_edge_mask"].sum()),
        "candidate_edge_count": int(graph_state["candidate_edge_mask"].sum()),
        "Z_shape": tuple(stage10_pack["Z"].shape),
        "adjacency_shape": tuple(graph_state["adjacency_logits"].shape),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Stage 11.1 graph construction",
        allow_abbrev=False,
    )
    parser.add_argument("--stage10-npz", default=str(DEFAULT_STAGE10_NPZ))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--self-logit", type=float, default=3.0)
    parser.add_argument("--strong-logit", type=float, default=1.5)
    parser.add_argument("--weak-logit", type=float, default=-0.5)
    parser.add_argument("--nonprior-logit", type=float, default=-2.0)
    parser.add_argument(
        "--restrict-residual-to-prior",
        action="store_true",
        help="If set, residual logits are only learnable on prior edges and self-loops.",
    )
    return parser


def parse_args():
    args, _unknown = build_arg_parser().parse_known_args()
    return args


def main():
    args = parse_args()
    stage10_npz = resolve_required_path(args.stage10_npz, DEFAULT_STAGE10_NPZ, "stage10 npz")
    result = run_stage11_graph_construction(
        stage10_npz_path=stage10_npz,
        output_root=Path(args.output_root),
        self_logit=args.self_logit,
        strong_logit=args.strong_logit,
        weak_logit=args.weak_logit,
        nonprior_logit=args.nonprior_logit,
        allow_nonprior_residual=not bool(args.restrict_residual_to_prior),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
