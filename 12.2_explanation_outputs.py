import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_GRAPH_REASONING_PACK = Path("output/stage11/11.2_graph_reasoning/graph_reasoning_pack.npz")
DEFAULT_PRIMARY_PACK = Path("output/stage12/12.1_primary_outputs_refit_all_seed2024/pred/primary_output_pack.npz")
FALLBACK_PRIMARY_PACK = Path("output/stage12/12.1_primary_outputs/pred/primary_output_pack.npz")
DEFAULT_OUTPUT_ROOT = Path("output/stage12/12.2_explanation_outputs")

EDGE_TYPE_NAME = {
    0: "none",
    1: "self",
    2: "strong",
    3: "weak",
}
EXPLANATION_SEMANTICS = (
    "Latent diffusion explanation only; organ/path outputs are model-induced explanation layers "
    "for OS/recurrence predictions and do not imply organ-level ground-truth supervision."
)



def resolve_required_path(path_arg, default_path, label):
    path = Path(path_arg) if str(path_arg).strip() else Path(default_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_primary_pack(path_arg):
    if str(path_arg).strip():
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"primary output pack not found: {path}")
        return path
    if DEFAULT_PRIMARY_PACK.exists():
        return DEFAULT_PRIMARY_PACK
    if FALLBACK_PRIMARY_PACK.exists():
        return FALLBACK_PRIMARY_PACK
    raise FileNotFoundError(
        f"primary output pack not found: {DEFAULT_PRIMARY_PACK} | {FALLBACK_PRIMARY_PACK}"
    )


def ensure_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_npz(path):
    with np.load(path, allow_pickle=True) as z:
        out = {}
        for key in z.files:
            value = z[key]
            if value.dtype.kind in {"U", "O"}:
                out[key] = value.astype(str)
            else:
                out[key] = value
    return out


def load_optional_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_graph_pack(pack):
    required = {
        "patient_ids",
        "organ_node_names",
        "organ_susceptibility",
        "edge_diffusion_prob",
        "edge_type_code",
        "prior_edge_mask",
    }
    missing = required - set(pack.keys())
    if missing:
        raise RuntimeError(f"graph reasoning pack missing required arrays: {sorted(missing)}")


def validate_primary_pack(pack):
    required = {
        "patient_ids",
        "recurrence_probability",
        "recurrence_location_probability",
        "recurrence_classes",
    }
    missing = required - set(pack.keys())
    if missing:
        raise RuntimeError(f"primary output pack missing required arrays: {sorted(missing)}")


def validate_patient_alignment(graph_pack, primary_pack):
    graph_ids = graph_pack["patient_ids"].tolist()
    primary_ids = primary_pack["patient_ids"].tolist()
    if graph_ids != primary_ids:
        raise RuntimeError("patient_ids mismatch between graph reasoning pack and primary output pack")


def infer_survival_mode(primary_pack):
    hazard_prob = np.asarray(primary_pack.get("hazard_prob", np.zeros((0, 0), dtype=np.float32)))
    if hazard_prob.ndim == 2 and hazard_prob.shape[1] > 0:
        return "discrete"
    return "cox"


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


def rank_organ_susceptibility(organ_node_names, organ_scores):
    organ_scores = np.asarray(organ_scores, dtype=np.float32)
    ranked_idx = np.argsort(-organ_scores)
    return [
        {
            "organ_index": int(idx),
            "organ_name": str(organ_node_names[idx]),
            "susceptibility": float(organ_scores[idx]),
            "rank": int(rank_pos + 1),
        }
        for rank_pos, idx in enumerate(ranked_idx.tolist())
    ]


def rank_edge_diffusion(organ_node_names, edge_prob, edge_type_code, prior_edge_mask, top_edge_k):
    rows = []
    num_nodes = len(organ_node_names)
    for src_idx in range(num_nodes):
        for dst_idx in range(num_nodes):
            if src_idx == dst_idx:
                continue
            rows.append(
                {
                    "src_index": int(src_idx),
                    "src_name": str(organ_node_names[src_idx]),
                    "dst_index": int(dst_idx),
                    "dst_name": str(organ_node_names[dst_idx]),
                    "edge_type": EDGE_TYPE_NAME[int(edge_type_code[src_idx, dst_idx])],
                    "is_prior_edge": int(prior_edge_mask[src_idx, dst_idx]),
                    "edge_diffusion_prob": float(edge_prob[src_idx, dst_idx]),
                }
            )
    rows.sort(key=lambda item: item["edge_diffusion_prob"], reverse=True)
    for rank_pos, row in enumerate(rows):
        row["rank"] = int(rank_pos + 1)
    return rows[: int(top_edge_k)]


def summarize_primary_prediction(primary_pack, patient_idx):
    recurrence_classes = primary_pack["recurrence_classes"].tolist()
    recurrence_probability = float(primary_pack["recurrence_probability"][patient_idx])
    recurrence_location_probability = np.asarray(
        primary_pack["recurrence_location_probability"][patient_idx],
        dtype=np.float32,
    )
    top_loc_idx = int(np.argmax(recurrence_location_probability))
    survival_mode = infer_survival_mode(primary_pack)

    out = {
        "survival_mode": str(survival_mode),
        "recurrence_probability": recurrence_probability,
        "predicted_recurrence_location": str(recurrence_classes[top_loc_idx]),
        "recurrence_location_probability": {
            str(name): float(recurrence_location_probability[idx])
            for idx, name in enumerate(recurrence_classes)
        },
    }
    if str(survival_mode) == "cox":
        out["os_log_risk"] = float(primary_pack["os_log_risk"][patient_idx])
    else:
        survival_curve = np.asarray(primary_pack["survival_curve"][patient_idx], dtype=np.float32)
        hazard_prob = np.asarray(primary_pack["hazard_prob"][patient_idx], dtype=np.float32)
        out["survival_curve"] = survival_curve.tolist()
        out["hazard_probability"] = hazard_prob.tolist()
        out["survival_probability_last_bin"] = (
            float(survival_curve[-1]) if survival_curve.size > 0 else None
        )
    return out


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(graph_pack, primary_pack, top_k, max_hops, beam_width, top_edge_k):
    patient_ids = graph_pack["patient_ids"].tolist()
    organ_node_names = graph_pack["organ_node_names"].tolist()
    organ_susceptibility = np.asarray(graph_pack["organ_susceptibility"], dtype=np.float32)
    edge_diffusion_prob = np.asarray(graph_pack["edge_diffusion_prob"], dtype=np.float32)
    edge_type_code = np.asarray(graph_pack["edge_type_code"], dtype=np.uint8)
    prior_edge_mask = np.asarray(graph_pack["prior_edge_mask"], dtype=np.uint8)

    organ_rows = []
    edge_rows = []
    patient_manifest_rows = []
    patient_explanations = []
    topk_paths_payload = []
    top1_path_counter = Counter()

    for patient_idx, patient_id in enumerate(patient_ids):
        organ_ranked = rank_organ_susceptibility(
            organ_node_names=organ_node_names,
            organ_scores=organ_susceptibility[patient_idx],
        )
        top_edges = rank_edge_diffusion(
            organ_node_names=organ_node_names,
            edge_prob=edge_diffusion_prob[patient_idx],
            edge_type_code=edge_type_code,
            prior_edge_mask=prior_edge_mask,
            top_edge_k=top_edge_k,
        )
        top_paths = derive_topk_paths_for_patient(
            p_edge=edge_diffusion_prob[patient_idx],
            organ_node_names=organ_node_names,
            top_k=top_k,
            max_hops=max_hops,
            beam_width=beam_width,
            source_name="Primary",
        )
        primary_summary = summarize_primary_prediction(primary_pack, patient_idx=patient_idx)

        for organ_idx, organ_name in enumerate(organ_node_names):
            organ_rows.append(
                {
                    "patient_id": str(patient_id),
                    "organ_index": int(organ_idx),
                    "organ_name": str(organ_name),
                    "susceptibility": float(organ_susceptibility[patient_idx, organ_idx]),
                }
            )

        for src_idx, src_name in enumerate(organ_node_names):
            for dst_idx, dst_name in enumerate(organ_node_names):
                edge_rows.append(
                    {
                        "patient_id": str(patient_id),
                        "src_index": int(src_idx),
                        "src_name": str(src_name),
                        "dst_index": int(dst_idx),
                        "dst_name": str(dst_name),
                        "edge_type": EDGE_TYPE_NAME[int(edge_type_code[src_idx, dst_idx])],
                        "is_prior_edge": int(prior_edge_mask[src_idx, dst_idx]),
                        "edge_diffusion_prob": float(edge_diffusion_prob[patient_idx, src_idx, dst_idx]),
                    }
                )

        if top_paths:
            top1_path_counter[tuple(top_paths[0]["path_names"])] += 1

        patient_manifest_rows.append(
            {
                "patient_id": str(patient_id),
                "recurrence_probability": float(primary_summary["recurrence_probability"]),
                "predicted_recurrence_location": str(primary_summary["predicted_recurrence_location"]),
                "top_susceptibility_organ": str(organ_ranked[0]["organ_name"]),
                "top_susceptibility_score": float(organ_ranked[0]["susceptibility"]),
                "top_edge_src": str(top_edges[0]["src_name"]) if top_edges else "",
                "top_edge_dst": str(top_edges[0]["dst_name"]) if top_edges else "",
                "top_edge_prob": float(top_edges[0]["edge_diffusion_prob"]) if top_edges else "",
                "top_path_json": json.dumps(top_paths[0], ensure_ascii=False) if top_paths else "",
            }
        )

        patient_explanations.append(
            {
                "patient_id": str(patient_id),
                "explanation_semantics": EXPLANATION_SEMANTICS,
                "primary_outputs": primary_summary,
                "organ_susceptibility_ranked": organ_ranked,
                "top_edges": top_edges,
                "top_paths": top_paths,
            }
        )
        topk_paths_payload.append({"patient_id": str(patient_id), "paths": top_paths})

    mean_susceptibility = organ_susceptibility.mean(axis=0)
    edge_prob_mean = edge_diffusion_prob.mean(axis=0)
    mean_organ_ranking = rank_organ_susceptibility(organ_node_names, mean_susceptibility)
    mean_edge_ranking = rank_edge_diffusion(
        organ_node_names=organ_node_names,
        edge_prob=edge_prob_mean,
        edge_type_code=edge_type_code,
        prior_edge_mask=prior_edge_mask,
        top_edge_k=max(10, int(top_edge_k)),
    )

    path_frequency = [
        {
            "path_names": list(path_names),
            "count": int(count),
            "fraction": float(count / max(len(patient_ids), 1)),
        }
        for path_names, count in top1_path_counter.most_common()
    ]

    return {
        "organ_rows": organ_rows,
        "edge_rows": edge_rows,
        "patient_manifest_rows": patient_manifest_rows,
        "patient_explanations": patient_explanations,
        "topk_paths_payload": topk_paths_payload,
        "summary": {
            "patient_count": int(len(patient_ids)),
            "num_nodes": int(len(organ_node_names)),
            "top_k": int(top_k),
            "max_hops": int(max_hops),
            "beam_width": int(beam_width),
            "top_edge_k": int(top_edge_k),
            "explanation_semantics": EXPLANATION_SEMANTICS,
            "ranges": {
                "organ_susceptibility_min": float(organ_susceptibility.min()),
                "organ_susceptibility_max": float(organ_susceptibility.max()),
                "edge_diffusion_prob_min": float(edge_diffusion_prob.min()),
                "edge_diffusion_prob_max": float(edge_diffusion_prob.max()),
            },
            "mean_organ_susceptibility_ranking": mean_organ_ranking,
            "mean_edge_diffusion_ranking": mean_edge_ranking,
            "top1_path_frequency": path_frequency,
        },
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Stage 12.2 explanation outputs (latent diffusion)",
        allow_abbrev=False,
    )
    parser.add_argument("--graph-pack", type=str, default=str(DEFAULT_GRAPH_REASONING_PACK))
    parser.add_argument("--primary-pack", type=str, default="")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--top-edge-k", type=int, default=5)
    return parser


def parse_args():
    args, _unknown = build_arg_parser().parse_known_args()
    return args


def main():
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.max_hops <= 0:
        raise SystemExit("--max-hops must be > 0")
    if args.beam_width <= 0:
        raise SystemExit("--beam-width must be > 0")
    if args.top_edge_k <= 0:
        raise SystemExit("--top-edge-k must be > 0")

    graph_pack_path = resolve_required_path(args.graph_pack, DEFAULT_GRAPH_REASONING_PACK, "graph pack")
    primary_pack_path = resolve_primary_pack(args.primary_pack)
    output_root = ensure_output_dir(args.output_root)

    print(f"[start] graph_pack={graph_pack_path}")
    print(f"[start] primary_pack={primary_pack_path}")
    print(f"[start] output_root={output_root}")

    graph_pack = load_npz(graph_pack_path)
    primary_pack = load_npz(primary_pack_path)
    validate_graph_pack(graph_pack)
    validate_primary_pack(primary_pack)
    validate_patient_alignment(graph_pack, primary_pack)
    graph_summary = load_optional_json(graph_pack_path.with_name("graph_reasoning_summary.json"))

    built = build_outputs(
        graph_pack=graph_pack,
        primary_pack=primary_pack,
        top_k=args.top_k,
        max_hops=args.max_hops,
        beam_width=args.beam_width,
        top_edge_k=args.top_edge_k,
    )

    out_npz = output_root / "explanation_pack.npz"
    np.savez_compressed(
        out_npz,
        patient_ids=np.asarray(graph_pack["patient_ids"]).astype(str),
        organ_node_names=np.asarray(graph_pack["organ_node_names"]).astype(str),
        organ_susceptibility=np.asarray(graph_pack["organ_susceptibility"], dtype=np.float32),
        edge_diffusion_prob=np.asarray(graph_pack["edge_diffusion_prob"], dtype=np.float32),
        edge_type_code=np.asarray(graph_pack["edge_type_code"], dtype=np.uint8),
        prior_edge_mask=np.asarray(graph_pack["prior_edge_mask"], dtype=np.uint8),
        recurrence_probability=np.asarray(primary_pack["recurrence_probability"], dtype=np.float32),
        recurrence_location_probability=np.asarray(primary_pack["recurrence_location_probability"], dtype=np.float32),
        recurrence_classes=np.asarray(primary_pack["recurrence_classes"]).astype(str),
        os_log_risk=np.asarray(primary_pack.get("os_log_risk", np.zeros((len(graph_pack["patient_ids"]),), dtype=np.float32)), dtype=np.float32),
        hazard_prob=np.asarray(primary_pack.get("hazard_prob", np.zeros((len(graph_pack["patient_ids"]), 0), dtype=np.float32)), dtype=np.float32),
        survival_curve=np.asarray(primary_pack.get("survival_curve", np.zeros((len(graph_pack["patient_ids"]), 0), dtype=np.float32)), dtype=np.float32),
    )

    organ_csv = output_root / "organ_susceptibility.csv"
    edge_csv = output_root / "edge_diffusion_long.csv"
    manifest_csv = output_root / "patient_explanation_manifest.csv"
    explanations_json = output_root / "patient_explanations.json"
    topk_json = output_root / "topk_paths.json"
    summary_json = output_root / "explanation_summary.json"

    write_csv(
        organ_csv,
        ["patient_id", "organ_index", "organ_name", "susceptibility"],
        built["organ_rows"],
    )
    write_csv(
        edge_csv,
        [
            "patient_id",
            "src_index",
            "src_name",
            "dst_index",
            "dst_name",
            "edge_type",
            "is_prior_edge",
            "edge_diffusion_prob",
        ],
        built["edge_rows"],
    )
    write_csv(
        manifest_csv,
        [
            "patient_id",
            "recurrence_probability",
            "predicted_recurrence_location",
            "top_susceptibility_organ",
            "top_susceptibility_score",
            "top_edge_src",
            "top_edge_dst",
            "top_edge_prob",
            "top_path_json",
        ],
        built["patient_manifest_rows"],
    )
    explanations_json.write_text(
        json.dumps(built["patient_explanations"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    topk_json.write_text(
        json.dumps(built["topk_paths_payload"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_payload = {
        "stage": "12.2_explanation_outputs",
        "graph_pack_path": str(graph_pack_path),
        "graph_summary_path": str(graph_pack_path.with_name("graph_reasoning_summary.json")),
        "primary_pack_path": str(primary_pack_path),
        "survival_mode": infer_survival_mode(primary_pack),
        "graph_reasoning_random_init_only": (
            None if graph_summary is None else graph_summary.get("random_init_only")
        ),
        "graph_reasoning_explanation_semantics": (
            None if graph_summary is None else graph_summary.get("explanation_semantics")
        ),
    }
    summary_payload.update(built["summary"])
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote: {out_npz}")
    print(f"wrote: {organ_csv}")
    print(f"wrote: {edge_csv}")
    print(f"wrote: {manifest_csv}")
    print(f"wrote: {explanations_json}")
    print(f"wrote: {topk_json}")
    print(f"wrote: {summary_json}")
    print(f"patient_count: {built['summary']['patient_count']}")
    print("complete")


if __name__ == "__main__":
    main()
