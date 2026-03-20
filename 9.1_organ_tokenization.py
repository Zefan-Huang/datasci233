import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

## data cleaning for the next step, basclly aligned everything into the same page

DEFAULT_MANIFEST_CSV = Path("output/patient_manifest.csv")
DEFAULT_STAGE6_TOKEN_CSV = Path("output/experiments/organ_seg/search_base24/infer/organ_imaging_tokens.csv")
DEFAULT_G_RNA_CSV = Path("output/stage7/7.2_rna_encoder/tokens/g_rna.csv")
DEFAULT_T_IMM_CSV = Path("output/stage7/7.3_immune_token/tokens/t_imm.csv")
DEFAULT_G_EHR_CSV = Path("output/stage8/8.2_ehr_encoder/tokens/g_ehr.csv")
DEFAULT_T_TUMOR_CSV = Path("output/preprocessed/roi_tokens.csv")
DEFAULT_T_SEM_CSV = Path("output/preprocessed/semantic_tokens.csv")
DEFAULT_AIM_DIR = Path("data/AIM_files_updated-11-10-2020")
DEFAULT_OUTPUT_ROOT = Path("output/stage9/9.1_organ_tokenization")

ORGAN_NODE_NAMES = [
    "Primary",
    "Lung",
    "Bone",
    "Liver",
    "LymphNodeMediastinum",
    "Brain",
]
STAGE6_ORGAN_TO_NODE = {
    "lung": "Lung",
    "bone": "Bone",
    "liver": "Liver",
    "brain": "Brain",
}
TOKEN_DIM_64_MODALITIES = ("t_img_nodes", "t_tumor", "t_sem", "t_imm")



def ensure_output_dir(output_root):
    output_root.mkdir(parents=True, exist_ok=True)


def resolve_required_path(path_arg, default_path, label):
    path = Path(path_arg) if path_arg.strip() else Path(default_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_optional_path(path_arg, default_path):
    path = Path(path_arg) if path_arg.strip() else Path(default_path)
    if not path.exists():
        return None
    return path


def load_manifest_patient_ids(manifest_csv_path, max_patients):
    patient_ids = []
    with manifest_csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "patient_id" not in (reader.fieldnames or []):
            raise RuntimeError("manifest csv missing patient_id column")
        for row in reader:
            patient_id = str(row.get("patient_id", "")).strip()
            if patient_id:
                patient_ids.append(patient_id)
    if max_patients > 0:
        patient_ids = patient_ids[: min(max_patients, len(patient_ids))]
    return patient_ids


def parse_token_json(token_json, expected_dim, label):
    values = json.loads(token_json)
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise RuntimeError(f"{label} must be 1D, got shape={arr.shape}")
    if expected_dim is not None and int(arr.shape[0]) != int(expected_dim):
        raise RuntimeError(
            f"{label} dim mismatch: expected={expected_dim} got={arr.shape[0]}"
        )
    return arr


def load_stage6_node_image_tokens(stage6_csv_path):
    by_patient = {}
    token_dim = None
    with stage6_csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"patient_id", "organ_name", "token_json", "missing_img_organ", "status"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(
                "stage6 token csv missing required columns: "
                + ",".join(sorted(required - set(reader.fieldnames or [])))
            )
        for row in reader:
            patient_id = str(row["patient_id"]).strip()
            organ_name = str(row["organ_name"]).strip().lower()
            node_name = STAGE6_ORGAN_TO_NODE.get(organ_name)
            if not node_name:
                continue
            if patient_id not in by_patient:
                by_patient[patient_id] = {}
            missing = int(row.get("missing_img_organ", 1) or 1)
            token_json = row.get("token_json", "")
            status = str(row.get("status", "")).strip().lower()
            if missing == 1 or not token_json or status != "ok":
                by_patient[patient_id][node_name] = None
                continue
            token = parse_token_json(token_json, token_dim, f"stage6:{patient_id}:{node_name}")
            if token_dim is None:
                token_dim = int(token.shape[0])
            by_patient[patient_id][node_name] = token
    if token_dim is None:
        raise RuntimeError("failed to infer stage6 image token dim from non-empty token_json")
    return by_patient, token_dim


def load_dense_token_csv(path, prefix_label):
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames or fieldnames[0] != "patient_id":
            raise RuntimeError(f"{prefix_label} csv missing patient_id first column: {path}")
        feature_fields = fieldnames[1:]
        rows = {}
        for row in reader:
            patient_id = str(row["patient_id"]).strip()
            if not patient_id:
                continue
            vec = np.asarray([float(row[col]) for col in feature_fields], dtype=np.float32)
            rows[patient_id] = vec
    if not rows:
        raise RuntimeError(f"{prefix_label} csv has no rows: {path}")
    dim = len(feature_fields)
    return rows, dim


def load_optional_json_token_csv(path, fallback_dim, label):
    if path is None:
        return {}, fallback_dim
    rows = {}
    token_dim = None
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"patient_id", "token_json"}
        if not required.issubset(set(reader.fieldnames or [])):
            return {}, fallback_dim
        for row in reader:
            patient_id = str(row.get("patient_id", "")).strip()
            token_json = row.get("token_json", "")
            if not patient_id or not token_json:
                continue
            token = parse_token_json(token_json, token_dim, f"{label}:{patient_id}")
            if token_dim is None:
                token_dim = int(token.shape[0])
            rows[patient_id] = token
    if token_dim is None:
        token_dim = fallback_dim
    return rows, token_dim


def find_aim_xml_path(patient_id, aim_dir):
    direct = Path(aim_dir) / f"{patient_id}.xml"
    if direct.exists():
        return direct
    candidates = sorted(Path(aim_dir).glob(f"{patient_id}*.xml"))
    if candidates:
        return candidates[0]
    return None


def parse_aim_feature_texts(aim_xml_path):
    if aim_xml_path is None:
        return []
    try:
        root = ET.parse(aim_xml_path).getroot()
    except Exception:
        return []

    features = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag:
            features.append(f"tag:{tag}")
        for key, value in elem.attrib.items():
            val = (value or "").strip()
            if val:
                features.append(f"{key}:{val}")
        text = (elem.text or "").strip()
        if text:
            features.append(f"text:{text}")
    return features


def build_semantic_token(feature_texts, token_dim):
    if not feature_texts:
        return []
    vec = [0.0 for _ in range(int(token_dim))]
    for text in feature_texts:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % int(token_dim)
        sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
        vec[idx] += sign

        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        for num_text in nums[:2]:
            try:
                value = float(num_text)
            except Exception:
                continue
            idx_num = int(digest[10:18], 16) % int(token_dim)
            vec[idx_num] += max(min(value, 10.0), -10.0) * 0.01

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def backfill_semantic_tokens_from_aim(patient_ids, t_sem_by_patient, token_dim, aim_dir):
    if int(token_dim) <= 0:
        return dict(t_sem_by_patient), 0
    aim_dir = Path(aim_dir)
    if not aim_dir.exists():
        return dict(t_sem_by_patient), 0

    out = dict(t_sem_by_patient)
    filled = 0
    for patient_id in patient_ids:
        if patient_id in out:
            continue
        aim_xml_path = find_aim_xml_path(patient_id, aim_dir)
        feature_texts = parse_aim_feature_texts(aim_xml_path)
        token = build_semantic_token(feature_texts, token_dim)
        if not token:
            continue
        out[patient_id] = np.asarray(token, dtype=np.float32)
        filled += 1
    return out, filled


def build_pack(
    patient_ids,
    stage6_img_by_patient,
    img_token_dim,
    g_rna_by_patient,
    g_rna_dim,
    g_ehr_by_patient,
    g_ehr_dim,
    t_imm_by_patient,
    t_imm_dim,
    t_tumor_by_patient,
    t_tumor_dim,
    t_sem_by_patient,
    t_sem_dim,
):
    n = len(patient_ids)
    num_nodes = len(ORGAN_NODE_NAMES)
    node_to_index = {name: idx for idx, name in enumerate(ORGAN_NODE_NAMES)}

    t_img_nodes = np.zeros((n, num_nodes, img_token_dim), dtype=np.float32)
    t_img_missing = np.ones((n, num_nodes), dtype=np.uint8)
    g_rna = np.zeros((n, g_rna_dim), dtype=np.float32)
    g_rna_missing = np.ones((n,), dtype=np.uint8)
    g_ehr = np.zeros((n, g_ehr_dim), dtype=np.float32)
    g_ehr_missing = np.ones((n,), dtype=np.uint8)
    t_imm = np.zeros((n, t_imm_dim), dtype=np.float32)
    t_imm_missing = np.ones((n,), dtype=np.uint8)
    t_tumor = np.zeros((n, t_tumor_dim), dtype=np.float32)
    t_tumor_missing = np.ones((n,), dtype=np.uint8)
    t_sem = np.zeros((n, t_sem_dim), dtype=np.float32)
    t_sem_missing = np.ones((n,), dtype=np.uint8)

    summary_rows = []
    for i, patient_id in enumerate(patient_ids):
        stage6_nodes = stage6_img_by_patient.get(patient_id, {})
        for node_name in ORGAN_NODE_NAMES:
            node_idx = node_to_index[node_name]
            token = stage6_nodes.get(node_name)
            if token is None:
                continue
            t_img_nodes[i, node_idx] = token
            t_img_missing[i, node_idx] = 0

        g_rna_token = g_rna_by_patient.get(patient_id)
        if g_rna_token is not None:
            g_rna[i] = g_rna_token
            g_rna_missing[i] = 0

        g_ehr_token = g_ehr_by_patient.get(patient_id)
        if g_ehr_token is not None:
            g_ehr[i] = g_ehr_token
            g_ehr_missing[i] = 0

        t_imm_token = t_imm_by_patient.get(patient_id)
        if t_imm_token is not None:
            t_imm[i] = t_imm_token
            t_imm_missing[i] = 0

        t_tumor_token = t_tumor_by_patient.get(patient_id)
        if t_tumor_token is not None:
            t_tumor[i] = t_tumor_token
            t_tumor_missing[i] = 0

        t_sem_token = t_sem_by_patient.get(patient_id)
        if t_sem_token is not None:
            t_sem[i] = t_sem_token
            t_sem_missing[i] = 0

        row = {
            "patient_id": patient_id,
            "has_g_rna": 0 if g_rna_missing[i] else 1,
            "has_t_imm": 0 if t_imm_missing[i] else 1,
            "has_g_ehr": 0 if g_ehr_missing[i] else 1,
            "has_t_tumor": 0 if t_tumor_missing[i] else 1,
            "has_t_sem": 0 if t_sem_missing[i] else 1,
        }
        for node_idx, node_name in enumerate(ORGAN_NODE_NAMES):
            row[f"has_img_{node_name}"] = 0 if t_img_missing[i, node_idx] else 1
        summary_rows.append(row)

    pack = {
        "patient_ids": np.asarray(patient_ids, dtype=object),
        "organ_node_names": np.asarray(ORGAN_NODE_NAMES, dtype=object),
        "t_img_nodes": t_img_nodes,
        "t_img_missing": t_img_missing,
        "g_rna": g_rna,
        "g_rna_missing": g_rna_missing,
        "g_ehr": g_ehr,
        "g_ehr_missing": g_ehr_missing,
        "t_imm": t_imm,
        "t_imm_missing": t_imm_missing,
        "t_tumor": t_tumor,
        "t_tumor_missing": t_tumor_missing,
        "t_sem": t_sem,
        "t_sem_missing": t_sem_missing,
    }
    return pack, summary_rows


def write_manifest_csv(path, rows):
    fieldnames = [
        "patient_id",
        "has_g_rna",
        "has_t_imm",
        "has_g_ehr",
        "has_t_tumor",
        "has_t_sem",
        "has_img_Primary",
        "has_img_Lung",
        "has_img_Bone",
        "has_img_Liver",
        "has_img_LymphNodeMediastinum",
        "has_img_Brain",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path, summary):
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 9 offline organ tokenization and evidence pack assembly.",
        allow_abbrev=False,
    )
    parser.add_argument("--manifest-csv", type=str, default=str(DEFAULT_MANIFEST_CSV))
    parser.add_argument("--stage6-token-csv", type=str, default=str(DEFAULT_STAGE6_TOKEN_CSV))
    parser.add_argument("--g-rna-csv", type=str, default=str(DEFAULT_G_RNA_CSV))
    parser.add_argument("--t-imm-csv", type=str, default=str(DEFAULT_T_IMM_CSV))
    parser.add_argument("--g-ehr-csv", type=str, default=str(DEFAULT_G_EHR_CSV))
    parser.add_argument("--t-tumor-csv", type=str, default=str(DEFAULT_T_TUMOR_CSV))
    parser.add_argument("--t-sem-csv", type=str, default=str(DEFAULT_T_SEM_CSV))
    parser.add_argument("--aim-dir", type=str, default=str(DEFAULT_AIM_DIR))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="0 means all manifest patients; >0 means first N patients.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    if args.max_patients < 0:
        raise SystemExit("--max-patients must be >= 0 (0 means all manifest patients)")

    manifest_csv = resolve_required_path(args.manifest_csv, DEFAULT_MANIFEST_CSV, "manifest csv")
    stage6_csv = resolve_required_path(args.stage6_token_csv, DEFAULT_STAGE6_TOKEN_CSV, "stage6 token csv")
    g_rna_csv = resolve_required_path(args.g_rna_csv, DEFAULT_G_RNA_CSV, "g_rna csv")
    t_imm_csv = resolve_required_path(args.t_imm_csv, DEFAULT_T_IMM_CSV, "t_imm csv")
    g_ehr_csv = resolve_required_path(args.g_ehr_csv, DEFAULT_G_EHR_CSV, "g_ehr csv")
    t_tumor_csv = resolve_optional_path(args.t_tumor_csv, DEFAULT_T_TUMOR_CSV)
    t_sem_csv = resolve_optional_path(args.t_sem_csv, DEFAULT_T_SEM_CSV)
    aim_dir = Path(args.aim_dir)
    output_root = Path(args.output_root)
    ensure_output_dir(output_root)

    patient_ids = load_manifest_patient_ids(manifest_csv, args.max_patients)
    stage6_img_by_patient, img_token_dim = load_stage6_node_image_tokens(stage6_csv)
    g_rna_by_patient, g_rna_dim = load_dense_token_csv(g_rna_csv, "g_rna")
    t_imm_by_patient, t_imm_dim = load_dense_token_csv(t_imm_csv, "t_imm")
    g_ehr_by_patient, g_ehr_dim = load_dense_token_csv(g_ehr_csv, "g_ehr")
    t_tumor_by_patient, t_tumor_dim = load_optional_json_token_csv(
        t_tumor_csv,
        fallback_dim=img_token_dim,
        label="t_tumor",
    )
    t_sem_by_patient, t_sem_dim = load_optional_json_token_csv(
        t_sem_csv,
        fallback_dim=img_token_dim,
        label="t_sem",
    )
    t_sem_csv_count = len(t_sem_by_patient)
    t_sem_by_patient, t_sem_aim_filled = backfill_semantic_tokens_from_aim(
        patient_ids=patient_ids,
        t_sem_by_patient=t_sem_by_patient,
        token_dim=t_sem_dim,
        aim_dir=aim_dir,
    )

    print(f"[start] manifest_csv={manifest_csv}")
    print(f"[start] stage6_token_csv={stage6_csv}")
    print(f"[start] g_rna_csv={g_rna_csv}")
    print(f"[start] t_imm_csv={t_imm_csv}")
    print(f"[start] g_ehr_csv={g_ehr_csv}")
    print(f"[start] t_tumor_csv={t_tumor_csv if t_tumor_csv else 'missing->zero_fill'}")
    print(f"[start] t_sem_csv={t_sem_csv if t_sem_csv else 'missing->zero_fill'}")
    print(f"[start] aim_dir={aim_dir}")
    print(
        f"[start] selected_patients={len(patient_ids)} "
        f"full_patients={1 if args.max_patients == 0 else 0}"
    )
    print(
        f"[dims] t_img={img_token_dim} g_rna={g_rna_dim} t_imm={t_imm_dim} "
        f"g_ehr={g_ehr_dim} t_tumor={t_tumor_dim} t_sem={t_sem_dim}"
    )
    print(
        f"[backfill] t_sem_csv_rows={t_sem_csv_count} "
        f"t_sem_aim_filled={t_sem_aim_filled}"
    )

    pack, summary_rows = build_pack(
        patient_ids=patient_ids,
        stage6_img_by_patient=stage6_img_by_patient,
        img_token_dim=img_token_dim,
        g_rna_by_patient=g_rna_by_patient,
        g_rna_dim=g_rna_dim,
        g_ehr_by_patient=g_ehr_by_patient,
        g_ehr_dim=g_ehr_dim,
        t_imm_by_patient=t_imm_by_patient,
        t_imm_dim=t_imm_dim,
        t_tumor_by_patient=t_tumor_by_patient,
        t_tumor_dim=t_tumor_dim,
        t_sem_by_patient=t_sem_by_patient,
        t_sem_dim=t_sem_dim,
    )

    out_npz = output_root / "organ_tokenization_pack.npz"
    out_manifest = output_root / "organ_tokenization_manifest.csv"
    out_summary = output_root / "organ_tokenization_summary.json"

    np.savez_compressed(out_npz, **pack)
    write_manifest_csv(out_manifest, summary_rows)

    summary = {
        "patient_count": int(len(patient_ids)),
        "organ_node_count": int(len(ORGAN_NODE_NAMES)),
        "organ_node_names": list(ORGAN_NODE_NAMES),
        "dims": {
            "t_img_nodes": int(img_token_dim),
            "g_rna": int(g_rna_dim),
            "g_ehr": int(g_ehr_dim),
            "t_imm": int(t_imm_dim),
            "t_tumor": int(t_tumor_dim),
            "t_sem": int(t_sem_dim),
        },
        "token_sources": {
            "t_sem_csv_rows": int(t_sem_csv_count),
            "t_sem_aim_filled": int(t_sem_aim_filled),
        },
        "available_counts": {
            "g_rna": int(sum(1 for r in summary_rows if r["has_g_rna"] == 1)),
            "t_imm": int(sum(1 for r in summary_rows if r["has_t_imm"] == 1)),
            "g_ehr": int(sum(1 for r in summary_rows if r["has_g_ehr"] == 1)),
            "t_tumor": int(sum(1 for r in summary_rows if r["has_t_tumor"] == 1)),
            "t_sem": int(sum(1 for r in summary_rows if r["has_t_sem"] == 1)),
            "img_primary": int(sum(1 for r in summary_rows if r["has_img_Primary"] == 1)),
            "img_lung": int(sum(1 for r in summary_rows if r["has_img_Lung"] == 1)),
            "img_bone": int(sum(1 for r in summary_rows if r["has_img_Bone"] == 1)),
            "img_liver": int(sum(1 for r in summary_rows if r["has_img_Liver"] == 1)),
            "img_lymphnode_mediastinum": int(
                sum(1 for r in summary_rows if r["has_img_LymphNodeMediastinum"] == 1)
            ),
            "img_brain": int(sum(1 for r in summary_rows if r["has_img_Brain"] == 1)),
        },
    }
    write_summary_json(out_summary, summary)

    print(f"wrote: {out_npz}")
    print(f"wrote: {out_manifest}")
    print(f"wrote: {out_summary}")
    print(f"pack_t_img_nodes_shape: {tuple(pack['t_img_nodes'].shape)}")
    print(f"pack_g_rna_shape: {tuple(pack['g_rna'].shape)}")
    print(f"pack_g_ehr_shape: {tuple(pack['g_ehr'].shape)}")
    print(f"pack_t_imm_shape: {tuple(pack['t_imm'].shape)}")
    print("complete")


if __name__ == "__main__":
    main()
