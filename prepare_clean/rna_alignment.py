import argparse
import csv
import json
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

PRIMARY_SERIES_MATRIX = Path("output/clean_data/GSE103584_series_matrix.txt")
FALLBACK_SERIES_MATRIX = Path("data/GSE103584_series_matrix.txt")

PRIMARY_EXPR_TSV = Path("output/clean_data/GSE103584_norm_counts_TPM_GRCh38.p13_NCBI.tsv")
FALLBACK_EXPR_TSV = Path("data/GSE103584_norm_counts_TPM_GRCh38.p13_NCBI.tsv")

PRIMARY_MANIFEST = Path("output/patient_manifest.csv")
FALLBACK_MANIFEST = Path("output/clean_data/patient_manifest.csv")

DEFAULT_OUTPUT_ROOT = Path("output/stage7/7.1_rna_alignment")

def check_dependencies():

    missing = []
    if np is None:
        missing.append("numpy")
    return missing

def resolve_input_path(primary_path, fallback_path):

    if primary_path.exists():
        return primary_path
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(f"input file not found: {primary_path} | {fallback_path}")

def ensure_output_dir(output_root):

    output_root.mkdir(parents=True, exist_ok=True)

def parse_series_matrix_mapping(series_matrix_path):

    sample_titles = None
    sample_geo = None
    quote = chr(34)

    with series_matrix_path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("!Sample_title\t"):
                sample_titles = [x.strip().strip(quote) for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession\t"):
                sample_geo = [x.strip().strip(quote) for x in line.rstrip("\n").split("\t")[1:]]

    if sample_titles is None or sample_geo is None:
        raise RuntimeError("failed to parse !Sample_title / !Sample_geo_accession from series_matrix")
    if len(sample_titles) != len(sample_geo):
        raise RuntimeError("series_matrix has mismatched sample title vs geo lengths")

    gsm_to_patient = {}
    duplicate_gsm = 0
    for patient_id, gsm_id in zip(sample_titles, sample_geo):
        if gsm_id in gsm_to_patient:
            duplicate_gsm += 1
        gsm_to_patient[gsm_id] = patient_id

    return {
        "sample_titles": sample_titles,
        "sample_geo": sample_geo,
        "gsm_to_patient": gsm_to_patient,
        "duplicate_gsm": duplicate_gsm,
    }

def load_manifest_patient_ids(manifest_csv_path):

    if manifest_csv_path is None:
        return set()
    if not manifest_csv_path.exists():
        return set()

    with manifest_csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "patient_id" not in (reader.fieldnames or []):
            return set()
        out = set()
        for row in reader:
            patient_id = (row.get("patient_id") or "").strip()
            if patient_id:
                out.add(patient_id)
    return out

def parse_expression_header(expr_tsv_path):

    with expr_tsv_path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    if len(header) < 2:
        raise RuntimeError("expression header is invalid")
    return header[1:]

def build_selected_columns(header_gsms, gsm_to_patient, manifest_patient_ids, use_manifest_filter, max_patients):

    selected_columns = []
    patient_order = []
    patient_seen = set()

    for idx, gsm_id in enumerate(header_gsms):
        patient_id = gsm_to_patient.get(gsm_id, "")
        if not patient_id:
            continue
        if use_manifest_filter and manifest_patient_ids and patient_id not in manifest_patient_ids:
            continue

        selected_columns.append({
            "sample_col_index": idx,
            "gsm_id": gsm_id,
            "patient_id": patient_id,
        })

        if patient_id not in patient_seen:
            patient_seen.add(patient_id)
            patient_order.append(patient_id)

    if max_patients > 0:
        keep_patients = set(patient_order[:max_patients])
        patient_order = patient_order[:max_patients]
        selected_columns = [x for x in selected_columns if x["patient_id"] in keep_patients]

    if not selected_columns:
        raise RuntimeError("no RNA samples selected after mapping/filtering")

    patient_to_col_positions = {}
    for pos, col in enumerate(selected_columns):
        patient_id = col["patient_id"]
        if patient_id not in patient_to_col_positions:
            patient_to_col_positions[patient_id] = []
        patient_to_col_positions[patient_id].append(pos)

    duplicate_patient_count = sum(1 for v in patient_to_col_positions.values() if len(v) > 1)

    return {
        "selected_columns": selected_columns,
        "patient_order": patient_order,
        "patient_to_col_positions": patient_to_col_positions,
        "duplicate_patient_count": duplicate_patient_count,
    }

def load_expression_matrix(expr_tsv_path, selected_columns):

    selected_sample_indices = [x["sample_col_index"] for x in selected_columns]
    selected_file_indices = [i + 1 for i in selected_sample_indices]

    gene_ids = []
    rows = []
    bad_value_count = 0

    with expr_tsv_path.open(encoding="utf-8") as f:
        _header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= 1:
                continue
            gene_id = parts[0].strip()
            if not gene_id:
                continue

            values = []
            for col_idx in selected_file_indices:
                if col_idx >= len(parts):
                    values.append(0.0)
                    bad_value_count += 1
                    continue
                raw = parts[col_idx].strip()
                if raw == "":
                    values.append(0.0)
                    bad_value_count += 1
                    continue
                try:
                    v = float(raw)
                except Exception:
                    v = 0.0
                    bad_value_count += 1
                if not np.isfinite(v):
                    v = 0.0
                    bad_value_count += 1
                values.append(v)

            gene_ids.append(gene_id)
            rows.append(values)

    if not rows:
        raise RuntimeError("expression matrix rows are empty")

    matrix_gene_by_col = np.asarray(rows, dtype=np.float32)
    return gene_ids, matrix_gene_by_col, bad_value_count

def aggregate_to_patient_matrix(matrix_gene_by_col, patient_order, patient_to_col_positions):

    gene_count = int(matrix_gene_by_col.shape[0])
    patient_count = len(patient_order)
    out = np.zeros((patient_count, gene_count), dtype=np.float32)

    for i, patient_id in enumerate(patient_order):
        col_positions = patient_to_col_positions.get(patient_id, [])
        if not col_positions:
            continue
        if len(col_positions) == 1:
            out[i] = matrix_gene_by_col[:, col_positions[0]]
        else:
            out[i] = matrix_gene_by_col[:, col_positions].mean(axis=1)

    return out

def build_x_rna_log1p_zscore(patient_by_gene):

    clipped = np.maximum(patient_by_gene, 0.0)
    log1p_matrix = np.log1p(clipped)

    gene_mean = log1p_matrix.mean(axis=0)
    gene_std = log1p_matrix.std(axis=0)
    safe_std = np.where(gene_std > 1e-8, gene_std, 1.0)

    x_rna = (log1p_matrix - gene_mean) / safe_std
    x_rna = x_rna.astype(np.float32)

    const_mask = gene_std <= 1e-8
    if const_mask.any():
        x_rna[:, const_mask] = 0.0

    return x_rna, gene_mean.astype(np.float32), gene_std.astype(np.float32)

def write_sample_manifest_csv(path, selected_columns):

    fieldnames = ["sample_col_index", "gsm_id", "patient_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_columns)

def write_summary_json(path, summary):

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def parse_args():

    parser = argparse.ArgumentParser(
        description="Stage 7.1 RNA alignment: GEO -> PatientID and x_rna generation.",
        allow_abbrev=False,
    )
    parser.add_argument("--series-matrix", type=str, default="")
    parser.add_argument("--expr-tsv", type=str, default="")
    parser.add_argument("--manifest-csv", type=str, default="")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="0 means all aligned patients; >0 means first N patients.",
    )
    parser.add_argument(
        "--no-manifest-filter",
        action="store_true",
        help="If set, do not filter RNA patients by manifest patient_id.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"ignore_unknown_args: {unknown}")
    return args

def main():

    args = parse_args()
    missing = check_dependencies()
    if missing:
        raise SystemExit(
            "missing dependency: "
            + ",".join(missing)
            + ". install example: .venv/bin/pip install numpy"
        )

    if args.max_patients < 0:
        raise SystemExit("--max-patients must be >= 0 (0 means all patients)")

    series_matrix_path = Path(args.series_matrix) if args.series_matrix.strip() else resolve_input_path(
        PRIMARY_SERIES_MATRIX,
        FALLBACK_SERIES_MATRIX,
    )
    expr_tsv_path = Path(args.expr_tsv) if args.expr_tsv.strip() else resolve_input_path(
        PRIMARY_EXPR_TSV,
        FALLBACK_EXPR_TSV,
    )

    if args.manifest_csv.strip():
        manifest_csv_path = Path(args.manifest_csv)
    else:
        manifest_csv_path = PRIMARY_MANIFEST if PRIMARY_MANIFEST.exists() else FALLBACK_MANIFEST

    output_root = Path(args.output_root)
    ensure_output_dir(output_root)

    use_manifest_filter = not bool(args.no_manifest_filter)

    print(f"[start] series_matrix={series_matrix_path}")
    print(f"[start] expr_tsv={expr_tsv_path}")
    print(f"[start] manifest_csv={manifest_csv_path}")
    print(
        f"[start] max_patients={args.max_patients} "
        f"full_patients={1 if args.max_patients == 0 else 0} "
        f"use_manifest_filter={1 if use_manifest_filter else 0}"
    )

    mapping = parse_series_matrix_mapping(series_matrix_path)
    gsm_to_patient = mapping["gsm_to_patient"]

    manifest_patient_ids = load_manifest_patient_ids(manifest_csv_path)
    header_gsms = parse_expression_header(expr_tsv_path)

    selected = build_selected_columns(
        header_gsms=header_gsms,
        gsm_to_patient=gsm_to_patient,
        manifest_patient_ids=manifest_patient_ids,
        use_manifest_filter=use_manifest_filter,
        max_patients=args.max_patients,
    )

    selected_columns = selected["selected_columns"]
    patient_order = selected["patient_order"]
    patient_to_col_positions = selected["patient_to_col_positions"]

    print(
        f"[select] selected_columns={len(selected_columns)} "
        f"selected_patients={len(patient_order)} "
        f"duplicate_patients={selected['duplicate_patient_count']}"
    )

    gene_ids, matrix_gene_by_col, bad_value_count = load_expression_matrix(
        expr_tsv_path,
        selected_columns,
    )
    patient_by_gene = aggregate_to_patient_matrix(
        matrix_gene_by_col=matrix_gene_by_col,
        patient_order=patient_order,
        patient_to_col_positions=patient_to_col_positions,
    )
    x_rna, gene_mean, gene_std = build_x_rna_log1p_zscore(patient_by_gene)

    out_npz = output_root / "x_rna_log1p_zscore.npz"
    out_manifest_csv = output_root / "rna_sample_manifest.csv"
    out_summary_json = output_root / "rna_alignment_summary.json"

    np.savez_compressed(
        out_npz,
        x_rna=x_rna,
        patient_ids=np.asarray(patient_order),
        gene_ids=np.asarray(gene_ids),
        gene_mean_log1p=np.asarray(gene_mean),
        gene_std_log1p=np.asarray(gene_std),
    )
    write_sample_manifest_csv(out_manifest_csv, selected_columns)

    constant_gene_count = int(np.sum(gene_std <= 1e-8))
    summary = {
        "series_matrix_path": str(series_matrix_path),
        "expr_tsv_path": str(expr_tsv_path),
        "manifest_csv_path": str(manifest_csv_path),
        "use_manifest_filter": int(use_manifest_filter),
        "max_patients": int(args.max_patients),
        "selected_patient_count": int(len(patient_order)),
        "selected_column_count": int(len(selected_columns)),
        "gene_count": int(len(gene_ids)),
        "x_rna_shape": [int(x_rna.shape[0]), int(x_rna.shape[1])],
        "duplicate_patient_count": int(selected["duplicate_patient_count"]),
        "duplicate_gsm_in_series": int(mapping["duplicate_gsm"]),
        "bad_value_count": int(bad_value_count),
        "constant_gene_count": int(constant_gene_count),
    }
    write_summary_json(out_summary_json, summary)

    print(f"wrote: {out_npz}")
    print(f"wrote: {out_manifest_csv}")
    print(f"wrote: {out_summary_json}")
    print(f"selected_patients: {len(patient_order)}")
    print(f"genes: {len(gene_ids)}")
    print("complete")

if __name__ == "__main__":
    main()
