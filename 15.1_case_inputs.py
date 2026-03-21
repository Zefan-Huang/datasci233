import argparse
import csv
import importlib.util
import json
from pathlib import Path

##

try:
    import numpy as np
except Exception:
    np = None


def resolve_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


ROOT = resolve_root()
DEFAULT_OUTPUT_ROOT = ROOT / "output/stage15/15.1_case_inputs"
DEFAULT_MANIFEST_CSV = ROOT / "output/patient_manifest.csv"
DEFAULT_LABELS_CSV = ROOT / "output/labels_time_zero.csv"
DEFAULT_STAGE7_X_RNA_NPZ = ROOT / "output/stage7/7.1_rna_alignment/x_rna_log1p_zscore.npz"
DEFAULT_STAGE7_G_RNA_CSV = ROOT / "output/stage7/7.2_rna_encoder/tokens/g_rna.csv"
DEFAULT_STAGE8_X_EHR_CSV = ROOT / "output/stage8/8.1_clinical_feature_engineering/x_ehr_features.csv"
DEFAULT_STAGE8_G_EHR_CSV = ROOT / "output/stage8/8.2_ehr_encoder/tokens/g_ehr.csv"


def load_local_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOTAL_TABLE = load_local_module(ROOT / "prepare_clean/total_table.py", "stage15_total_table")
IMAGING_PREP = load_local_module(ROOT / "prepare_clean/imaging_preprocessing.py", "stage15_imaging_preprocessing")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 15.1 case-input packaging for inference",
        allow_abbrev=False,
    )
    parser.add_argument("--patient-id", type=str, default="")
    parser.add_argument("--ct-path", type=str, default="")
    parser.add_argument("--pet-path", type=str, default="")
    parser.add_argument("--tumor-seg-path", type=str, default="")
    parser.add_argument("--aim-path", type=str, default="")
    parser.add_argument("--clinical-csv", type=str, default="")
    parser.add_argument("--clinical-json", type=str, default="")
    parser.add_argument("--clinical-row-id", type=str, default="")
    parser.add_argument("--clinical-id-column", type=str, default="")
    parser.add_argument("--rna-path", type=str, default="")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--disable-internal-lookup", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    return args


def apply_overrides(args, overrides):
    for key, value in overrides.items():
        setattr(args, str(key).replace("-", "_"), value)
    return args


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def ensure_output_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_json_object(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"json object expected at {path}")
    return obj


def find_row(rows, candidate_keys, target_value):
    target = normalize_text(target_value)
    if not target:
        return None
    for key in candidate_keys:
        for row in rows:
            if normalize_text(row.get(key)) == target:
                return row
    return None


def choose_single_row(rows):
    if len(rows) == 1:
        return rows[0]
    return None


def summarize_path(path_value):
    path_text = normalize_text(path_value)
    if not path_text:
        return {
            "available": False,
            "path": "",
            "exists": False,
            "kind": "",
        }
    path = Path(path_text)
    summary = {
        "available": path.exists(),
        "path": str(path.resolve()) if path.exists() else str(path),
        "exists": bool(path.exists()),
        "kind": "",
    }
    if path.exists():
        if path.is_dir():
            summary["kind"] = "directory"
            try:
                summary["child_count"] = int(sum(1 for _ in path.iterdir()))
            except Exception:
                summary["child_count"] = None
        else:
            summary["kind"] = "file"
            summary["suffix"] = "".join(path.suffixes)
            try:
                summary["size_bytes"] = int(path.stat().st_size)
            except Exception:
                summary["size_bytes"] = None
    return summary


def validate_required_path(path_value, label):
    path_text = normalize_text(path_value)
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path.resolve())


def score_pet_series(series_description, num_images):
    desc = normalize_text(series_description).lower()
    score = 0
    if "pet" in desc or "wb" in desc or "whole body" in desc:
        score += 5
    if "attenuation" in desc or "ac" in desc:
        score -= 2
    score += min(int(num_images / 100), 5)
    return score


def pick_primary_pet_series(patient_id, series_index):
    candidates = []
    for row in series_index.get(patient_id, []):
        if normalize_text(row.get("Modality")) != "PT":
            continue
        desc = row.get("Series Description") or ""
        num_images = int(float(row.get("Number of Images") or 0))
        candidates.append((score_pet_series(desc, num_images), num_images, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def summarize_series_row(row):
    if row is None:
        return {"available": False, "path": "", "exists": False, "kind": ""}
    series_dir = IMAGING_PREP.resolve_series_dir(row.get("File Location") or "")
    out = summarize_path(series_dir)
    out.update(
        {
            "available": True,
            "modality": normalize_text(row.get("Modality")),
            "series_description": normalize_text(row.get("Series Description")),
            "study_description": normalize_text(row.get("Study Description")),
            "number_of_images": int(float(row.get("Number of Images") or 0)),
            "study_date": normalize_text(row.get("Study Date")),
            "series_instance_uid": normalize_text(row.get("Series Instance UID")),
        }
    )
    return out


def load_dense_row_summary(csv_path, patient_id):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {"available": False, "path": str(csv_path), "exists": False}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "patient_id" not in fieldnames:
            return {"available": False, "path": str(csv_path.resolve()), "exists": True}
        value_cols = [name for name in fieldnames if name != "patient_id"]
        for row in reader:
            if normalize_text(row.get("patient_id")) != patient_id:
                continue
            return {
                "available": True,
                "path": str(csv_path.resolve()),
                "exists": True,
                "row_available": True,
                "feature_dim": len(value_cols),
            }
    return {
        "available": False,
        "path": str(csv_path.resolve()),
        "exists": True,
        "row_available": False,
    }


def load_npz_row_summary(npz_path, patient_id):
    npz_path = Path(npz_path)
    if np is None:
        return {"available": False, "path": str(npz_path), "exists": npz_path.exists(), "note": "numpy unavailable"}
    if not npz_path.exists():
        return {"available": False, "path": str(npz_path), "exists": False}
    with np.load(npz_path, allow_pickle=True) as z:
        patient_ids = [str(x) for x in z["patient_ids"].tolist()]
        gene_count = int(z["gene_ids"].shape[0]) if "gene_ids" in z else None
        return {
            "available": patient_id in patient_ids,
            "path": str(npz_path.resolve()),
            "exists": True,
            "row_available": patient_id in patient_ids,
            "gene_count": gene_count,
            "patient_count": len(patient_ids),
        }


def load_manifest_lookup(path):
    rows = read_csv_rows(path)
    return {normalize_text(row.get("patient_id")): row for row in rows if normalize_text(row.get("patient_id"))}


def load_internal_context():
    manifest_lookup = load_manifest_lookup(DEFAULT_MANIFEST_CSV) if DEFAULT_MANIFEST_CSV.exists() else {}
    labels_lookup = load_manifest_lookup(DEFAULT_LABELS_CSV) if DEFAULT_LABELS_CSV.exists() else {}
    metadata_rows = IMAGING_PREP.load_metadata_rows() if IMAGING_PREP.METADATA_CSV.exists() else []
    series_index = IMAGING_PREP.build_series_index(metadata_rows)
    clinical_csv_path = TOTAL_TABLE.resolve_input_path(
        TOTAL_TABLE.PRIMARY_CLINICAL_CSV,
        TOTAL_TABLE.LEGACY_CLINICAL_CSV,
        "clinical csv",
    )
    clinical_rows = read_csv_rows(clinical_csv_path)
    return {
        "manifest_lookup": manifest_lookup,
        "labels_lookup": labels_lookup,
        "series_index": series_index,
        "clinical_csv_path": clinical_csv_path,
        "clinical_rows": clinical_rows,
        "series_matrix_path": TOTAL_TABLE.resolve_input_path(
            TOTAL_TABLE.PRIMARY_SERIES_MATRIX_TXT,
            TOTAL_TABLE.LEGACY_SERIES_MATRIX_TXT,
            "series matrix",
        ),
    }


def build_internal_case_bundle(patient_id, context):
    manifest_row = context["manifest_lookup"].get(patient_id)
    if manifest_row is None:
        return None

    ct_row = IMAGING_PREP.pick_primary_ct_series(patient_id, context["series_index"])
    pet_row = pick_primary_pet_series(patient_id, context["series_index"])
    seg_row = IMAGING_PREP.pick_seg_series(patient_id, context["series_index"])
    aim_path = IMAGING_PREP.find_aim_file(patient_id)
    clinical_row = find_row(context["clinical_rows"], ["Case ID", "patient_id"], patient_id)
    labels_row = context["labels_lookup"].get(patient_id)

    bundle = {
        "source_mode": "internal",
        "patient_id": patient_id,
        "patient_manifest_row": manifest_row,
        "labels_row": labels_row,
        "clinical_row": clinical_row,
        "modalities": {
            "ct": summarize_series_row(ct_row),
            "pet": summarize_series_row(pet_row),
            "tumor_segmentation": summarize_series_row(seg_row),
            "aim": summarize_path(aim_path),
            "clinical_tabular": {
                "available": clinical_row is not None,
                "path": str(context["clinical_csv_path"]),
                "source": "internal_clinical_csv",
                "id_column": "Case ID",
            },
            "rna": {
                "available": normalize_text(manifest_row.get("has_rnaseq")) == "1",
                "source": "internal_rna_alignment",
                "gsm_id": normalize_text(manifest_row.get("gsm_id")),
                "series_matrix_path": str(context["series_matrix_path"]),
            },
        },
        "derived_assets": {
            "ct_preprocessed_npz": summarize_path(ROOT / f"output/preprocessed/ct_norm/{patient_id}.npz"),
            "seg_preprocessed_npz": summarize_path(ROOT / f"output/preprocessed/seg_masks/{patient_id}.npz"),
            "stage7_x_rna": load_npz_row_summary(DEFAULT_STAGE7_X_RNA_NPZ, patient_id),
            "stage7_g_rna": load_dense_row_summary(DEFAULT_STAGE7_G_RNA_CSV, patient_id),
            "stage8_x_ehr": load_dense_row_summary(DEFAULT_STAGE8_X_EHR_CSV, patient_id),
            "stage8_g_ehr": load_dense_row_summary(DEFAULT_STAGE8_G_EHR_CSV, patient_id),
        },
    }
    return bundle


def build_external_clinical_payload(args, patient_id):
    clinical_json = normalize_text(args.clinical_json)
    clinical_csv = normalize_text(args.clinical_csv)
    row_id = normalize_text(args.clinical_row_id) or patient_id
    id_column = normalize_text(args.clinical_id_column)

    if clinical_json:
        validate_required_path(clinical_json, "clinical json")
        return {
            "row": read_json_object(clinical_json),
            "summary": {
                "available": True,
                "path": str(Path(clinical_json).resolve()),
                "source": "external_clinical_json",
                "id_column": "",
            },
        }

    if clinical_csv:
        csv_path = Path(validate_required_path(clinical_csv, "clinical csv"))
        rows = read_csv_rows(csv_path)
        candidate_keys = [id_column] if id_column else []
        candidate_keys.extend(["patient_id", "Case ID", "case_id", "PatientID"])
        row = find_row(rows, candidate_keys, row_id)
        if row is None:
            row = choose_single_row(rows)
        if row is None:
            raise RuntimeError(
                f"failed to locate clinical row for patient_id='{patient_id}' in {csv_path}"
            )
        return {
            "row": row,
            "summary": {
                "available": True,
                "path": str(csv_path),
                "source": "external_clinical_csv",
                "id_column": id_column or "auto",
            },
        }
    return {"row": None, "summary": {"available": False, "path": "", "source": ""}}


def overlay_explicit_paths(bundle, args):
    explicit_used = False
    if normalize_text(args.ct_path):
        explicit_used = True
        bundle["modalities"]["ct"] = summarize_path(validate_required_path(args.ct_path, "ct path"))
        bundle["modalities"]["ct"]["source"] = "explicit_path"
    if normalize_text(args.pet_path):
        explicit_used = True
        bundle["modalities"]["pet"] = summarize_path(validate_required_path(args.pet_path, "pet path"))
        bundle["modalities"]["pet"]["source"] = "explicit_path"
    if normalize_text(args.tumor_seg_path):
        explicit_used = True
        bundle["modalities"]["tumor_segmentation"] = summarize_path(
            validate_required_path(args.tumor_seg_path, "tumor segmentation path")
        )
        bundle["modalities"]["tumor_segmentation"]["source"] = "explicit_path"
    if normalize_text(args.aim_path):
        explicit_used = True
        bundle["modalities"]["aim"] = summarize_path(validate_required_path(args.aim_path, "aim path"))
        bundle["modalities"]["aim"]["source"] = "explicit_path"
    if normalize_text(args.rna_path):
        explicit_used = True
        bundle["modalities"]["rna"] = summarize_path(validate_required_path(args.rna_path, "rna path"))
        bundle["modalities"]["rna"]["source"] = "explicit_path"

    external_clinical = build_external_clinical_payload(args, bundle["patient_id"])
    if external_clinical["summary"]["available"]:
        explicit_used = True
        bundle["clinical_row"] = external_clinical["row"]
        bundle["modalities"]["clinical_tabular"] = external_clinical["summary"]

    if explicit_used:
        if bundle["source_mode"] == "internal":
            bundle["source_mode"] = "hybrid"
        else:
            bundle["source_mode"] = "external"


def build_case_bundle(args):
    patient_id = normalize_text(args.patient_id)
    if not patient_id:
        raise RuntimeError("--patient-id is required")

    internal_bundle = None
    if not args.disable_internal_lookup:
        internal_bundle = build_internal_case_bundle(patient_id, load_internal_context())

    if internal_bundle is None:
        bundle = {
            "source_mode": "external",
            "patient_id": patient_id,
            "patient_manifest_row": None,
            "labels_row": None,
            "clinical_row": None,
            "modalities": {
                "ct": {"available": False, "path": "", "exists": False, "kind": ""},
                "pet": {"available": False, "path": "", "exists": False, "kind": ""},
                "tumor_segmentation": {"available": False, "path": "", "exists": False, "kind": ""},
                "aim": {"available": False, "path": "", "exists": False, "kind": ""},
                "clinical_tabular": {"available": False, "path": "", "source": ""},
                "rna": {"available": False, "path": "", "source": ""},
            },
            "derived_assets": {},
        }
    else:
        bundle = internal_bundle

    overlay_explicit_paths(bundle, args)
    return finalize_bundle(bundle)


def finalize_bundle(bundle):
    warnings = []
    infos = []
    missing_required = []

    has_ct = bool(bundle["modalities"]["ct"].get("available"))
    has_clinical = bool(bundle["modalities"]["clinical_tabular"].get("available"))
    has_seg = bool(bundle["modalities"]["tumor_segmentation"].get("available"))
    has_pet = bool(bundle["modalities"]["pet"].get("available"))
    has_rna = bool(bundle["modalities"]["rna"].get("available"))
    has_aim = bool(bundle["modalities"]["aim"].get("available"))

    if not has_ct:
        missing_required.append("ct")
    if not has_clinical:
        missing_required.append("clinical_tabular")
    if not has_seg:
        warnings.append(
            "Tumor segmentation is missing. The current pipeline can still run with missing masks, "
            "but tumor-specific evidence will be absent; external cases may need a segmenter."
        )
    if not has_rna:
        infos.append("RNA is optional and can be omitted; the pipeline will use missing-modality masking.")
    if not has_pet:
        infos.append("PET is optional. The current implemented Stage 9-13 pipeline is CT-centric.")
    if not has_aim:
        infos.append("AIM semantics are optional; missing semantic evidence will be masked.")

    bundle["availability"] = {
        "has_ct": int(has_ct),
        "has_pet": int(has_pet),
        "has_tumor_segmentation": int(has_seg),
        "has_aim": int(has_aim),
        "has_clinical_tabular": int(has_clinical),
        "has_rna": int(has_rna),
    }
    bundle["current_pipeline_requirements"] = {
        "required": ["ct", "clinical_tabular"],
        "optional": ["pet", "tumor_segmentation", "aim", "rna"],
    }
    bundle["ready_for_current_pipeline"] = len(missing_required) == 0
    bundle["missing_required_inputs"] = missing_required
    bundle["warnings"] = warnings
    bundle["notes"] = infos
    return bundle


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_sources_rows(bundle):
    rows = []
    for name, payload in bundle["modalities"].items():
        rows.append(
            {
                "group": "modality",
                "name": name,
                "available": int(bool(payload.get("available"))),
                "path": normalize_text(payload.get("path")),
                "source": normalize_text(payload.get("source")),
                "kind": normalize_text(payload.get("kind")),
            }
        )
    for name, payload in bundle.get("derived_assets", {}).items():
        rows.append(
            {
                "group": "derived_asset",
                "name": name,
                "available": int(bool(payload.get("available"))),
                "path": normalize_text(payload.get("path")),
                "source": normalize_text(payload.get("source")),
                "kind": normalize_text(payload.get("kind")),
            }
        )
    return rows


def write_sources_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "name", "available", "path", "source", "kind"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None, **overrides):
    args = apply_overrides(parse_args(argv), overrides)
    if not normalize_text(getattr(args, "patient_id", "")):
        raise RuntimeError(
            "patient_id is required. In Python Console, call "
            "run_case_inputs(patient_id='R01-003')."
        )
    bundle = build_case_bundle(args)
    output_dir = ensure_output_dir(Path(args.output_root) / bundle["patient_id"])

    bundle_path = output_dir / "case_input_bundle.json"
    summary_path = output_dir / "case_input_summary.json"
    sources_csv_path = output_dir / "case_input_sources.csv"

    summary = {
        "patient_id": bundle["patient_id"],
        "source_mode": bundle["source_mode"],
        "ready_for_current_pipeline": bool(bundle["ready_for_current_pipeline"]),
        "availability": bundle["availability"],
        "missing_required_inputs": bundle["missing_required_inputs"],
        "warnings": bundle["warnings"],
        "notes": bundle["notes"],
    }

    write_json(bundle_path, bundle)
    write_json(summary_path, summary)
    write_sources_csv(sources_csv_path, build_sources_rows(bundle))

    if bundle.get("clinical_row") is not None:
        write_json(output_dir / "clinical_row.json", bundle["clinical_row"])
    if bundle.get("patient_manifest_row") is not None:
        write_json(output_dir / "patient_manifest_row.json", bundle["patient_manifest_row"])
    if bundle.get("labels_row") is not None:
        write_json(output_dir / "labels_row.json", bundle["labels_row"])

    print(f"wrote: {bundle_path}")
    print(f"wrote: {summary_path}")
    print(f"wrote: {sources_csv_path}")
    if bundle.get("clinical_row") is not None:
        print(f"wrote: {output_dir / 'clinical_row.json'}")
    if bundle.get("patient_manifest_row") is not None:
        print(f"wrote: {output_dir / 'patient_manifest_row.json'}")
    if bundle.get("labels_row") is not None:
        print(f"wrote: {output_dir / 'labels_row.json'}")
    print(f"ready_for_current_pipeline: {bundle['ready_for_current_pipeline']}")
    print("complete")
    return {
        "patient_id": bundle["patient_id"],
        "output_dir": str(output_dir),
        "bundle_path": str(bundle_path),
        "summary_path": str(summary_path),
        "sources_csv_path": str(sources_csv_path),
        "ready_for_current_pipeline": bool(bundle["ready_for_current_pipeline"]),
    }


def run_case_inputs(
    *,
    patient_id,
    ct_path="",
    pet_path="",
    tumor_seg_path="",
    aim_path="",
    clinical_csv="",
    clinical_json="",
    clinical_row_id="",
    clinical_id_column="",
    rna_path="",
    output_root=str(DEFAULT_OUTPUT_ROOT),
    disable_internal_lookup=False,
):
    return main(
        patient_id=patient_id,
        ct_path=ct_path,
        pet_path=pet_path,
        tumor_seg_path=tumor_seg_path,
        aim_path=aim_path,
        clinical_csv=clinical_csv,
        clinical_json=clinical_json,
        clinical_row_id=clinical_row_id,
        clinical_id_column=clinical_id_column,
        rna_path=rna_path,
        output_root=output_root,
        disable_internal_lookup=disable_internal_lookup,
    )


if __name__ == "__main__":
    main()
