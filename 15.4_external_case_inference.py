import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError:
    nib = None


def resolve_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


ROOT = resolve_root()
DEFAULT_OUTPUT_ROOT = ROOT / "output/stage15/15.4_external_case_inference"
DEFAULT_CASE_INPUT_ROOT = DEFAULT_OUTPUT_ROOT / "case_inputs"
DEFAULT_SYSTEM_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "system_outputs"
DEFAULT_RUNTIME_ROOT = DEFAULT_OUTPUT_ROOT / "runtime"
DEFAULT_VIS_ROOT = DEFAULT_OUTPUT_ROOT / "visualization"

DEFAULT_STAGE81_NPZ = ROOT / "output/stage8/8.1_clinical_feature_engineering/x_ehr_features.npz"
DEFAULT_STAGE81_CONT_STATS = ROOT / "output/stage8/8.1_clinical_feature_engineering/continuous_stats.csv"
DEFAULT_STAGE81_CAT_VOCAB = ROOT / "output/stage8/8.1_clinical_feature_engineering/categorical_vocab.csv"
DEFAULT_STAGE71_NPZ = ROOT / "output/stage7/7.1_rna_alignment/x_rna_log1p_zscore.npz"
DEFAULT_EHR_MODEL = ROOT / "output/stage8/8.2_ehr_encoder/model/ehr_encoder.pt"
DEFAULT_RNA_MODEL = ROOT / "output/stage7/7.2_rna_encoder/model/rna_encoder.pt"
DEFAULT_IMM_MODEL = ROOT / "output/stage7/7.3_immune_token/model/immune_token_mlp.pt"
DEFAULT_PHASE3_MODEL = ROOT / "output/stage13/13.1_phase3_baseline/phase3_model/model/explanation_guided_model.pt"
DEFAULT_PHASE4_MODEL = ROOT / "output/stage13/13.2_phase4_tune/phase4_model_best/model/explanation_guided_model.pt"
DEFAULT_STAGE10_SUMMARY = ROOT / "output/stage10/10.1_multimodal_fusion/fusion_summary.json"
DEFAULT_STAGE11_GRAPH_SUMMARY = ROOT / "output/stage11/11.1_graph_construction/graph_construction_summary.json"
DEFAULT_STAGE11_REASON_SUMMARY = ROOT / "output/stage11/11.2_graph_reasoning/graph_reasoning_summary.json"
DEFAULT_STAGE10_ATTN_NPZ = ROOT / "output/stage10/10.1_multimodal_fusion/fused_organ_tokens.npz"

EXPLANATION_SEMANTICS = (
    "Latent diffusion explanation only; organ/path outputs are model-induced explanation "
    "layers for OS/recurrence predictions and do not imply organ-level ground-truth supervision."
)


def load_local_module(path, module_name):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASE_INPUT_MOD = load_local_module(ROOT / "15.1_case_inputs.py", "stage15_case_inputs")
PHASE_UTILS = load_local_module(ROOT / "13.0_phase_utils.py", "stage15_phase_utils")
STAGE6_MOD = load_local_module(ROOT / "6.2_infer_mask.py", "stage6_infer_mask")
STAGE71_RNA_MOD = load_local_module(ROOT / "7.2_rna_encoder.py", "stage72_rna_encoder")
STAGE73_IMM_MOD = load_local_module(ROOT / "7.3_immune_token.py", "stage73_immune_token")
STAGE81_MOD = load_local_module(ROOT / "8.1_clinical_feature_engineering.py", "stage81_clinical")
STAGE82_MOD = load_local_module(ROOT / "8.2_ehr_encoder.py", "stage82_ehr_encoder")
STAGE9_MOD = load_local_module(ROOT / "9.1_organ_tokenization.py", "stage9_organ_tokenization")
STAGE10_MOD = load_local_module(ROOT / "10.1_multimodal_fusion.py", "stage10_multimodal_fusion")
STAGE11_GRAPH_MOD = load_local_module(ROOT / "11.1_graph_construction.py", "stage11_graph_construction")
STAGE11_REASON_MOD = load_local_module(ROOT / "11.2_graph_reasoning.py", "stage11_graph_reasoning")
EXPL_TRAIN_MOD = load_local_module(ROOT / "12.2_explanation_training.py", "stage12_explanation_training")
EXPL_OUT_MOD = load_local_module(ROOT / "12.2_explanation_outputs.py", "stage12_explanation_outputs")
VIS_MOD = load_local_module(ROOT / "13.4_visualize_diffusion.py", "stage13_visualize_diffusion")
IMAGING_PREP_MOD = load_local_module(
    ROOT / "prepare_clean/imaging_preprocessing.py",
    "stage5_imaging_preprocessing",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run true end-to-end inference for a new external case",
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
    parser.add_argument("--disable-internal-lookup", action="store_true")
    parser.add_argument("--force-external-inference", action="store_true")
    parser.add_argument("--model-strategy", choices=["auto", "phase3", "phase4"], default="auto")
    parser.add_argument("--rna-transform", choices=["raw", "log1p", "zscore"], default="raw")
    parser.add_argument("--organ-seg-run-tag", type=str, default="search_base24")
    parser.add_argument("--organ-seg-model-path", type=str, default="")
    parser.add_argument("--allow-legacy-model-fallback", action="store_true")
    parser.add_argument("--phase3-model-path", type=str, default=str(DEFAULT_PHASE3_MODEL))
    parser.add_argument("--phase4-model-path", type=str, default=str(DEFAULT_PHASE4_MODEL))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
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
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def choose_device(device_arg):
    value = str(device_arg).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested cuda but cuda is not available")
    return torch.device(value)


def load_case_bundle_and_write(args, case_input_root):
    bundle = CASE_INPUT_MOD.build_case_bundle(args)
    output_dir = ensure_output_dir(Path(case_input_root) / bundle["patient_id"])
    summary = {
        "patient_id": bundle["patient_id"],
        "source_mode": bundle["source_mode"],
        "ready_for_current_pipeline": bool(bundle["ready_for_current_pipeline"]),
        "availability": bundle["availability"],
        "missing_required_inputs": bundle["missing_required_inputs"],
        "warnings": bundle["warnings"],
        "notes": bundle["notes"],
    }
    CASE_INPUT_MOD.write_json(output_dir / "case_input_bundle.json", bundle)
    CASE_INPUT_MOD.write_json(output_dir / "case_input_summary.json", summary)
    CASE_INPUT_MOD.write_sources_csv(
        output_dir / "case_input_sources.csv",
        CASE_INPUT_MOD.build_sources_rows(bundle),
    )
    if bundle.get("clinical_row") is not None:
        CASE_INPUT_MOD.write_json(output_dir / "clinical_row.json", bundle["clinical_row"])
    if bundle.get("patient_manifest_row") is not None:
        CASE_INPUT_MOD.write_json(output_dir / "patient_manifest_row.json", bundle["patient_manifest_row"])
    if bundle.get("labels_row") is not None:
        CASE_INPUT_MOD.write_json(output_dir / "labels_row.json", bundle["labels_row"])
    return bundle, summary, output_dir


def load_optional_npz(path):
    with np.load(path, allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def prepare_ct_volume(patient_id, ct_path, runtime_root, deps):
    ct_path = Path(ct_path)
    prepared_ct_dir = ensure_output_dir(Path(runtime_root) / "prepared" / "ct_norm")
    ct_npz_path = prepared_ct_dir / f"{patient_id}.npz"

    if ct_path.is_dir():
        ct_result = IMAGING_PREP_MOD.load_ct_volume_and_normalize(ct_path, deps)
        if ct_result.get("status") != "ok":
            raise RuntimeError(f"ct preprocessing failed: {ct_result.get('status')} {ct_result.get('error', '')}")
        np.savez_compressed(
            ct_npz_path,
            ct_volume=np.asarray(ct_result["volume_norm"], dtype=np.float32),
            spacing_target=np.asarray(ct_result["spacing_target"], dtype=np.float32),
        )
        return {
            "source_path": str(ct_path),
            "source_kind": "dicom_series_dir",
            "ct_npz_path": str(ct_npz_path),
            "volume_norm": np.asarray(ct_result["volume_norm"], dtype=np.float32),
            "volume_norm_native": np.asarray(ct_result["volume_norm_native"], dtype=np.float32),
            "volume_original": np.asarray(ct_result["volume_original"], dtype=np.float32),
            "spacing_target": tuple(float(x) for x in ct_result["spacing_target"]),
            "ct_geometry": ct_result.get("ct_geometry", {}),
            "ct_status": "ok",
        }

    if ct_path.suffix.lower() == ".npz":
        payload = load_optional_npz(ct_path)
        if "ct_volume" in payload:
            volume = np.asarray(payload["ct_volume"], dtype=np.float32)
        elif "volume_norm" in payload:
            volume = np.asarray(payload["volume_norm"], dtype=np.float32)
        else:
            raise RuntimeError("ct npz must contain ct_volume or volume_norm")
        spacing_target = payload.get("spacing_target", np.asarray(IMAGING_PREP_MOD.TARGET_SPACING, dtype=np.float32))
        np.savez_compressed(
            ct_npz_path,
            ct_volume=volume.astype(np.float32),
            spacing_target=np.asarray(spacing_target, dtype=np.float32),
        )
        return {
            "source_path": str(ct_path),
            "source_kind": "preprocessed_npz",
            "ct_npz_path": str(ct_npz_path),
            "volume_norm": volume.astype(np.float32),
            "volume_norm_native": volume.astype(np.float32),
            "volume_original": volume.astype(np.float32),
            "spacing_target": tuple(float(x) for x in np.asarray(spacing_target, dtype=np.float32).tolist()),
            "ct_geometry": {},
            "ct_status": "ok",
        }

    raise RuntimeError(
        "unsupported ct input. expected a DICOM directory or a preprocessed .npz with ct_volume."
    )


def load_binary_mask_array(mask_path):
    mask_path = Path(mask_path)
    suffixes = [x.lower() for x in mask_path.suffixes]

    if mask_path.suffix.lower() == ".npy":
        return np.asarray(np.load(mask_path, allow_pickle=True), dtype=np.float32)

    if mask_path.suffix.lower() == ".npz":
        payload = load_optional_npz(mask_path)
        for key in ("mask", "tumor_mask", "binary_mask", "segmentation"):
            if key in payload and np.asarray(payload[key]).ndim == 3:
                return np.asarray(payload[key], dtype=np.float32)
        for key in payload:
            arr = np.asarray(payload[key])
            if arr.ndim == 3:
                return arr.astype(np.float32)
        raise RuntimeError(f"no 3D array found in segmentation npz: {mask_path}")

    if suffixes[-2:] == [".nii", ".gz"] or mask_path.suffix.lower() == ".nii":
        if nib is None:
            raise RuntimeError("nibabel is required to read NIfTI segmentation files")
        return np.asarray(nib.load(str(mask_path)).get_fdata(), dtype=np.float32)

    raise RuntimeError("unsupported segmentation file. expected .npy, .npz, or .nii/.nii.gz")


def prepare_tumor_mask(patient_id, seg_path, ct_context, runtime_root, deps):
    prepared_seg_dir = ensure_output_dir(Path(runtime_root) / "prepared" / "seg_masks")
    seg_npz_path = prepared_seg_dir / f"{patient_id}.npz"

    if not normalize_text(seg_path):
        return {
            "status": "missing",
            "seg_npz_path": "",
            "mask": None,
            "source_kind": "",
            "note": "tumor segmentation not provided",
        }

    seg_path = Path(seg_path)
    if seg_path.is_dir() or seg_path.suffix.lower() == ".dcm":
        if ct_context["source_kind"] != "dicom_series_dir":
            raise RuntimeError(
                "dicom segmentation input requires CT to be provided as the raw DICOM series directory"
            )
        seg_dir = seg_path if seg_path.is_dir() else seg_path.parent
        seg_result = IMAGING_PREP_MOD.load_tumor_mask(
            seg_dir,
            ct_context["volume_original"].shape,
            ct_context.get("ct_geometry", {}),
            deps,
        )
        if seg_result.get("status") != "ok":
            return {
                "status": seg_result.get("status", "seg_failed"),
                "seg_npz_path": "",
                "mask": None,
                "source_kind": "dicom_seg",
                "note": seg_result.get("error", ""),
                "align_mode": seg_result.get("align_mode", ""),
                "fail_reason": seg_result.get("fail_reason", ""),
            }
        mask = np.asarray(seg_result["mask"], dtype=np.uint8)
        np.savez_compressed(seg_npz_path, mask=mask)
        return {
            "status": "ok",
            "seg_npz_path": str(seg_npz_path),
            "mask": mask,
            "source_kind": "dicom_seg",
            "align_mode": seg_result.get("align_mode", ""),
            "fail_reason": seg_result.get("fail_reason", ""),
        }

    mask = load_binary_mask_array(seg_path)
    if mask.shape != ct_context["volume_norm_native"].shape:
        raise RuntimeError(
            f"segmentation shape mismatch: expected {tuple(ct_context['volume_norm_native'].shape)} got {tuple(mask.shape)}"
        )
    mask = (mask > 0).astype(np.uint8)
    np.savez_compressed(seg_npz_path, mask=mask)
    return {
        "status": "ok",
        "seg_npz_path": str(seg_npz_path),
        "mask": mask,
        "source_kind": "aligned_mask_file",
        "align_mode": "pre_aligned",
        "fail_reason": "",
    }


def prepare_semantic_token(aim_path):
    if not normalize_text(aim_path):
        return {
            "status": "missing",
            "token": None,
            "aim_path": "",
        }
    aim_path = Path(aim_path)
    feature_texts = IMAGING_PREP_MOD.parse_aim_feature_texts(str(aim_path))
    token = IMAGING_PREP_MOD.build_semantic_token(feature_texts, IMAGING_PREP_MOD.SEMANTIC_TOKEN_DIM)
    if not token:
        return {
            "status": "aim_missing_or_empty",
            "token": None,
            "aim_path": str(aim_path),
        }
    return {
        "status": "ok",
        "token": np.asarray(token, dtype=np.float32),
        "aim_path": str(aim_path),
    }


def prepare_roi_token(ct_context, tumor_context, deps):
    if tumor_context.get("mask") is None:
        return {"status": "missing", "token": None}
    roi_result = IMAGING_PREP_MOD.compute_roi_token(
        ct_context["volume_norm_native"],
        np.asarray(tumor_context["mask"], dtype=np.uint8),
        deps,
    )
    if roi_result.get("status") != "ok":
        return {
            "status": roi_result.get("status", "roi_failed"),
            "token": None,
        }
    return {
        "status": "ok",
        "token": np.asarray(roi_result["token"], dtype=np.float32),
    }


def run_stage6_single_case(patient_id, ct_context, runtime_root, device, args):
    model_path = STAGE6_MOD.resolve_model_path(
        model_path_arg=normalize_text(args.organ_seg_model_path),
        output_root=ROOT / "output/experiments/organ_seg",
        run_tag=args.organ_seg_run_tag,
        allow_legacy_model_fallback=bool(args.allow_legacy_model_fallback),
    )
    infer_paths = STAGE6_MOD.resolve_infer_paths(runtime_root / "stage6", args.organ_seg_run_tag)
    STAGE6_MOD.ensure_output_dirs(infer_paths["infer_root"], infer_paths["mask_dir"])

    model, meta = STAGE6_MOD.load_model(model_path, device)
    pred_mask, token_sums, token_voxel_counts = STAGE6_MOD.infer_volume_multilabel_mask_and_token_stats(
        model=model,
        ct_volume=np.asarray(ct_context["volume_norm"], dtype=np.float32),
        image_size=int(meta["image_size"]),
        batch_slices=8,
        device=device,
        num_context_slices=int(meta["num_context_slices"]),
        slice_stride=int(meta["slice_stride"]),
        organ_map=meta["organ_map"],
    )
    missing_dict, voxel_dict = STAGE6_MOD.build_missing_flags(
        pred_mask=pred_mask,
        organ_map=meta["organ_map"],
        min_organ_voxels=50,
    )
    mask_npz_path = infer_paths["mask_dir"] / f"{patient_id}.npz"
    STAGE6_MOD.save_patient_mask_npz(
        path=mask_npz_path,
        pred_mask=pred_mask,
        organ_map=meta["organ_map"],
        source_ct_npz=ct_context["ct_npz_path"],
    )
    token_rows = STAGE6_MOD.extract_organ_tokens_for_case(
        patient_id=patient_id,
        organ_map=meta["organ_map"],
        mask_npz_path=mask_npz_path,
        min_organ_voxels=50,
        token_sums=token_sums,
        token_voxel_counts=token_voxel_counts,
    )
    for row in token_rows:
        row["run_tag"] = args.organ_seg_run_tag
        row["model_path"] = str(model_path)
        row["batch_slices"] = 8
        row["min_organ_voxels"] = 50
        row["token_dim"] = int(meta["token_dim"])

    manifest_rows = [
        {
            "run_tag": args.organ_seg_run_tag,
            "patient_id": patient_id,
            "ct_npz_path": str(ct_context["ct_npz_path"]),
            "mask_npz_path": str(mask_npz_path),
            "model_path": str(model_path),
            "batch_slices": 8,
            "min_organ_voxels": 50,
            "status": "ok",
            "error": "",
            "missing_img_organ_json": json.dumps(missing_dict),
            "organ_voxel_count_json": json.dumps(voxel_dict),
        }
    ]
    long_rows = []
    for organ_id, organ_name in meta["organ_map"].items():
        long_rows.append(
            {
                "run_tag": args.organ_seg_run_tag,
                "patient_id": patient_id,
                "organ_id": organ_id,
                "organ_name": organ_name,
                "model_path": str(model_path),
                "batch_slices": 8,
                "min_organ_voxels": 50,
                "voxel_count": voxel_dict[organ_name],
                "missing_img_organ": missing_dict[organ_name],
                "mask_npz_path": str(mask_npz_path),
                "status": "ok",
            }
        )

    write_csv(
        infer_paths["manifest_csv"],
        [
            "run_tag",
            "patient_id",
            "ct_npz_path",
            "mask_npz_path",
            "model_path",
            "batch_slices",
            "min_organ_voxels",
            "status",
            "error",
            "missing_img_organ_json",
            "organ_voxel_count_json",
        ],
        manifest_rows,
    )
    write_csv(
        infer_paths["long_csv"],
        [
            "run_tag",
            "patient_id",
            "organ_id",
            "organ_name",
            "model_path",
            "batch_slices",
            "min_organ_voxels",
            "voxel_count",
            "missing_img_organ",
            "mask_npz_path",
            "status",
        ],
        long_rows,
    )
    write_csv(
        infer_paths["token_csv"],
        [
            "run_tag",
            "patient_id",
            "organ_id",
            "organ_name",
            "model_path",
            "batch_slices",
            "min_organ_voxels",
            "token_dim",
            "voxel_count",
            "missing_img_organ",
            "token_json",
            "mask_npz_path",
            "status",
        ],
        token_rows,
    )
    return {
        "token_rows": token_rows,
        "token_csv_path": str(infer_paths["token_csv"]),
        "mask_npz_path": str(mask_npz_path),
        "organ_map": meta["organ_map"],
        "voxel_counts": voxel_dict,
        "missing_flags": missing_dict,
        "model_path": str(model_path),
    }


def load_ehr_feature_artifacts():
    if not DEFAULT_STAGE81_NPZ.exists():
        raise FileNotFoundError(f"stage8.1 npz not found: {DEFAULT_STAGE81_NPZ}")
    with np.load(DEFAULT_STAGE81_NPZ, allow_pickle=True) as z:
        feature_names = np.asarray(z["feature_names"]).astype(str).tolist()
    continuous_rows = []
    with DEFAULT_STAGE81_CONT_STATS.open(encoding="utf-8", newline="") as f:
        continuous_rows = list(csv.DictReader(f))
    categorical_rows = []
    with DEFAULT_STAGE81_CAT_VOCAB.open(encoding="utf-8", newline="") as f:
        categorical_rows = list(csv.DictReader(f))
    return feature_names, continuous_rows, categorical_rows


def build_ehr_feature_vector(clinical_row):
    feature_names, continuous_rows, categorical_rows = load_ehr_feature_artifacts()
    feature_set = set(feature_names)
    values = {name: 0.0 for name in feature_names}

    for row in continuous_rows:
        column_name = row["column_name"]
        token = STAGE81_MOD.safe_feature_token(column_name)
        cont_name = f"cont::{token}"
        miss_name = f"miss::{token}"
        raw_value = STAGE81_MOD.parse_numeric(clinical_row.get(column_name))
        if raw_value is None:
            if miss_name in feature_set:
                values[miss_name] = 1.0
            continue
        mean_raw = float(row["mean_raw"])
        safe_std = float(row["safe_std_used"])
        values[cont_name] = float((raw_value - mean_raw) / safe_std)
        if miss_name in feature_set:
            values[miss_name] = 0.0

    grouped_vocab = {}
    for row in categorical_rows:
        grouped_vocab.setdefault(row["column_name"], []).append(row)

    for column_name, rows in grouped_vocab.items():
        category_value = STAGE81_MOD.canonical_category(clinical_row.get(column_name))
        col_token = STAGE81_MOD.safe_feature_token(column_name)
        matched = None
        for row in rows:
            if STAGE81_MOD.canonical_category(row["category_value"]) == category_value:
                matched = row["category_value"]
                break
        if matched is None:
            continue
        cat_token = STAGE81_MOD.safe_feature_token(matched)
        feature_name = f"cat::{col_token}::{cat_token}"
        if feature_name in feature_set:
            values[feature_name] = 1.0

    x_ehr = np.asarray([[values[name] for name in feature_names]], dtype=np.float32)
    return x_ehr, feature_names


def encode_ehr(clinical_row, runtime_root, device):
    if clinical_row is None:
        return {
            "x_ehr": None,
            "g_ehr": None,
            "missing": 1,
            "feature_names": [],
            "artifacts": {},
        }

    payload = safe_torch_load(DEFAULT_EHR_MODEL, map_location="cpu")
    model = STAGE82_MOD.EHREncoderMLP(
        input_dim=int(payload["input_dim"]),
        g_dim=int(payload["g_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        dropout=0.1,
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    x_ehr, feature_names = build_ehr_feature_vector(clinical_row)
    if x_ehr.shape[1] != int(payload["input_dim"]):
        raise RuntimeError(
            f"x_ehr input dim mismatch: expected={payload['input_dim']} got={x_ehr.shape[1]}"
        )
    g_ehr = STAGE82_MOD.infer_g_ehr(model=model, x_ehr=x_ehr, device=device, infer_batch_size=1)
    g_ehr = STAGE82_MOD.l2_normalize_rows(g_ehr)

    ehr_dir = ensure_output_dir(Path(runtime_root) / "prepared" / "ehr")
    np.savez_compressed(
        ehr_dir / "ehr_case_features.npz",
        patient_ids=np.asarray(["case"], dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
        x_ehr=x_ehr.astype(np.float32),
        g_ehr=g_ehr.astype(np.float32),
    )
    return {
        "x_ehr": x_ehr,
        "g_ehr": g_ehr,
        "missing": 0,
        "feature_names": feature_names,
        "artifacts": {
            "ehr_case_features_npz": str(ehr_dir / "ehr_case_features.npz"),
        },
    }


def detect_delimiter(sample_text):
    if "\t" in sample_text and sample_text.count("\t") >= sample_text.count(","):
        return "\t"
    return ","


def load_external_rna_map(rna_path):
    rna_path = Path(rna_path)
    if rna_path.suffix.lower() == ".json":
        payload = json.loads(rna_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("rna json must be a gene_id -> value object")
        out = {}
        for key, value in payload.items():
            try:
                out[STAGE73_IMM_MOD.normalize_gene_id(key)] = float(value)
            except Exception:
                continue
        if not out:
            raise RuntimeError("rna json produced no numeric entries")
        return out

    sample = rna_path.read_text(encoding="utf-8-sig", errors="ignore")[:8192]
    delimiter = detect_delimiter(sample)
    with rna_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter=delimiter))
    if not rows:
        raise RuntimeError(f"empty rna file: {rna_path}")

    if len(rows) >= 2 and len(rows[0]) > 2 and len(rows[1]) == len(rows[0]):
        header = rows[0]
        data_row = rows[1]
        out = {}
        for gene_id, value in zip(header, data_row):
            gene_id = STAGE73_IMM_MOD.normalize_gene_id(gene_id)
            try:
                out[gene_id] = float(value)
            except Exception:
                continue
        if out:
            return out

    header = rows[0]
    if len(header) >= 2:
        dict_rows = []
        with rna_path.open(encoding="utf-8-sig", newline="") as f:
            dict_rows = list(csv.DictReader(f, delimiter=delimiter))
        if dict_rows:
            fieldnames = list(dict_rows[0].keys())
            gene_candidates = [
                "gene_id",
                "gene",
                "entrez",
                "entrez_gene_id",
                "id",
                "feature",
            ]
            value_candidates = [
                "expression",
                "value",
                "count",
                "counts",
                "tpm",
                "fpkm",
                "fpkm_uq",
                "log1p",
                "zscore",
            ]
            gene_col = next((name for name in gene_candidates if name in fieldnames), fieldnames[0])
            value_col = next((name for name in value_candidates if name in fieldnames), None)
            if value_col is None and len(fieldnames) >= 2:
                value_col = fieldnames[1]
            out = {}
            for row in dict_rows:
                gene_id = STAGE73_IMM_MOD.normalize_gene_id(row.get(gene_col))
                try:
                    out[gene_id] = float(row.get(value_col, ""))
                except Exception:
                    continue
            if out:
                return out

    out = {}
    for row in rows:
        if len(row) < 2:
            continue
        gene_id = STAGE73_IMM_MOD.normalize_gene_id(row[0])
        try:
            out[gene_id] = float(row[1])
        except Exception:
            continue
    if not out:
        raise RuntimeError(f"failed to parse RNA table: {rna_path}")
    return out


def compute_gene_feature_value(raw_value, mean_log1p, std_log1p, transform):
    safe_std = std_log1p if abs(std_log1p) > 1e-8 else 1.0
    if transform == "raw":
        raw_value = max(float(raw_value), 0.0)
        log_value = float(np.log1p(raw_value))
        return float((log_value - mean_log1p) / safe_std)
    if transform == "log1p":
        log_value = float(raw_value)
        return float((log_value - mean_log1p) / safe_std)
    return float(raw_value)


def encode_rna_and_immune(patient_id, rna_path, runtime_root, device, transform):
    if not normalize_text(rna_path):
        return {
            "g_rna": None,
            "t_imm": None,
            "missing": 1,
            "gene_coverage": {},
            "artifacts": {},
        }

    rna_map = load_external_rna_map(rna_path)
    with np.load(DEFAULT_STAGE71_NPZ, allow_pickle=True) as z:
        gene_ids = np.asarray(z["gene_ids"]).astype(str)
        gene_mean = np.asarray(z["gene_mean_log1p"], dtype=np.float32)
        gene_std = np.asarray(z["gene_std_log1p"], dtype=np.float32)
    gene_index = {STAGE73_IMM_MOD.normalize_gene_id(gid): idx for idx, gid in enumerate(gene_ids.tolist())}

    rna_payload = safe_torch_load(DEFAULT_RNA_MODEL, map_location="cpu")
    selected_gene_ids = [STAGE73_IMM_MOD.normalize_gene_id(x) for x in np.asarray(rna_payload["selected_gene_ids"]).astype(str).tolist()]
    selected_x = np.zeros((1, len(selected_gene_ids)), dtype=np.float32)
    selected_present = 0
    for idx, gene_id in enumerate(selected_gene_ids):
        raw_value = rna_map.get(gene_id)
        if raw_value is None or gene_id not in gene_index:
            continue
        gene_idx = gene_index[gene_id]
        selected_x[0, idx] = compute_gene_feature_value(
            raw_value=raw_value,
            mean_log1p=float(gene_mean[gene_idx]),
            std_log1p=float(gene_std[gene_idx]),
            transform=transform,
        )
        selected_present += 1

    rna_model = STAGE71_RNA_MOD.RNAEncoderMLP(
        input_dim=int(rna_payload["input_dim"]),
        g_dim=int(rna_payload["g_dim"]),
        num_tokens=int(rna_payload["num_tokens"]),
        token_dim=int(rna_payload["token_dim"]),
        dropout=0.1,
    ).to(device)
    rna_model.load_state_dict(rna_payload["state_dict"], strict=True)
    rna_model.eval()
    g_rna, _token_flat = STAGE71_RNA_MOD.infer_embeddings(
        model=rna_model,
        x_selected=selected_x,
        device=device,
        infer_batch_size=1,
    )
    g_rna = STAGE71_RNA_MOD.l2_normalize_rows(g_rna)

    imm_payload = safe_torch_load(DEFAULT_IMM_MODEL, map_location="cpu")
    signature_names = [str(x) for x in np.asarray(imm_payload["signature_names"]).astype(str).tolist()]
    marker_sets = imm_payload["marker_sets"]
    signature_raw = np.zeros((1, len(signature_names)), dtype=np.float32)
    marker_present = set()
    for sig_idx, sig_name in enumerate(signature_names):
        markers = [STAGE73_IMM_MOD.normalize_gene_id(x) for x in marker_sets.get(sig_name, [])]
        values = []
        for gene_id in markers:
            raw_value = rna_map.get(gene_id)
            if raw_value is None or gene_id not in gene_index:
                continue
            gene_idx = gene_index[gene_id]
            values.append(
                compute_gene_feature_value(
                    raw_value=raw_value,
                    mean_log1p=float(gene_mean[gene_idx]),
                    std_log1p=float(gene_std[gene_idx]),
                    transform=transform,
                )
            )
            marker_present.add(gene_id)
        if values:
            signature_raw[0, sig_idx] = float(np.mean(np.asarray(values, dtype=np.float32)))
    signature_mean = np.asarray(imm_payload["signature_mean"], dtype=np.float32)
    signature_std = np.asarray(imm_payload["signature_std"], dtype=np.float32)
    safe_signature_std = np.where(np.abs(signature_std) > 1e-8, signature_std, 1.0)
    signature_z = ((signature_raw - signature_mean.reshape(1, -1)) / safe_signature_std.reshape(1, -1)).astype(np.float32)

    imm_model = STAGE73_IMM_MOD.ImmuneTokenMLP(
        input_dim=int(imm_payload["input_dim"]),
        token_dim=int(imm_payload["token_dim"]),
        hidden_dim=int(imm_payload["hidden_dim"]),
        dropout=0.1,
    ).to(device)
    imm_model.load_state_dict(imm_payload["state_dict"], strict=True)
    imm_model.eval()
    t_imm = STAGE73_IMM_MOD.infer_t_imm(
        model=imm_model,
        sig_z=signature_z,
        device=device,
        infer_batch_size=1,
    )
    t_imm = STAGE73_IMM_MOD.l2_normalize_rows(t_imm)

    rna_dir = ensure_output_dir(Path(runtime_root) / "prepared" / "rna")
    np.savez_compressed(
        rna_dir / "rna_case_features.npz",
        patient_ids=np.asarray([patient_id], dtype=object),
        selected_gene_ids=np.asarray(selected_gene_ids, dtype=object),
        x_rna_selected=selected_x.astype(np.float32),
        g_rna=g_rna.astype(np.float32),
        signature_names=np.asarray(signature_names, dtype=object),
        immune_signatures_raw=signature_raw.astype(np.float32),
        immune_signatures_z=signature_z.astype(np.float32),
        t_imm=t_imm.astype(np.float32),
    )
    return {
        "g_rna": g_rna,
        "t_imm": t_imm,
        "missing": 0,
        "gene_coverage": {
            "selected_gene_present_count": int(selected_present),
            "selected_gene_total_count": int(len(selected_gene_ids)),
            "immune_marker_present_count": int(len(marker_present)),
            "immune_marker_total_count": int(
                len(
                    {
                        STAGE73_IMM_MOD.normalize_gene_id(gene_id)
                        for genes in marker_sets.values()
                        for gene_id in genes
                    }
                )
            ),
        },
        "artifacts": {
            "rna_case_features_npz": str(rna_dir / "rna_case_features.npz"),
        },
    }


def pick_stage6_node_tokens(patient_id, token_rows):
    by_patient = {}
    for row in token_rows:
        if normalize_text(row.get("patient_id")) != patient_id:
            continue
        organ_name = normalize_text(row.get("organ_name")).lower()
        node_name = STAGE9_MOD.STAGE6_ORGAN_TO_NODE.get(organ_name)
        if not node_name:
            continue
        if normalize_text(row.get("status")).lower() != "ok":
            by_patient[node_name] = None
            continue
        token_json = normalize_text(row.get("token_json"))
        if not token_json:
            by_patient[node_name] = None
            continue
        by_patient[node_name] = STAGE9_MOD.parse_token_json(token_json, None, f"stage6:{patient_id}:{node_name}")
    return {patient_id: by_patient}


def choose_model_strategy(model_strategy, has_rna):
    if model_strategy == "phase3":
        return "phase3"
    if model_strategy == "phase4":
        return "phase4"
    return "phase4" if bool(has_rna) else "phase3"


def build_stage9_pack(patient_id, stage6_result, ehr_result, rna_result, roi_result, semantic_result, runtime_root, chosen_strategy):
    stage6_img_by_patient = pick_stage6_node_tokens(patient_id, stage6_result["token_rows"])
    sample_token = None
    for token in stage6_img_by_patient.get(patient_id, {}).values():
        if token is not None:
            sample_token = token
            break
    img_token_dim = 64 if sample_token is None else int(sample_token.shape[0])

    g_rna_by_patient = {}
    t_imm_by_patient = {}
    if rna_result["g_rna"] is not None:
        g_rna_by_patient[patient_id] = np.asarray(rna_result["g_rna"][0], dtype=np.float32)
    if rna_result["t_imm"] is not None:
        t_imm_by_patient[patient_id] = np.asarray(rna_result["t_imm"][0], dtype=np.float32)
    g_ehr_by_patient = {}
    if ehr_result["g_ehr"] is not None:
        g_ehr_by_patient[patient_id] = np.asarray(ehr_result["g_ehr"][0], dtype=np.float32)
    t_tumor_by_patient = {}
    if roi_result["token"] is not None:
        t_tumor_by_patient[patient_id] = np.asarray(roi_result["token"], dtype=np.float32)
    t_sem_by_patient = {}
    if semantic_result["token"] is not None:
        t_sem_by_patient[patient_id] = np.asarray(semantic_result["token"], dtype=np.float32)

    pack, summary_rows = STAGE9_MOD.build_pack(
        patient_ids=[patient_id],
        stage6_img_by_patient=stage6_img_by_patient,
        img_token_dim=img_token_dim,
        g_rna_by_patient=g_rna_by_patient,
        g_rna_dim=128,
        g_ehr_by_patient=g_ehr_by_patient,
        g_ehr_dim=128,
        t_imm_by_patient=t_imm_by_patient,
        t_imm_dim=64,
        t_tumor_by_patient=t_tumor_by_patient,
        t_tumor_dim=64,
        t_sem_by_patient=t_sem_by_patient,
        t_sem_dim=64,
    )
    if chosen_strategy == "phase3":
        pack = PHASE_UTILS.disable_rna_modalities(pack)
        row = dict(summary_rows[0])
        row["has_g_rna"] = 0
        row["has_t_imm"] = 0
        summary_rows = [row]

    stage9_root = ensure_output_dir(Path(runtime_root) / "stage9")
    pack_path = stage9_root / "organ_tokenization_pack.npz"
    manifest_path = stage9_root / "organ_tokenization_manifest.csv"
    summary_path = stage9_root / "organ_tokenization_summary.json"
    np.savez_compressed(pack_path, **pack)
    STAGE9_MOD.write_manifest_csv(manifest_path, summary_rows)
    STAGE9_MOD.write_summary_json(
        summary_path,
        {
            "patient_count": 1,
            "organ_node_count": len(STAGE9_MOD.ORGAN_NODE_NAMES),
            "organ_node_names": list(STAGE9_MOD.ORGAN_NODE_NAMES),
            "model_strategy": chosen_strategy,
            "available_counts": {
                "g_rna": int(summary_rows[0]["has_g_rna"]),
                "t_imm": int(summary_rows[0]["has_t_imm"]),
                "g_ehr": int(summary_rows[0]["has_g_ehr"]),
                "t_tumor": int(summary_rows[0]["has_t_tumor"]),
                "t_sem": int(summary_rows[0]["has_t_sem"]),
                "img_primary": int(summary_rows[0]["has_img_Primary"]),
                "img_lung": int(summary_rows[0]["has_img_Lung"]),
                "img_bone": int(summary_rows[0]["has_img_Bone"]),
                "img_liver": int(summary_rows[0]["has_img_Liver"]),
                "img_lymphnode_mediastinum": int(summary_rows[0]["has_img_LymphNodeMediastinum"]),
                "img_brain": int(summary_rows[0]["has_img_Brain"]),
            },
        },
    )
    return {
        "pack": pack,
        "pack_path": str(pack_path),
        "summary_rows": summary_rows,
        "summary_path": str(summary_path),
    }


def load_json_optional(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_stage10_to_stage11(stage9_pack_path, runtime_root, device):
    stage10_cfg = load_json_optional(DEFAULT_STAGE10_SUMMARY)
    stage11_graph_cfg = load_json_optional(DEFAULT_STAGE11_GRAPH_SUMMARY)
    stage11_reason_cfg = load_json_optional(DEFAULT_STAGE11_REASON_SUMMARY)

    stage10_root = Path(runtime_root) / "stage10"
    stage11_graph_root = Path(runtime_root) / "stage11_graph_construction"
    stage11_reason_root = Path(runtime_root) / "stage11_graph_reasoning"

    STAGE10_MOD.run_stage10_fusion(
        stage9_pack_path=stage9_pack_path,
        stage9_module_path=ROOT / "9.2_organ_query.py",
        output_root=stage10_root,
        batch_size=int(stage10_cfg.get("batch_size", 64)),
        d_model=int(stage10_cfg.get("d_model", 128)),
        num_heads=int(stage10_cfg.get("num_heads", 8)),
        ffn_hidden_dim=int(stage10_cfg.get("ffn_hidden_dim", 256)),
        dropout=float(stage10_cfg.get("dropout", 0.1)),
        device=str(device),
        seed=int(stage10_cfg.get("seed", 1337)),
    )
    STAGE11_GRAPH_MOD.run_stage11_graph_construction(
        stage10_npz_path=stage10_root / "fused_organ_tokens.npz",
        output_root=stage11_graph_root,
        self_logit=float(stage11_graph_cfg.get("self_logit", 3.0)),
        strong_logit=float(stage11_graph_cfg.get("strong_logit", 1.5)),
        weak_logit=float(stage11_graph_cfg.get("weak_logit", -0.5)),
        nonprior_logit=float(stage11_graph_cfg.get("nonprior_logit", -2.0)),
        allow_nonprior_residual=bool(stage11_graph_cfg.get("allow_nonprior_residual", True)),
    )
    STAGE11_REASON_MOD.run_stage11_graph_reasoning(
        stage11_pack_path=stage11_graph_root / "graph_construction_pack.npz",
        output_root=stage11_reason_root,
        d_model=int(stage11_reason_cfg.get("d_model", 128)),
        num_heads=int(stage11_reason_cfg.get("num_heads", 8)),
        num_layers=int(stage11_reason_cfg.get("num_layers", 2)),
        ffn_hidden_dim=int(stage11_reason_cfg.get("ffn_hidden_dim", 256)),
        dropout=float(stage11_reason_cfg.get("dropout", 0.1)),
        explanation_hidden_dim=int(stage11_reason_cfg.get("explanation_hidden_dim", 256)),
        top_k=int(stage11_reason_cfg.get("top_k", 3)),
        max_hops=int(stage11_reason_cfg.get("max_hops", 3)),
        beam_width=int(stage11_reason_cfg.get("beam_width", 8)),
        device=str(device),
        seed=int(stage11_reason_cfg.get("seed", 1337)),
    )
    return {
        "stage10_root": str(stage10_root),
        "stage11_graph_root": str(stage11_graph_root),
        "stage11_reason_root": str(stage11_reason_root),
        "stage11_pack_path": str(stage11_reason_root / "graph_reasoning_pack.npz"),
        "attention_npz": str(stage10_root / "fused_organ_tokens.npz"),
        "attention_summary_json": str(stage10_root / "fusion_summary.json"),
    }


def map_recurrence_location(value, recurrence_classes):
    text = normalize_text(value).lower()
    if not text:
        return None
    for idx, name in enumerate(recurrence_classes):
        label = str(name).strip().lower()
        if text == label:
            return idx
    if "local" in text:
        return recurrence_classes.index("local") if "local" in recurrence_classes else None
    if "regional" in text or "node" in text or "lymph" in text:
        return recurrence_classes.index("regional") if "regional" in recurrence_classes else None
    if "distant" in text or "met" in text:
        return recurrence_classes.index("distant") if "distant" in recurrence_classes else None
    return None


def build_inference_supervision(bundle, recurrence_classes):
    event_os = np.zeros((1,), dtype=np.float32)
    time_os_days = np.zeros((1,), dtype=np.float32)
    os_label_known = np.zeros((1,), dtype=np.uint8)
    event_rec = np.zeros((1,), dtype=np.float32)
    time_rec_days = np.zeros((1,), dtype=np.float32)
    rec_label_known = np.zeros((1,), dtype=np.uint8)
    rec_location_index = np.zeros((1,), dtype=np.int64)
    rec_location_known = np.zeros((1,), dtype=np.uint8)

    labels_row = bundle.get("labels_row") or {}
    if labels_row:
        if normalize_text(labels_row.get("os_label_known")) == "1":
            os_label_known[0] = 1
            time_os_days[0] = float(labels_row.get("time_os_days") or 0.0)
            event_os[0] = float(labels_row.get("event_os") or 0.0)
        if normalize_text(labels_row.get("rec_label_known")) == "1":
            rec_label_known[0] = 1
            time_rec_days[0] = float(labels_row.get("time_rec_days") or 0.0)
            event_rec[0] = float(labels_row.get("event_rec") or 0.0)
        loc_idx = map_recurrence_location(labels_row.get("rec_location_class"), recurrence_classes)
        if loc_idx is not None:
            rec_location_known[0] = 1
            rec_location_index[0] = int(loc_idx)

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


def run_trained_model_inference(patient_id, bundle, stage11_pack_path, runtime_root, chosen_strategy, args, device):
    model_path = Path(args.phase4_model_path if chosen_strategy == "phase4" else args.phase3_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {model_path}")

    stage11_pack = EXPL_TRAIN_MOD.load_stage11_pack(stage11_pack_path)
    payload = safe_torch_load(model_path, map_location="cpu")
    config = payload["config"]

    model = EXPL_TRAIN_MOD.ExplanationGuidedPrimaryModel(
        d_model=int(config["d_model"]),
        num_nodes=int(config["num_nodes"]),
        organ_node_names=list(config["organ_node_names"]),
        recurrence_classes=list(config["recurrence_classes"]),
        edge_feature_dim=int(config["edge_feature_dim"]),
        pool_mode=str(config["pool_mode"]),
        pool_hidden_dim=int(config["pool_hidden_dim"]),
        trunk_hidden_dim=int(config["trunk_hidden_dim"]),
        explanation_hidden_dim=int(config["explanation_hidden_dim"]),
        dropout=float(config["dropout"]),
        survival_mode=str(config["survival_mode"]),
        num_time_bins=int(config["num_time_bins"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    z_prime = torch.from_numpy(stage11_pack["Z_prime"]).float().to(device)
    edge_features = EXPL_TRAIN_MOD.build_edge_feature_tensor(stage11_pack, device=device)
    candidate_edge_mask = torch.from_numpy(stage11_pack["candidate_edge_mask"]).to(device)
    with torch.no_grad():
        outputs = model(
            z_prime=z_prime,
            edge_features=edge_features,
            candidate_edge_mask=candidate_edge_mask,
        )
    outputs_np = EXPL_TRAIN_MOD.extract_numpy_outputs(outputs, survival_mode=str(config["survival_mode"]))
    recurrence_classes = [str(x) for x in config["recurrence_classes"]]
    supervision = build_inference_supervision(bundle, recurrence_classes)
    split_name = np.asarray(["infer"], dtype=object)

    model_root = ensure_output_dir(Path(runtime_root) / "model_inference")
    pred_dir = ensure_output_dir(model_root / "pred")
    graph_summary_json = model_root / "graph_reasoning_summary.json"
    primary_pred_csv = pred_dir / "patient_primary_predictions.csv"
    primary_pack_npz = pred_dir / "primary_output_pack.npz"
    graph_pack_npz = model_root / "graph_reasoning_pack.npz"

    EXPL_TRAIN_MOD.PRIMARY_MOD.write_prediction_csv(
        path=primary_pred_csv,
        patient_ids=np.asarray([patient_id], dtype=object),
        split_name=split_name,
        supervision=supervision,
        recurrence_classes=recurrence_classes,
        survival_mode=str(config["survival_mode"]),
        outputs=outputs_np,
    )
    np.savez_compressed(
        primary_pack_npz,
        patient_ids=np.asarray([patient_id]).astype(str),
        organ_node_names=stage11_pack["organ_node_names"].astype(str),
        split_name=split_name.astype(str),
        recurrence_classes=np.asarray(recurrence_classes).astype(str),
        pool_weights=outputs_np["pool_weights"].astype(np.float32),
        pooled_u=outputs_np["pooled_u"].astype(np.float32),
        base_trunk=outputs_np["base_trunk"].astype(np.float32),
        trunk=outputs_np["trunk"].astype(np.float32),
        os_log_risk=outputs_np["os_log_risk"].astype(np.float32),
        hazard_prob=outputs_np["hazard_prob"].astype(np.float32),
        survival_curve=outputs_np["survival_curve"].astype(np.float32),
        recurrence_probability=outputs_np["rec_prob"].astype(np.float32),
        recurrence_location_probability=outputs_np["rec_location_prob"].astype(np.float32),
        explanation_recurrence_probability=outputs_np["explanation_rec_prob"].astype(np.float32),
        explanation_location_probability=outputs_np["explanation_location_prob"].astype(np.float32),
        event_os=supervision["event_os"].astype(np.float32),
        time_os_days=supervision["time_os_days"].astype(np.float32),
        os_label_known=supervision["os_label_known"].astype(np.uint8),
        event_rec=supervision["event_rec"].astype(np.float32),
        time_rec_days=supervision["time_rec_days"].astype(np.float32),
        rec_label_known=supervision["rec_label_known"].astype(np.uint8),
        rec_location_index=supervision["rec_location_index"].astype(np.int64),
        rec_location_known=supervision["rec_location_known"].astype(np.uint8),
        time_bin_edges=np.asarray(config.get("bin_edges", []), dtype=np.float32),
    )
    np.savez_compressed(
        graph_pack_npz,
        patient_ids=np.asarray([patient_id]).astype(str),
        organ_node_names=stage11_pack["organ_node_names"].astype(str),
        Z_prime=stage11_pack["Z_prime"].astype(np.float32),
        organ_susceptibility=outputs_np["organ_susceptibility"].astype(np.float32),
        edge_diffusion_prob=outputs_np["edge_diffusion_prob"].astype(np.float32),
        edge_type_code=stage11_pack["edge_type_code"].astype(np.uint8),
        prior_edge_mask=stage11_pack["prior_edge_mask"].astype(np.uint8),
        candidate_edge_mask=stage11_pack["candidate_edge_mask"].astype(np.uint8),
        adjacency_logits=stage11_pack["adjacency_logits"].astype(np.float32),
        adjacency_prob=stage11_pack["adjacency_prob"].astype(np.float32),
        split_name=split_name.astype(str),
        explanation_recurrence_probability=outputs_np["explanation_rec_prob"].astype(np.float32),
        explanation_location_probability=outputs_np["explanation_location_prob"].astype(np.float32),
        diffusion_features=outputs_np["diffusion_features"].astype(np.float32),
        susceptibility_context=outputs_np["susceptibility_context"].astype(np.float32),
        edge_context=outputs_np["edge_context"].astype(np.float32),
        primary_out_edge=outputs_np["primary_out_edge"].astype(np.float32),
    )
    write_json(
        graph_summary_json,
        {
            "stage": "15.4_external_case_inference",
            "patient_count": 1,
            "num_nodes": int(stage11_pack["Z_prime"].shape[1]),
            "d_model": int(stage11_pack["Z_prime"].shape[2]),
            "model_strategy": chosen_strategy,
            "checkpoint_path": str(model_path),
            "random_init_only": False,
            "explanation_semantics": EXPLANATION_SEMANTICS,
            "survival_mode": str(config["survival_mode"]),
            "recurrence_classes": recurrence_classes,
            "ranges": {
                "organ_susceptibility_min": float(outputs_np["organ_susceptibility"].min()),
                "organ_susceptibility_max": float(outputs_np["organ_susceptibility"].max()),
                "edge_diffusion_prob_min": float(outputs_np["edge_diffusion_prob"].min()),
                "edge_diffusion_prob_max": float(outputs_np["edge_diffusion_prob"].max()),
            },
        },
    )
    return {
        "primary_pred_csv": str(primary_pred_csv),
        "primary_pack_npz": str(primary_pack_npz),
        "graph_pack_npz": str(graph_pack_npz),
        "graph_summary_json": str(graph_summary_json),
        "model_path": str(model_path),
        "outputs_np": outputs_np,
        "recurrence_classes": recurrence_classes,
        "survival_mode": str(config["survival_mode"]),
    }


def build_explanation_outputs(graph_pack_npz, primary_pack_npz, runtime_root):
    graph_pack = EXPL_OUT_MOD.load_npz(graph_pack_npz)
    primary_pack = EXPL_OUT_MOD.load_npz(primary_pack_npz)
    built = EXPL_OUT_MOD.build_outputs(
        graph_pack=graph_pack,
        primary_pack=primary_pack,
        top_k=3,
        max_hops=3,
        beam_width=8,
        top_edge_k=5,
    )
    output_root = ensure_output_dir(Path(runtime_root) / "explanation_outputs")
    np.savez_compressed(
        output_root / "explanation_pack.npz",
        patient_ids=np.asarray(graph_pack["patient_ids"]).astype(str),
        organ_node_names=np.asarray(graph_pack["organ_node_names"]).astype(str),
        organ_susceptibility=np.asarray(graph_pack["organ_susceptibility"], dtype=np.float32),
        edge_diffusion_prob=np.asarray(graph_pack["edge_diffusion_prob"], dtype=np.float32),
        edge_type_code=np.asarray(graph_pack["edge_type_code"], dtype=np.uint8),
        prior_edge_mask=np.asarray(graph_pack["prior_edge_mask"], dtype=np.uint8),
        recurrence_probability=np.asarray(primary_pack["recurrence_probability"], dtype=np.float32),
        recurrence_location_probability=np.asarray(primary_pack["recurrence_location_probability"], dtype=np.float32),
        recurrence_classes=np.asarray(primary_pack["recurrence_classes"]).astype(str),
        os_log_risk=np.asarray(primary_pack.get("os_log_risk", np.zeros((1,), dtype=np.float32)), dtype=np.float32),
        hazard_prob=np.asarray(primary_pack.get("hazard_prob", np.zeros((1, 0), dtype=np.float32)), dtype=np.float32),
        survival_curve=np.asarray(primary_pack.get("survival_curve", np.zeros((1, 0), dtype=np.float32)), dtype=np.float32),
    )
    write_csv(
        output_root / "organ_susceptibility.csv",
        ["patient_id", "organ_index", "organ_name", "susceptibility"],
        built["organ_rows"],
    )
    write_csv(
        output_root / "edge_diffusion_long.csv",
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
        output_root / "patient_explanation_manifest.csv",
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
    write_json(output_root / "patient_explanations.json", built["patient_explanations"])
    write_json(output_root / "topk_paths.json", built["topk_paths_payload"])
    write_json(
        output_root / "explanation_summary.json",
        {
            "stage": "12.2_explanation_outputs",
            "graph_pack_path": str(graph_pack_npz),
            "graph_summary_path": str(Path(graph_pack_npz).with_name("graph_reasoning_summary.json")),
            "primary_pack_path": str(primary_pack_npz),
            "survival_mode": EXPL_OUT_MOD.infer_survival_mode(primary_pack),
            "graph_reasoning_random_init_only": False,
            "graph_reasoning_explanation_semantics": EXPLANATION_SEMANTICS,
            **built["summary"],
        },
    )
    return {
        "output_root": str(output_root),
        "summary": built["summary"],
    }


def render_visualizations(explanation_root, output_root):
    explanation_root = Path(explanation_root)
    output_root = ensure_output_dir(output_root)
    manifest_rows = VIS_MOD.read_csv_rows(explanation_root / "patient_explanation_manifest.csv")
    patient_explanations = VIS_MOD.load_patient_explanations(explanation_root / "patient_explanations.json")
    organ_rows = VIS_MOD.read_csv_rows(explanation_root / "organ_susceptibility.csv")
    edge_rows = VIS_MOD.read_csv_rows(explanation_root / "edge_diffusion_long.csv")
    susceptibility_by_patient, organ_names = VIS_MOD.group_susceptibility(organ_rows)
    edge_by_patient = VIS_MOD.group_edges(edge_rows)

    patient_records = []
    for manifest_row in manifest_rows:
        patient_id = str(manifest_row["patient_id"])
        patient_records.append(
            VIS_MOD.build_patient_record(
                manifest_row=manifest_row,
                patient_entry=patient_explanations[patient_id],
                patient_edges=edge_by_patient.get(patient_id, []),
                patient_sus=susceptibility_by_patient.get(patient_id, {}),
            )
        )

    selected_rows = VIS_MOD.render_patient_svgs(output_root, organ_names, patient_records)
    cohort_svg = VIS_MOD.render_cohort_svg(output_root, organ_names, patient_records)
    VIS_MOD.render_dashboard(output_root, explanation_root, selected_rows, cohort_svg)
    return {
        "visualization_root": str(output_root),
        "cohort_svg": "" if cohort_svg is None else str(cohort_svg),
    }


def build_bundle_index_html(patient_id, case_input_rel, report_rel):
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8"/>',
            f"  <title>External Inference Bundle: {patient_id}</title>",
            "  <style>",
            "    body {",
            "      margin: 0;",
            "      padding: 32px;",
            "      background: linear-gradient(180deg, #f4efe5 0%, #ede4d2 100%);",
            "      color: #2c2318;",
            '      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;',
            "    }",
            "    .panel {",
            "      background: rgba(255, 253, 248, 0.96);",
            "      border: 1px solid #d7ccb5;",
            "      border-radius: 22px;",
            "      padding: 24px;",
            "      margin-bottom: 24px;",
            "      box-shadow: 0 10px 28px rgba(77, 59, 31, 0.08);",
            "    }",
            "    a {",
            "      color: #20406d;",
            "      text-decoration: none;",
            "    }",
            "    a:hover {",
            "      text-decoration: underline;",
            "    }",
            "  </style>",
            "</head>",
            "<body>",
            '  <div class="panel">',
            f"    <h1>External Inference Bundle: {patient_id}</h1>",
            "    <p>This bundle was generated by on-the-fly external-case preprocessing, multimodal encoding, deterministic Stage 10/11 feature construction, and Stage 13 checkpoint inference.</p>",
            "  </div>",
            '  <div class="panel">',
            "    <h2>Outputs</h2>",
            "    <ul>",
            f'      <li><a href="{case_input_rel}">Case inputs summary</a></li>',
            f'      <li><a href="{report_rel}">System outputs report</a></li>',
            "    </ul>",
            "  </div>",
            "</body>",
            "</html>",
        ]
    )


def main(argv=None, **overrides):
    args = apply_overrides(parse_args(argv), overrides)

    patient_id = normalize_text(args.patient_id)
    if not patient_id:
        raise RuntimeError(
            "patient_id is required. In Python Console, call "
            "run_external_case_inference(patient_id='external-001', ct_path='...', clinical_json='...')."
        )

    output_root = ensure_output_dir(args.output_root)
    case_input_root = ensure_output_dir(Path(output_root) / "case_inputs")
    system_output_root = ensure_output_dir(Path(output_root) / "system_outputs")
    runtime_root = ensure_output_dir(Path(output_root) / "runtime" / patient_id)
    visualization_root = ensure_output_dir(Path(output_root) / "visualization")
    device = choose_device(args.device)

    bundle, case_summary, _case_dir = load_case_bundle_and_write(args, case_input_root)
    if not bool(case_summary.get("ready_for_current_pipeline")):
        raise RuntimeError(
            "case inputs are not ready for the current pipeline; missing required inputs: "
            + ",".join(case_summary.get("missing_required_inputs", []))
        )

    deps = IMAGING_PREP_MOD.check_imaging_dependencies()
    if not deps.get("ready"):
        raise RuntimeError(
            "imaging dependencies missing: " + ",".join(deps.get("missing", []))
        )

    ct_path = normalize_text(bundle["modalities"]["ct"].get("path"))
    tumor_seg_path = normalize_text(bundle["modalities"]["tumor_segmentation"].get("path"))
    aim_path = normalize_text(bundle["modalities"]["aim"].get("path"))
    rna_path = normalize_text(bundle["modalities"]["rna"].get("path"))

    ct_context = prepare_ct_volume(
        patient_id=patient_id,
        ct_path=ct_path,
        runtime_root=runtime_root,
        deps=deps,
    )
    tumor_context = prepare_tumor_mask(
        patient_id=patient_id,
        seg_path=tumor_seg_path,
        ct_context=ct_context,
        runtime_root=runtime_root,
        deps=deps,
    )
    semantic_context = prepare_semantic_token(aim_path)
    roi_context = prepare_roi_token(ct_context=ct_context, tumor_context=tumor_context, deps=deps)
    stage6_result = run_stage6_single_case(
        patient_id=patient_id,
        ct_context=ct_context,
        runtime_root=runtime_root,
        device=device,
        args=args,
    )
    ehr_result = encode_ehr(bundle.get("clinical_row"), runtime_root=runtime_root, device=device)
    rna_result = encode_rna_and_immune(
        patient_id=patient_id,
        rna_path=rna_path,
        runtime_root=runtime_root,
        device=device,
        transform=args.rna_transform,
    )

    chosen_strategy = choose_model_strategy(
        model_strategy=args.model_strategy,
        has_rna=(rna_result["g_rna"] is not None),
    )
    stage9_result = build_stage9_pack(
        patient_id=patient_id,
        stage6_result=stage6_result,
        ehr_result=ehr_result,
        rna_result=rna_result,
        roi_result=roi_context,
        semantic_result=semantic_context,
        runtime_root=runtime_root,
        chosen_strategy=chosen_strategy,
    )
    stage10_11_result = run_stage10_to_stage11(
        stage9_pack_path=stage9_result["pack_path"],
        runtime_root=runtime_root,
        device=device,
    )
    inference_result = run_trained_model_inference(
        patient_id=patient_id,
        bundle=bundle,
        stage11_pack_path=stage10_11_result["stage11_pack_path"],
        runtime_root=runtime_root,
        chosen_strategy=chosen_strategy,
        args=args,
        device=device,
    )
    explanation_result = build_explanation_outputs(
        graph_pack_npz=inference_result["graph_pack_npz"],
        primary_pack_npz=inference_result["primary_pack_npz"],
        runtime_root=runtime_root,
    )
    vis_result = render_visualizations(
        explanation_root=explanation_result["output_root"],
        output_root=visualization_root,
    )

    PHASE_UTILS.run_python_script(
        "15.2_system_outputs.py",
        [
            "--explanation-root",
            explanation_result["output_root"],
            "--primary-predictions-csv",
            inference_result["primary_pred_csv"],
            "--attention-npz",
            stage10_11_result["attention_npz"],
            "--attention-summary-json",
            stage10_11_result["attention_summary_json"],
            "--visualization-root",
            vis_result["visualization_root"],
            "--case-input-root",
            case_input_root,
            "--output-root",
            system_output_root,
            "--patient-ids",
            patient_id,
        ],
    )

    runtime_summary = {
        "patient_id": patient_id,
        "model_strategy": chosen_strategy,
        "checkpoint_path": inference_result["model_path"],
        "device": str(device),
        "ct_source_kind": ct_context["source_kind"],
        "tumor_segmentation_status": tumor_context["status"],
        "semantic_status": semantic_context["status"],
        "roi_status": roi_context["status"],
        "has_rna": 0 if rna_result["g_rna"] is None else 1,
        "rna_gene_coverage": rna_result.get("gene_coverage", {}),
        "stage6_token_csv": stage6_result["token_csv_path"],
        "stage9_pack_path": stage9_result["pack_path"],
        "stage11_pack_path": stage10_11_result["stage11_pack_path"],
        "primary_pack_path": inference_result["primary_pack_npz"],
        "graph_pack_path": inference_result["graph_pack_npz"],
        "explanation_root": explanation_result["output_root"],
        "visualization_root": vis_result["visualization_root"],
    }
    write_json(Path(runtime_root) / "external_inference_summary.json", runtime_summary)

    bundle_summary = {
        "patient_id": patient_id,
        "output_root": str(output_root),
        "case_input_summary": str(Path(case_input_root) / patient_id / "case_input_summary.json"),
        "system_output_report": str(Path(system_output_root) / "cases" / patient_id / "report.html"),
        "system_output_manifest": str(Path(system_output_root) / "system_output_manifest.csv"),
        "runtime_summary": str(Path(runtime_root) / "external_inference_summary.json"),
        "model_strategy": chosen_strategy,
        "checkpoint_path": inference_result["model_path"],
    }
    write_json(Path(output_root) / "bundle_summary.json", bundle_summary)
    case_input_rel = Path("case_inputs") / patient_id / "case_input_summary.json"
    report_rel = Path("system_outputs") / "cases" / patient_id / "report.html"
    (Path(output_root) / "index.html").write_text(
        build_bundle_index_html(patient_id, str(case_input_rel.as_posix()), str(report_rel.as_posix())),
        encoding="utf-8",
    )

    print(f"wrote: {Path(output_root) / 'bundle_summary.json'}")
    print(f"wrote: {Path(output_root) / 'index.html'}")
    print(f"wrote: {Path(runtime_root) / 'external_inference_summary.json'}")
    print(f"model_strategy: {chosen_strategy}")
    print("complete")
    return {
        "patient_id": patient_id,
        "output_root": str(output_root),
        "bundle_summary_json": str(Path(output_root) / "bundle_summary.json"),
        "runtime_summary_json": str(Path(runtime_root) / "external_inference_summary.json"),
        "index_html": str(Path(output_root) / "index.html"),
        "model_strategy": chosen_strategy,
    }


def run_external_case_inference(
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
    disable_internal_lookup=False,
    force_external_inference=False,
    model_strategy="auto",
    rna_transform="raw",
    organ_seg_run_tag="search_base24",
    organ_seg_model_path="",
    allow_legacy_model_fallback=False,
    phase3_model_path=str(DEFAULT_PHASE3_MODEL),
    phase4_model_path=str(DEFAULT_PHASE4_MODEL),
    device="auto",
    output_root=str(DEFAULT_OUTPUT_ROOT),
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
        disable_internal_lookup=disable_internal_lookup,
        force_external_inference=force_external_inference,
        model_strategy=model_strategy,
        rna_transform=rna_transform,
        organ_seg_run_tag=organ_seg_run_tag,
        organ_seg_model_path=organ_seg_model_path,
        allow_legacy_model_fallback=allow_legacy_model_fallback,
        phase3_model_path=phase3_model_path,
        phase4_model_path=phase4_model_path,
        device=device,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
