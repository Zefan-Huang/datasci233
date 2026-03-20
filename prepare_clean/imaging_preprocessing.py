import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pydicom as pydicom
import scipy.ndimage as ndimage

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
PATIENT_MANIFEST_CSV = OUTPUT_DIR / "patient_manifest.csv"
LEGACY_PATIENT_MANIFEST_CSV = DATA_DIR / "patient_manifest.csv"
METADATA_CSV = DATA_DIR / "manifest-1622561851074" / "metadata.csv"
RADIOGENOMICS_ROOT = DATA_DIR / "manifest-1622561851074"
AIM_DIR = DATA_DIR / "AIM_files_updated-11-10-2020"

OUTPUT_ROOT = OUTPUT_DIR / "preprocessed"
OUTPUT_CT_DIR = OUTPUT_ROOT / "ct_norm"
OUTPUT_SEG_DIR = OUTPUT_ROOT / "seg_masks"
OUTPUT_SUMMARY_CSV = OUTPUT_ROOT / "imaging_preprocess_summary.csv"
OUTPUT_SEMANTIC_CSV = OUTPUT_ROOT / "semantic_tokens.csv"
OUTPUT_ROI_CSV = OUTPUT_ROOT / "roi_tokens.csv"

TARGET_SPACING = (1.5, 1.5, 1.5)
HU_CLIP = (-1000.0, 400.0)
SEMANTIC_TOKEN_DIM = 64
ROI_TOKEN_DIM = 64
ROI_MARGIN = 4

def ensure_output_dirs():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_CT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SEG_DIR.mkdir(parents=True, exist_ok=True)

def check_imaging_dependencies():

    return {"ready": True, "missing": [], "np": np, "pydicom": pydicom, "ndimage": ndimage}

def load_patient_ids():

    manifest_path = PATIENT_MANIFEST_CSV if PATIENT_MANIFEST_CSV.exists() else LEGACY_PATIENT_MANIFEST_CSV
    patient_ids = []
    with manifest_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_ids.append(row["patient_id"].strip())
    return patient_ids

def load_metadata_rows():

    rows = []
    with METADATA_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def build_series_index(metadata_rows):

    index = {}
    for row in metadata_rows:
        if row.get("Collection") != "NSCLC Radiogenomics":
            continue
        pid = row.get("Subject ID", "").strip()
        if not pid:
            continue
        if pid not in index:
            index[pid] = []
        index[pid].append(row)
    return index

def score_ct_series(series_description, num_images):

    desc = (series_description or "").lower()
    score = 0
    if "chest" in desc or "thorax" in desc or "lung" in desc:
        score += 10
    if "thin" in desc or "1.25" in desc or ".625" in desc:
        score += 3
    if "wo" in desc or "w/o" in desc:
        score += 1
    if "coronal" in desc or "sagittal" in desc or "mip" in desc:
        score -= 8
    if "scout" in desc or "localizer" in desc:
        score -= 20
    score += min(int(num_images / 100), 5)
    return score

def resolve_series_dir(file_location):

    rel = (file_location or "").strip()
    if rel.startswith("./"):
        rel = rel[2:]
    return RADIOGENOMICS_ROOT / rel

def pick_primary_ct_series(patient_id, series_index):

    candidates = []
    for row in series_index.get(patient_id, []):
        if (row.get("Modality") or "").strip() != "CT":
            continue
        desc = row.get("Series Description") or ""
        lower_desc = desc.lower()
        if "segmentation result" in lower_desc or "epad generated dso" in lower_desc:
            continue
        num_images = int(float(row.get("Number of Images") or 0))
        candidates.append((score_ct_series(desc, num_images), num_images, row))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]

def pick_seg_series(patient_id, series_index):

    seg_rows = []
    for row in series_index.get(patient_id, []):
        if (row.get("Modality") or "").strip() == "SEG":
            num_images = int(float(row.get("Number of Images") or 0))
            seg_rows.append((num_images, row))
    if not seg_rows:
        return None
    seg_rows.sort(key=lambda x: x[0], reverse=True)
    return seg_rows[0][1]

def find_aim_file(patient_id):

    direct = AIM_DIR / f"{patient_id}.xml"
    if direct.exists():
        return str(direct)
    candidates = sorted(AIM_DIR.glob(f"{patient_id}*.xml"))
    if candidates:
        return str(candidates[0])
    return ""

def normalize_vector(vec, np):

    arr = np.asarray(vec, dtype="float64")
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return None
    return arr / norm

def extract_segment_numbers(seg_ds):

    keywords = ("tumor", "tumour", "lesion", "gtv", "target", "mass", "nodule", "primary")
    all_segments = set()
    tumor_segments = set()
    seg_seq = getattr(seg_ds, "SegmentSequence", [])
    for seg in seg_seq:
        seg_num = int(getattr(seg, "SegmentNumber", 0))
        if seg_num <= 0:
            continue
        all_segments.add(seg_num)
        text_parts = [
            str(getattr(seg, "SegmentLabel", "")),
            str(getattr(seg, "SegmentDescription", "")),
        ]
        for attr in ("SegmentedPropertyTypeCodeSequence", "AnatomicRegionSequence"):
            code_seq = getattr(seg, attr, [])
            for item in code_seq:
                text_parts.append(str(getattr(item, "CodeMeaning", "")))
                text_parts.append(str(getattr(item, "CodeValue", "")))
        merged_text = " ".join(text_parts).lower()
        if any(key in merged_text for key in keywords):
            tumor_segments.add(seg_num)

    if tumor_segments:
        return tumor_segments, all_segments
    return all_segments, all_segments

def frame_segment_number(frame_fg):

    seg_ident = getattr(frame_fg, "SegmentIdentificationSequence", [])
    if not seg_ident:
        return None
    return int(getattr(seg_ident[0], "ReferencedSegmentNumber", 0)) or None

def frame_referenced_sop_uid(frame_fg):

    deriv = getattr(frame_fg, "DerivationImageSequence", [])
    for deriv_item in deriv:
        source_seq = getattr(deriv_item, "SourceImageSequence", [])
        for source_item in source_seq:
            sop_uid = str(getattr(source_item, "ReferencedSOPInstanceUID", "")).strip()
            if sop_uid:
                return sop_uid
    return ""

def frame_image_position(frame_fg):

    plane_pos = getattr(frame_fg, "PlanePositionSequence", [])
    if not plane_pos:
        return None
    ipp = getattr(plane_pos[0], "ImagePositionPatient", None)
    if ipp is None or len(ipp) < 3:
        return None
    try:
        return (float(ipp[0]), float(ipp[1]), float(ipp[2]))
    except Exception:
        return None

def load_ct_volume_and_normalize(ct_series_dir, deps):

    if not deps["ready"]:
        return {"status": "blocked_missing_dependency", "error": ",".join(deps["missing"])}

    np = deps["np"]
    pydicom = deps["pydicom"]
    ndimage = deps["ndimage"]

    dcm_files = sorted(ct_series_dir.glob("*.dcm"))
    if not dcm_files:
        return {"status": "ct_not_found", "error": "no dcm file"}

    slices = []
    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(str(fp), force=True)
            if hasattr(ds, "PixelData"):
                slices.append(ds)
        except Exception:
            continue
    if not slices:
        return {"status": "ct_not_found", "error": "no readable slice"}

    first_with_iop = None
    for ds in slices:
        iop = getattr(ds, "ImageOrientationPatient", None)
        if iop is not None and len(iop) >= 6:
            first_with_iop = ds
            break
    normal = None
    if first_with_iop is not None:
        row_dir = [float(first_with_iop.ImageOrientationPatient[i]) for i in range(3)]
        col_dir = [float(first_with_iop.ImageOrientationPatient[i]) for i in range(3, 6)]
        normal = normalize_vector(np.cross(row_dir, col_dir), np)

    ct_meta = []
    for ds in slices:
        ipp = None
        ipp_raw = getattr(ds, "ImagePositionPatient", None)
        if ipp_raw is not None and len(ipp_raw) >= 3:
            try:
                ipp = (float(ipp_raw[0]), float(ipp_raw[1]), float(ipp_raw[2]))
            except Exception:
                ipp = None
        instance = getattr(ds, "InstanceNumber", None)
        try:
            instance = float(instance) if instance is not None else None
        except Exception:
            instance = None
        sop_uid = str(getattr(ds, "SOPInstanceUID", "")).strip()
        scalar = None
        if ipp is not None and normal is not None:
            scalar = float(np.dot(np.asarray(ipp, dtype="float64"), normal))
        elif ipp is not None:
            scalar = float(ipp[2])
        ct_meta.append(
            {
                "ds": ds,
                "ipp": ipp,
                "instance": instance,
                "sop_uid": sop_uid,
                "scalar": scalar,
            }
        )

    def slice_sort_key(item):
        if item["scalar"] is not None:
            return (0, float(item["scalar"]))
        if item["instance"] is not None:
            return (1, float(item["instance"]))
        return (2, 0.0)

    ct_meta.sort(key=slice_sort_key)
    slices = [m["ds"] for m in ct_meta]
    hu_slices = []
    for ds in slices:
        px = ds.pixel_array.astype("float32")
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu_slices.append(px * slope + intercept)

    volume = np.stack(hu_slices, axis=0)
    clipped_native = np.clip(volume, HU_CLIP[0], HU_CLIP[1])
    normalized_native = (clipped_native - HU_CLIP[0]) / (HU_CLIP[1] - HU_CLIP[0])
    normalized_native = normalized_native.astype("float32")

    first = slices[0]
    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])

    scalar_values = [m["scalar"] for m in ct_meta if m["scalar"] is not None]
    if len(scalar_values) > 1:
        sorted_scalars = sorted(scalar_values)
        diffs = [abs(sorted_scalars[i] - sorted_scalars[i - 1]) for i in range(1, len(sorted_scalars))]
        diffs = [d for d in diffs if d > 1e-5]
        if diffs:
            slice_spacing = float(np.median(np.asarray(diffs, dtype="float64")))
        else:
            slice_spacing = float(getattr(first, "SliceThickness", 1.0))
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))

    current_spacing = (slice_spacing, row_spacing, col_spacing)
    zoom_factors = (
        current_spacing[0] / TARGET_SPACING[0],
        current_spacing[1] / TARGET_SPACING[1],
        current_spacing[2] / TARGET_SPACING[2],
    )
    resampled = ndimage.zoom(volume, zoom=zoom_factors, order=1)
    clipped = np.clip(resampled, HU_CLIP[0], HU_CLIP[1])
    normalized = (clipped - HU_CLIP[0]) / (HU_CLIP[1] - HU_CLIP[0])
    normalized = normalized.astype("float32")

    sop_uid_to_index = {}
    slice_scalars = []
    for idx, meta in enumerate(ct_meta):
        if meta["sop_uid"]:
            sop_uid_to_index[meta["sop_uid"]] = idx
        if meta["scalar"] is None:
            slice_scalars.append(float(idx))
        else:
            slice_scalars.append(float(meta["scalar"]))

    scalar_tolerance = max(float(slice_spacing) * 0.6, 1e-3)
    ct_geometry = {
        "normal": None if normal is None else [float(x) for x in normal.tolist()],
        "slice_scalars": slice_scalars,
        "sop_uid_to_index": sop_uid_to_index,
        "rows": int(getattr(first, "Rows", volume.shape[1])),
        "cols": int(getattr(first, "Columns", volume.shape[2])),
        "scalar_tolerance": float(scalar_tolerance),
    }

    return {
        "status": "ok",
        "volume_original": volume,
        "volume_norm_native": normalized_native,
        "volume_norm": normalized,
        "spacing_original": current_spacing,
        "spacing_target": TARGET_SPACING,
        "ct_geometry": ct_geometry,
    }

def load_tumor_mask(seg_series_dir, ct_shape, ct_geometry, deps):

    if not deps["ready"]:
        return {"status": "blocked_missing_dependency", "error": ",".join(deps["missing"])}
    if seg_series_dir is None:
        return {"status": "seg_not_found", "error": "no seg series"}

    np = deps["np"]
    pydicom = deps["pydicom"]
    seg_files = sorted(seg_series_dir.glob("*.dcm"))
    if not seg_files:
        return {"status": "seg_not_found", "error": "no seg dcm file"}

    ct_sop_to_index = ct_geometry.get("sop_uid_to_index", {})
    ct_scalars = ct_geometry.get("slice_scalars", [])
    ct_rows = int(ct_geometry.get("rows", ct_shape[1]))
    ct_cols = int(ct_geometry.get("cols", ct_shape[2]))
    ct_normal = ct_geometry.get("normal")
    scalar_tolerance = float(ct_geometry.get("scalar_tolerance", 1.0))
    if ct_normal is not None:
        ct_normal = np.asarray(ct_normal, dtype="float64")

    mask = np.zeros(ct_shape, dtype="uint8")
    seg_dcm_count = 0
    seg_instance_count = 0
    seg_frame_count = 0
    source_shape = None
    seen_uid = set()

    for seg_fp in seg_files:
        try:
            seg_ds = pydicom.dcmread(str(seg_fp), force=True)
        except Exception:
            continue
        if not hasattr(seg_ds, "PixelData"):
            continue
        seg_dcm_count += 1
        sop_uid = str(getattr(seg_ds, "SOPInstanceUID", "")).strip()
        if sop_uid and sop_uid in seen_uid:
            continue
        if sop_uid:
            seen_uid.add(sop_uid)
        seg_instance_count += 1

        try:
            arr = np.asarray(seg_ds.pixel_array)
        except Exception as exc:
            return {
                "status": "seg_read_failed",
                "error": str(exc),
                "align_mode": "failed",
                "fail_reason": "pixel_array_decode_failed",
                "seg_dcm_count": seg_dcm_count,
                "seg_instance_count": seg_instance_count,
                "seg_frame_count": seg_frame_count,
            }

        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 4:
            arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
        if arr.ndim != 3:
            return {
                "status": "seg_geometry_mismatch",
                "error": f"unsupported seg ndim={arr.ndim}",
                "align_mode": "failed",
                "fail_reason": "unsupported_seg_ndim",
                "seg_dcm_count": seg_dcm_count,
                "seg_instance_count": seg_instance_count,
                "seg_frame_count": seg_frame_count,
            }

        if source_shape is None:
            source_shape = tuple(arr.shape)
        if arr.shape[1] != ct_rows or arr.shape[2] != ct_cols:
            return {
                "status": "seg_geometry_mismatch",
                "error": f"frame_hw={arr.shape[1:]} != ct_hw={(ct_rows, ct_cols)}",
                "align_mode": "failed",
                "fail_reason": "frame_size_mismatch",
                "source_shape": source_shape,
                "seg_dcm_count": seg_dcm_count,
                "seg_instance_count": seg_instance_count,
                "seg_frame_count": seg_frame_count,
            }

        selected_segments, _all_segments = extract_segment_numbers(seg_ds)
        per_frame = getattr(seg_ds, "PerFrameFunctionalGroupsSequence", [])
        if per_frame and len(per_frame) != arr.shape[0]:
            return {
                "status": "seg_geometry_mismatch",
                "error": f"per_frame_len={len(per_frame)} arr_frames={arr.shape[0]}",
                "align_mode": "failed",
                "fail_reason": "per_frame_length_mismatch",
                "source_shape": source_shape,
                "seg_dcm_count": seg_dcm_count,
                "seg_instance_count": seg_instance_count,
                "seg_frame_count": seg_frame_count,
            }

        for frame_idx in range(arr.shape[0]):
            frame = arr[frame_idx]
            seg_frame_count += 1
            if float(frame.max()) <= 0.0:
                continue

            if per_frame:
                frame_fg = per_frame[frame_idx]
                frame_seg = frame_segment_number(frame_fg)
                if selected_segments and frame_seg is not None and frame_seg not in selected_segments:
                    continue

                target_index = None
                ref_uid = frame_referenced_sop_uid(frame_fg)
                if ref_uid and ref_uid in ct_sop_to_index:
                    target_index = int(ct_sop_to_index[ref_uid])
                else:
                    pos = frame_image_position(frame_fg)
                    if pos is not None:
                        if ct_normal is not None:
                            frame_scalar = float(np.dot(np.asarray(pos, dtype="float64"), ct_normal))
                        else:
                            frame_scalar = float(pos[2])
                        if ct_scalars:
                            diffs = [abs(float(s) - frame_scalar) for s in ct_scalars]
                            best_idx = int(np.argmin(np.asarray(diffs)))
                            if diffs[best_idx] <= scalar_tolerance:
                                target_index = best_idx

                if target_index is None or target_index < 0 or target_index >= ct_shape[0]:
                    return {
                        "status": "seg_geometry_mismatch",
                        "error": f"frame={frame_idx} could not map to ct slice",
                        "align_mode": "failed",
                        "fail_reason": "frame_to_ct_mapping_failed",
                        "source_shape": source_shape,
                        "seg_dcm_count": seg_dcm_count,
                        "seg_instance_count": seg_instance_count,
                        "seg_frame_count": seg_frame_count,
                    }
            else:
                return {
                    "status": "seg_geometry_mismatch",
                    "error": "missing PerFrameFunctionalGroupsSequence",
                    "align_mode": "failed",
                    "fail_reason": "missing_per_frame_groups",
                    "source_shape": source_shape,
                    "seg_dcm_count": seg_dcm_count,
                    "seg_instance_count": seg_instance_count,
                    "seg_frame_count": seg_frame_count,
                }

            mask[target_index] = np.maximum(mask[target_index], (frame > 0).astype("uint8"))

    if seg_instance_count == 0:
        return {
            "status": "seg_not_found",
            "error": "no readable seg object",
            "align_mode": "failed",
            "fail_reason": "no_readable_seg_object",
            "seg_dcm_count": seg_dcm_count,
            "seg_instance_count": seg_instance_count,
            "seg_frame_count": seg_frame_count,
        }
    if int(mask.sum()) == 0:
        return {
            "status": "seg_empty",
            "error": "mask is empty",
            "align_mode": "geometry_exact",
            "fail_reason": "empty_after_filtering",
            "source_shape": source_shape,
            "seg_dcm_count": seg_dcm_count,
            "seg_instance_count": seg_instance_count,
            "seg_frame_count": seg_frame_count,
        }
    return {
        "status": "ok",
        "mask": mask,
        "align_mode": "geometry_exact",
        "fail_reason": "",
        "source_shape": source_shape,
        "seg_dcm_count": seg_dcm_count,
        "seg_instance_count": seg_instance_count,
        "seg_frame_count": seg_frame_count,
    }

def compute_roi_token(volume_norm, tumor_mask, deps):

    if not deps["ready"]:
        return {"status": "blocked_missing_dependency", "error": ",".join(deps["missing"])}

    np = deps["np"]
    indices = np.where(tumor_mask > 0)
    if len(indices[0]) == 0:
        return {"status": "roi_empty", "error": "mask is empty"}

    zmin, zmax = int(indices[0].min()), int(indices[0].max())
    ymin, ymax = int(indices[1].min()), int(indices[1].max())
    xmin, xmax = int(indices[2].min()), int(indices[2].max())

    zmin = max(0, zmin - ROI_MARGIN)
    ymin = max(0, ymin - ROI_MARGIN)
    xmin = max(0, xmin - ROI_MARGIN)
    zmax = min(volume_norm.shape[0] - 1, zmax + ROI_MARGIN)
    ymax = min(volume_norm.shape[1] - 1, ymax + ROI_MARGIN)
    xmax = min(volume_norm.shape[2] - 1, xmax + ROI_MARGIN)

    roi = volume_norm[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1]
    roi_mask = tumor_mask[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1] > 0
    roi_vals = roi[roi_mask]
    if roi_vals.size == 0:
        return {"status": "roi_empty", "error": "no roi voxel"}

    features = [
        float(roi_vals.mean()),
        float(roi_vals.std()),
        float(roi_vals.min()),
        float(roi_vals.max()),
        float(np.percentile(roi_vals, 25)),
        float(np.percentile(roi_vals, 50)),
        float(np.percentile(roi_vals, 75)),
        float(roi_vals.size),
        float((roi_vals > 0.5).mean()),
        float((zmax - zmin + 1) / volume_norm.shape[0]),
        float((ymax - ymin + 1) / volume_norm.shape[1]),
        float((xmax - xmin + 1) / volume_norm.shape[2]),
        float((zmin + zmax) / 2.0 / max(volume_norm.shape[0] - 1, 1)),
        float((ymin + ymax) / 2.0 / max(volume_norm.shape[1] - 1, 1)),
        float((xmin + xmax) / 2.0 / max(volume_norm.shape[2] - 1, 1)),
    ]
    feat = np.asarray(features, dtype="float32")
    rng = np.random.RandomState(2026)
    proj = rng.normal(loc=0.0, scale=0.1, size=(ROI_TOKEN_DIM, feat.shape[0])).astype("float32")
    token = np.tanh(proj.dot(feat))
    norm = float(np.linalg.norm(token))
    if norm > 0:
        token = token / norm
    return {"status": "ok", "token": token.tolist()}

def parse_aim_feature_texts(aim_xml_path):

    if not aim_xml_path:
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
    vec = [0.0 for _ in range(token_dim)]
    for text in feature_texts:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % token_dim
        sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
        vec[idx] += sign

        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        for num_text in nums[:2]:
            try:
                value = float(num_text)
            except Exception:
                continue
            idx_num = int(digest[10:18], 16) % token_dim
            vec[idx_num] += max(min(value, 10.0), -10.0) * 0.01

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec

def save_ct_npz(patient_id, ct_result, deps):

    np = deps["np"]
    out_path = OUTPUT_CT_DIR / f"{patient_id}.npz"
    np.savez_compressed(
        out_path,
        ct_volume=ct_result["volume_norm"],
        spacing_target=np.asarray(ct_result["spacing_target"], dtype="float32"),
    )
    return str(out_path)

def save_seg_npz(patient_id, mask, deps):

    np = deps["np"]
    out_path = OUTPUT_SEG_DIR / f"{patient_id}.npz"
    np.savez_compressed(out_path, mask=mask.astype("uint8"))
    return str(out_path)

def process_patient(patient_id, series_index, deps):

    ct_row = pick_primary_ct_series(patient_id, series_index)
    seg_row = pick_seg_series(patient_id, series_index)
    aim_xml = find_aim_file(patient_id)

    summary = {
        "patient_id": patient_id,
        "dependencies_ready": 1 if deps["ready"] else 0,
        "missing_dependencies": ",".join(deps["missing"]),
        "ct_series_dir": "",
        "ct_status": "",
        "ct_norm_npz": "",
        "ct_shape_norm": "",
        "seg_series_dir": "",
        "seg_status": "",
        "seg_mask_npz": "",
        "seg_align_mode": "",
        "seg_fail_reason": "",
        "seg_source_shape": "",
        "seg_dcm_count": 0,
        "seg_instance_count": 0,
        "seg_frame_count": 0,
        "roi_token_status": "",
        "roi_token_dim": 0,
        "semantic_status": "",
        "semantic_token_dim": 0,
        "aim_xml_path": aim_xml,
    }

    semantic_row = {"patient_id": patient_id, "token_json": ""}
    roi_row = {"patient_id": patient_id, "token_json": ""}

    feature_texts = parse_aim_feature_texts(aim_xml)
    semantic_token = build_semantic_token(feature_texts, SEMANTIC_TOKEN_DIM)
    if semantic_token:
        semantic_row["token_json"] = json.dumps(semantic_token)
        summary["semantic_status"] = "ok"
        summary["semantic_token_dim"] = len(semantic_token)
    else:
        summary["semantic_status"] = "aim_missing_or_empty"
        summary["semantic_token_dim"] = 0

    ct_result = None
    if ct_row is None:
        summary["ct_status"] = "ct_series_not_found"
    else:
        ct_dir = resolve_series_dir(ct_row.get("File Location"))
        summary["ct_series_dir"] = str(ct_dir)
        ct_result = load_ct_volume_and_normalize(ct_dir, deps)
        summary["ct_status"] = ct_result["status"]
        if ct_result["status"] == "ok":
            summary["ct_shape_norm"] = "x".join(str(x) for x in ct_result["volume_norm"].shape)
            summary["ct_norm_npz"] = save_ct_npz(patient_id, ct_result, deps)

    seg_result = None
    if seg_row is None:
        summary["seg_status"] = "seg_series_not_found"
        summary["seg_align_mode"] = "failed"
        summary["seg_fail_reason"] = "seg_series_not_found"
    else:
        seg_dir = resolve_series_dir(seg_row.get("File Location"))
        summary["seg_series_dir"] = str(seg_dir)
        if ct_result is None or ct_result.get("status") != "ok":
            summary["seg_status"] = "blocked_ct_unavailable"
            summary["seg_align_mode"] = "failed"
            summary["seg_fail_reason"] = "ct_unavailable"
        else:
            seg_result = load_tumor_mask(
                seg_dir,
                ct_result["volume_original"].shape,
                ct_result.get("ct_geometry", {}),
                deps,
            )
            summary["seg_status"] = seg_result["status"]
            summary["seg_align_mode"] = seg_result.get("align_mode", "")
            summary["seg_fail_reason"] = seg_result.get("fail_reason", "")
            summary["seg_dcm_count"] = int(seg_result.get("seg_dcm_count", 0))
            summary["seg_instance_count"] = int(seg_result.get("seg_instance_count", 0))
            summary["seg_frame_count"] = int(seg_result.get("seg_frame_count", 0))
            if seg_result["status"] == "ok":
                summary["seg_mask_npz"] = save_seg_npz(patient_id, seg_result["mask"], deps)
                source_shape = seg_result.get("source_shape")
                if source_shape:
                    summary["seg_source_shape"] = "x".join(str(v) for v in source_shape)

    if ct_result is None or ct_result.get("status") != "ok":
        summary["roi_token_status"] = "blocked_ct_unavailable"
    elif seg_result is None or seg_result.get("status") != "ok":
        summary["roi_token_status"] = "blocked_seg_unavailable"
    else:

        roi_result = compute_roi_token(ct_result["volume_norm_native"], seg_result["mask"], deps)
        summary["roi_token_status"] = roi_result["status"]
        if roi_result["status"] == "ok":
            roi_row["token_json"] = json.dumps(roi_result["token"])
            summary["roi_token_dim"] = len(roi_result["token"])

    return {"summary": summary, "semantic_row": semantic_row, "roi_row": roi_row}

def write_csv(path, fieldnames, rows):

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def run_pipeline(max_cases):

    ensure_output_dirs()
    deps = check_imaging_dependencies()
    patient_ids = load_patient_ids()
    metadata_rows = load_metadata_rows()
    series_index = build_series_index(metadata_rows)

    if max_cases is None:
        max_cases = 0
    if max_cases < 0:
        raise SystemExit("--max-cases must be >= 0 (0 means all patients)")

    total_cases = len(patient_ids)
    if max_cases > 0:
        patient_ids = patient_ids[: min(max_cases, total_cases)]
    print(
        f"[start] selected_cases={len(patient_ids)} total_cases={total_cases} "
        f"full_cases={1 if max_cases == 0 else 0}"
    )

    summary_rows = []
    semantic_rows = []
    roi_rows = []

    for pid in patient_ids:
        result = process_patient(pid, series_index, deps)
        summary_rows.append(result["summary"])
        semantic_rows.append(result["semantic_row"])
        roi_rows.append(result["roi_row"])
        if len(summary_rows) % 20 == 0:
            print(f"progress: {len(summary_rows)}/{len(patient_ids)}")

    summary_fields = [
        "patient_id",
        "dependencies_ready",
        "missing_dependencies",
        "ct_series_dir",
        "ct_status",
        "ct_norm_npz",
        "ct_shape_norm",
        "seg_series_dir",
        "seg_status",
        "seg_mask_npz",
        "seg_align_mode",
        "seg_fail_reason",
        "seg_source_shape",
        "seg_dcm_count",
        "seg_instance_count",
        "seg_frame_count",
        "roi_token_status",
        "roi_token_dim",
        "semantic_status",
        "semantic_token_dim",
        "aim_xml_path",
    ]
    token_fields = ["patient_id", "token_json"]

    write_csv(OUTPUT_SUMMARY_CSV, summary_fields, summary_rows)
    write_csv(OUTPUT_SEMANTIC_CSV, token_fields, semantic_rows)
    write_csv(OUTPUT_ROI_CSV, token_fields, roi_rows)

    ct_ok = sum(1 for r in summary_rows if r["ct_status"] == "ok")
    seg_ok = sum(1 for r in summary_rows if r["seg_status"] == "ok")
    roi_ok = sum(1 for r in summary_rows if r["roi_token_status"] == "ok")
    sem_ok = sum(1 for r in summary_rows if r["semantic_status"] == "ok")
    print(f"wrote: {OUTPUT_SUMMARY_CSV}")
    print(f"wrote: {OUTPUT_SEMANTIC_CSV}")
    print(f"wrote: {OUTPUT_ROI_CSV}")
    print(f"processed_cases: {len(summary_rows)}")
    print(f"ct_ok: {ct_ok}")
    print(f"seg_ok: {seg_ok}")
    print(f"roi_token_ok: {roi_ok}")
    print(f"semantic_token_ok: {sem_ok}")
    if not deps["ready"]:
        print(f"warning_missing_dependencies: {','.join(deps['missing'])}")

def parse_args():

    parser = argparse.ArgumentParser(description="Run imaging preprocessing pipeline (5.1~5.4).")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="0 means process all patients; >0 means process first N patients.",
    )
    return parser.parse_args()

def main():

    args = parse_args()
    run_pipeline(args.max_cases)

if __name__ == "__main__":
    main()
