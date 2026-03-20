import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pydicom
import nibabel as nib
from scipy import ndimage


## still cleaning the dataset, but it's final clean, after this i will do the model part
## main thing i did: Iterate through all patients, determining which ones to process based on the manifest or metadata.csv.

try:
    import pydicom_seg
except ImportError:
    pydicom_seg = None

try:
    import highdicom
except ImportError:
    highdicom = None


SEG_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.66.4"
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0]) if np is not None else None
TUMOR_KEYWORDS = ("tumor", "tumour", "lesion", "gtv", "target", "mass", "nodule", "primary")


def ensure_output_dir(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_float_list(val):
    out = []
    if val is None:
        return out
    try:
        for x in val:
            out.append(float(x))
    except Exception:
        return []
    return out


def normalize_vec(vec):
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return None
    return arr / norm


def safe_dcmread(path, stop_before_pixels):
    try:
        return pydicom.dcmread(str(path), force=True, stop_before_pixels=stop_before_pixels)
    except Exception:
        return None


def scan_case_headers(case_dir):
    headers = []
    for dcm_path in sorted(case_dir.rglob("*.dcm")):
        ds = safe_dcmread(dcm_path, stop_before_pixels=True)
        if ds is None:
            continue
        entry = {
            "path": dcm_path,
            "modality": str(getattr(ds, "Modality", "")).strip(),
            "sop_class_uid": str(getattr(ds, "SOPClassUID", "")).strip(),
            "series_uid": str(getattr(ds, "SeriesInstanceUID", "")).strip(),
            "sop_uid": str(getattr(ds, "SOPInstanceUID", "")).strip(),
            "patient_id": str(getattr(ds, "PatientID", "")).strip(),
        }
        headers.append(entry)
    return headers


def get_seg_candidates(headers):
    out = []
    for h in headers:
        if h["modality"] == "SEG" or h["sop_class_uid"] == SEG_SOP_CLASS_UID:
            out.append(h["path"])
    return sorted(set(out))


def load_seg_dataset(case_dir, seg_file_arg, headers):
    if seg_file_arg:
        seg_path = Path(seg_file_arg)
        if not seg_path.exists():
            raise RuntimeError(f"seg file not found: {seg_path}")
        seg_ds = safe_dcmread(seg_path, stop_before_pixels=False)
        if seg_ds is None:
            raise RuntimeError(f"failed to read seg file: {seg_path}")
        return seg_path, seg_ds

    candidates = get_seg_candidates(headers)
    if not candidates:
        raise RuntimeError(f"no SEG file found under case_dir: {case_dir}")
    if len(candidates) > 1:
        preview = "\n".join(str(x) for x in candidates[:10])
        raise RuntimeError(
            "multiple SEG files found; specify --seg-file explicitly.\n"
            + preview
        )
    seg_path = candidates[0]
    seg_ds = safe_dcmread(seg_path, stop_before_pixels=False)
    if seg_ds is None:
        raise RuntimeError(f"failed to read seg file: {seg_path}")
    return seg_path, seg_ds


def collect_referenced_series_uids(seg_ds):
    out = []

    if hasattr(seg_ds, "ReferencedSeriesSequence"):
        for item in getattr(seg_ds, "ReferencedSeriesSequence", []):
            uid = str(getattr(item, "SeriesInstanceUID", "")).strip()
            if uid:
                out.append(uid)

    if hasattr(seg_ds, "ReferencedStudySequence"):
        for study_item in getattr(seg_ds, "ReferencedStudySequence", []):
            for series_item in getattr(study_item, "ReferencedSeriesSequence", []):
                uid = str(getattr(series_item, "SeriesInstanceUID", "")).strip()
                if uid:
                    out.append(uid)

    return sorted(set(out))


def extract_segment_records(seg_ds):
    out = []
    seg_seq = getattr(seg_ds, "SegmentSequence", [])
    for seg in seg_seq:
        seg_num = int(getattr(seg, "SegmentNumber", 0))
        if seg_num <= 0:
            continue

        texts = [
            str(getattr(seg, "SegmentLabel", "")),
            str(getattr(seg, "SegmentDescription", "")),
        ]
        for attr in ("SegmentedPropertyTypeCodeSequence", "AnatomicRegionSequence"):
            for item in getattr(seg, attr, []):
                texts.append(str(getattr(item, "CodeMeaning", "")))
                texts.append(str(getattr(item, "CodeValue", "")))
        merged = " ".join(t for t in texts if t).strip()
        out.append(
            {
                "segment_number": seg_num,
                "segment_label": str(getattr(seg, "SegmentLabel", "")),
                "segment_description": str(getattr(seg, "SegmentDescription", "")),
                "text": merged,
            }
        )
    return out


def choose_segment_number(seg_ds, segment_number_arg, segment_label_arg):
    segments = extract_segment_records(seg_ds)
    if not segments:
        raise RuntimeError("SEG has empty SegmentSequence")

    by_num = {x["segment_number"]: x for x in segments}
    if segment_number_arg is not None:
        seg_num = int(segment_number_arg)
        if seg_num not in by_num:
            raise RuntimeError(
                f"segment_number={seg_num} not found; available={sorted(by_num.keys())}"
            )
        return by_num[seg_num]

    if segment_label_arg:
        target = segment_label_arg.strip().lower()
        matched = []
        for seg in segments:
            label = (seg["segment_label"] or "").strip().lower()
            desc = (seg["segment_description"] or "").strip().lower()
            full = (seg["text"] or "").strip().lower()
            if target == label or target == desc or target in full:
                matched.append(seg)
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            nums = sorted(x["segment_number"] for x in matched)
            raise RuntimeError(
                f"segment label '{segment_label_arg}' matched multiple segments: {nums}; use --segment-number."
            )
        raise RuntimeError(
            f"segment label '{segment_label_arg}' not found; use --segment-number."
        )

    keyword_hits = []
    for seg in segments:
        text = (seg["text"] or "").lower()
        if any(k in text for k in TUMOR_KEYWORDS):
            keyword_hits.append(seg)
    if len(keyword_hits) == 1:
        return keyword_hits[0]
    if len(keyword_hits) > 1:
        nums = sorted(x["segment_number"] for x in keyword_hits)
        raise RuntimeError(
            "multiple keyword-matched tumor segments found: "
            + f"{nums}. use --segment-number to disambiguate."
        )

    available = [(x["segment_number"], x["segment_label"]) for x in segments]
    raise RuntimeError(
        "no tumor-like segment found by keyword. "
        + "strict mode requires explicit --segment-number or --segment-label. "
        + f"available={available}"
    )


def get_ct_series_map(headers):
    series_map = {}
    for h in headers:
        if h["modality"] != "CT":
            continue
        uid = h["series_uid"]
        if not uid:
            continue
        if uid not in series_map:
            series_map[uid] = []
        series_map[uid].append(h["path"])
    for uid in series_map:
        series_map[uid] = sorted(series_map[uid])
    return series_map


def choose_ct_series_uid(series_map, referenced_series_uids, ct_series_uid_arg):
    if not series_map:
        raise RuntimeError("no CT DICOM found in case_dir")

    if ct_series_uid_arg:
        if ct_series_uid_arg not in series_map:
            keys = sorted(series_map.keys())
            raise RuntimeError(
                f"--ct-series-uid not found: {ct_series_uid_arg}; available={keys}"
            )
        return ct_series_uid_arg

    hit = [uid for uid in referenced_series_uids if uid in series_map]
    if len(hit) == 1:
        return hit[0]
    if len(hit) > 1:
        raise RuntimeError(
            "SEG references multiple CT series in this case; specify --ct-series-uid."
        )

    if len(series_map) == 1:
        return list(series_map.keys())[0]

    preview = [(uid, len(paths)) for uid, paths in series_map.items()]
    raise RuntimeError(
        "multiple CT series found and SEG reference not resolvable; "
        + "specify --ct-series-uid. "
        + f"candidates={preview}"
    )


def get_first_valid_iop(slices):
    for ds in slices:
        iop = parse_float_list(getattr(ds, "ImageOrientationPatient", None))
        if len(iop) >= 6:
            return iop[:6]
    return None


def get_slice_sort_info(ds, slice_dir):
    ipp = parse_float_list(getattr(ds, "ImagePositionPatient", None))
    ipp_tuple = None
    scalar = None
    if len(ipp) >= 3:
        ipp_tuple = (float(ipp[0]), float(ipp[1]), float(ipp[2]))
        scalar = float(np.dot(np.asarray(ipp_tuple, dtype=np.float64), slice_dir))

    instance = getattr(ds, "InstanceNumber", None)
    try:
        instance = float(instance) if instance is not None else None
    except Exception:
        instance = None
    return scalar, instance, ipp_tuple


def compute_slice_spacing(sorted_scalars, fallback_thickness):
    if len(sorted_scalars) > 1:
        diffs = []
        for i in range(1, len(sorted_scalars)):
            d = abs(float(sorted_scalars[i]) - float(sorted_scalars[i - 1]))
            if d > 1e-6:
                diffs.append(d)
        if diffs:
            return float(np.median(np.asarray(diffs, dtype=np.float64)))
    if fallback_thickness is not None and float(fallback_thickness) > 0:
        return float(fallback_thickness)
    return 1.0


def load_ct_geometry(series_dcm_paths):
    slices = []
    for p in series_dcm_paths:
        ds = safe_dcmread(p, stop_before_pixels=False)
        if ds is None:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        slices.append(ds)
    if not slices:
        raise RuntimeError("no readable CT slices with PixelData")

    iop = get_first_valid_iop(slices)
    if iop is None:
        row_dir = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        col_dir = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        row_dir = normalize_vec(iop[:3])
        col_dir = normalize_vec(iop[3:6])
        if row_dir is None or col_dir is None:
            row_dir = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            col_dir = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

    slice_dir = normalize_vec(np.cross(row_dir, col_dir))
    if slice_dir is None:
        slice_dir = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    enriched = []
    for ds in slices:
        scalar, instance, ipp_tuple = get_slice_sort_info(ds, slice_dir)
        enriched.append(
            {
                "ds": ds,
                "scalar": scalar,
                "instance": instance,
                "ipp": ipp_tuple,
                "sop_uid": str(getattr(ds, "SOPInstanceUID", "")).strip(),
            }
        )

    def sort_key(item):
        if item["scalar"] is not None:
            return (0, float(item["scalar"]))
        if item["instance"] is not None:
            return (1, float(item["instance"]))
        return (2, 0.0)

    enriched.sort(key=sort_key)
    sorted_ds = [x["ds"] for x in enriched]
    sorted_scalars = [x["scalar"] for x in enriched if x["scalar"] is not None]

    first = sorted_ds[0]
    pixel_spacing = parse_float_list(getattr(first, "PixelSpacing", None))
    if len(pixel_spacing) < 2:
        row_spacing, col_spacing = 1.0, 1.0
    else:
        row_spacing, col_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])
    thickness = getattr(first, "SliceThickness", None)
    try:
        thickness = float(thickness) if thickness is not None else None
    except Exception:
        thickness = None
    slice_spacing = compute_slice_spacing(sorted_scalars, thickness)

    hu_slices = []
    for ds in sorted_ds:
        px = np.asarray(ds.pixel_array, dtype=np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu_slices.append(px * slope + intercept)
    ct_volume_zyx = np.stack(hu_slices, axis=0).astype(np.float32)

    origin_ipp = None
    for x in enriched:
        if x["ipp"] is not None:
            origin_ipp = np.asarray(x["ipp"], dtype=np.float64)
            break
    if origin_ipp is None:
        origin_ipp = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    affine_lps_yxz = np.eye(4, dtype=np.float64)
    affine_lps_yxz[:3, 0] = row_dir * row_spacing
    affine_lps_yxz[:3, 1] = col_dir * col_spacing
    affine_lps_yxz[:3, 2] = slice_dir * slice_spacing
    affine_lps_yxz[:3, 3] = origin_ipp
    affine_ras_yxz = LPS_TO_RAS.dot(affine_lps_yxz)

    sop_uid_to_index = {}
    slice_scalars = []
    for idx, item in enumerate(enriched):
        sop_uid = item["sop_uid"]
        if sop_uid:
            sop_uid_to_index[sop_uid] = idx
        if item["scalar"] is None:
            slice_scalars.append(float(idx))
        else:
            slice_scalars.append(float(item["scalar"]))

    return {
        "ct_volume_zyx": ct_volume_zyx,
        "ct_shape_zyx": tuple(int(x) for x in ct_volume_zyx.shape),
        "spacing_zyx": [float(slice_spacing), float(row_spacing), float(col_spacing)],
        "origin_lps": [float(x) for x in origin_ipp.tolist()],
        "direction_lps": [
            [float(x) for x in row_dir.tolist()],
            [float(x) for x in col_dir.tolist()],
            [float(x) for x in slice_dir.tolist()],
        ],
        "affine_ras_yxz": affine_ras_yxz,
        "sop_uid_to_index": sop_uid_to_index,
        "slice_scalars": slice_scalars,
        "rows": int(ct_volume_zyx.shape[1]),
        "cols": int(ct_volume_zyx.shape[2]),
        "depth": int(ct_volume_zyx.shape[0]),
        "slice_dir": slice_dir,
        "scalar_tolerance": max(float(slice_spacing) * 0.6, 1e-3),
        "ct_series_uid": str(getattr(first, "SeriesInstanceUID", "")).strip(),
        "patient_id": str(getattr(first, "PatientID", "")).strip(),
        "sop_uids_sorted": [str(getattr(ds, "SOPInstanceUID", "")).strip() for ds in sorted_ds],
    }


def frame_segment_number(frame_fg):
    seq = getattr(frame_fg, "SegmentIdentificationSequence", [])
    if not seq:
        return None
    try:
        num = int(getattr(seq[0], "ReferencedSegmentNumber", 0))
        return num if num > 0 else None
    except Exception:
        return None


def frame_referenced_sop_uid(frame_fg):
    deriv = getattr(frame_fg, "DerivationImageSequence", [])
    for d in deriv:
        for src in getattr(d, "SourceImageSequence", []):
            uid = str(getattr(src, "ReferencedSOPInstanceUID", "")).strip()
            if uid:
                return uid
    return ""


def frame_image_position(frame_fg):
    seq = getattr(frame_fg, "PlanePositionSequence", [])
    if not seq:
        return None
    ipp = parse_float_list(getattr(seq[0], "ImagePositionPatient", None))
    if len(ipp) < 3:
        return None
    return (float(ipp[0]), float(ipp[1]), float(ipp[2]))


def resize_2d_nearest(mask_hw, target_h, target_w):
    if int(mask_hw.shape[0]) == int(target_h) and int(mask_hw.shape[1]) == int(target_w):
        return (mask_hw > 0).astype(np.uint8)
    zoom_h = float(target_h) / float(mask_hw.shape[0])
    zoom_w = float(target_w) / float(mask_hw.shape[1])
    out = ndimage.zoom(mask_hw.astype(np.uint8), zoom=(zoom_h, zoom_w), order=0)
    out = (out > 0).astype(np.uint8)
    if out.shape[0] != int(target_h) or out.shape[1] != int(target_w):
        fixed = np.zeros((int(target_h), int(target_w)), dtype=np.uint8)
        h = min(int(target_h), out.shape[0])
        w = min(int(target_w), out.shape[1])
        fixed[:h, :w] = out[:h, :w]
        out = fixed
    return out


def resample_3d_nearest(mask_zyx, target_shape_zyx):
    if tuple(mask_zyx.shape) == tuple(target_shape_zyx):
        return (mask_zyx > 0).astype(np.uint8)
    zf = float(target_shape_zyx[0]) / float(mask_zyx.shape[0])
    yf = float(target_shape_zyx[1]) / float(mask_zyx.shape[1])
    xf = float(target_shape_zyx[2]) / float(mask_zyx.shape[2])
    out = ndimage.zoom(mask_zyx.astype(np.uint8), zoom=(zf, yf, xf), order=0)
    out = (out > 0).astype(np.uint8)
    if tuple(out.shape) != tuple(target_shape_zyx):
        fixed = np.zeros(target_shape_zyx, dtype=np.uint8)
        z = min(target_shape_zyx[0], out.shape[0])
        y = min(target_shape_zyx[1], out.shape[1])
        x = min(target_shape_zyx[2], out.shape[2])
        fixed[:z, :y, :x] = out[:z, :y, :x]
        out = fixed
    return out


def decode_seg_with_pydicom_seg(seg_ds, segment_number):
    if pydicom_seg is None:
        return None

    try:
        if hasattr(pydicom_seg, "SegmentReader"):
            reader = pydicom_seg.SegmentReader()
            result = reader.read(seg_ds)
            if hasattr(result, "segment_data"):
                arr = np.asarray(result.segment_data(segment_number))
                if arr.ndim == 3:
                    return (arr > 0).astype(np.uint8)
        if hasattr(pydicom_seg, "MultiClassReader"):
            reader = pydicom_seg.MultiClassReader()
            result = reader.read(seg_ds)
            data = np.asarray(getattr(result, "data", None))
            if data.ndim == 3:
                if data.max() <= 1 and segment_number == 1:
                    return (data > 0).astype(np.uint8)
                return (data == int(segment_number)).astype(np.uint8)
    except Exception:
        return None
    return None


def decode_seg_with_highdicom(seg_ds, segment_number, ct_info):
    if highdicom is None:
        return None

    try:
        seg_obj = None
        if hasattr(highdicom, "seg") and hasattr(highdicom.seg, "Segmentation"):
            seg_obj = highdicom.seg.Segmentation.from_dataset(seg_ds)
        if seg_obj is None:
            return None
        if hasattr(seg_obj, "get_pixels_by_source_instance"):
            arr = seg_obj.get_pixels_by_source_instance(
                source_sop_instance_uids=ct_info["sop_uids_sorted"],
                segment_numbers=[int(segment_number)],
                combine_segments=True,
            )
            arr = np.asarray(arr)
            if arr.ndim == 3:
                return (arr > 0).astype(np.uint8)
    except Exception:
        return None
    return None


def decode_seg_with_pydicom_manual(seg_ds, segment_number, ct_info):
    arr = np.asarray(seg_ds.pixel_array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 4:
        arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
    if arr.ndim != 3:
        raise RuntimeError(f"unsupported SEG pixel_array ndim: {arr.ndim}")

    per_frame = getattr(seg_ds, "PerFrameFunctionalGroupsSequence", [])
    if per_frame and len(per_frame) != arr.shape[0]:
        raise RuntimeError(
            f"PerFrameFunctionalGroupsSequence length mismatch: {len(per_frame)} vs {arr.shape[0]}"
        )

    ct_shape = ct_info["ct_shape_zyx"]
    mask = np.zeros(ct_shape, dtype=np.uint8)
    target_h = int(ct_info["rows"])
    target_w = int(ct_info["cols"])
    sop_uid_to_index = ct_info["sop_uid_to_index"]
    slice_scalars = ct_info["slice_scalars"]
    slice_dir = np.asarray(ct_info["slice_dir"], dtype=np.float64)
    tol = float(ct_info["scalar_tolerance"])

    align_mode = "geometry_exact"
    painted = 0
    for i in range(arr.shape[0]):
        frame = arr[i]
        if float(frame.max()) <= 0.0:
            continue

        if per_frame:
            frame_fg = per_frame[i]
            frame_seg_num = frame_segment_number(frame_fg)
            if frame_seg_num is not None and int(frame_seg_num) != int(segment_number):
                continue
            if frame_seg_num is None and int(segment_number) != 1:
                raise RuntimeError(
                    f"frame {i} missing ReferencedSegmentNumber for non-1 target segment."
                )
        elif int(segment_number) != 1:
            raise RuntimeError(
                "SEG without PerFrameFunctionalGroupsSequence supports only segment_number=1 in strict fallback."
            )

        target_index = None
        if per_frame:
            uid = frame_referenced_sop_uid(per_frame[i])
            if uid and uid in sop_uid_to_index:
                target_index = int(sop_uid_to_index[uid])
                align_mode = "referenced_sop_uid"
            else:
                pos = frame_image_position(per_frame[i])
                if pos is not None and slice_scalars:
                    frame_scalar = float(np.dot(np.asarray(pos, dtype=np.float64), slice_dir))
                    diffs = [abs(float(s) - frame_scalar) for s in slice_scalars]
                    best_idx = int(np.argmin(np.asarray(diffs, dtype=np.float64)))
                    if diffs[best_idx] <= tol:
                        target_index = best_idx
        else:
            if arr.shape[0] == ct_shape[0]:
                target_index = int(i)
                align_mode = "frame_index"

        if target_index is None or target_index < 0 or target_index >= ct_shape[0]:
            raise RuntimeError(f"frame {i} cannot map to CT slices in strict mode")

        frame_bin = (frame > 0).astype(np.uint8)
        if frame_bin.shape[0] != target_h or frame_bin.shape[1] != target_w:
            frame_bin = resize_2d_nearest(frame_bin, target_h, target_w)
            align_mode = "resampled_xy"

        before = int(mask[target_index].sum())
        mask[target_index] = np.maximum(mask[target_index], frame_bin)
        after = int(mask[target_index].sum())
        painted += max(0, after - before)

    if painted <= 0 and int(mask.sum()) <= 0:
        raise RuntimeError("decoded SEG mask is empty for selected segment")
    return mask, align_mode


def decode_tumor_mask(seg_ds, segment_number, ct_info, force_resample):
    target_shape = ct_info["ct_shape_zyx"]

    mask = decode_seg_with_pydicom_seg(seg_ds, segment_number)
    if mask is not None:
        align_mode = "decoder_native"
        if tuple(mask.shape) != tuple(target_shape):
            mask = resample_3d_nearest(mask, target_shape)
            align_mode = "resampled_3d"
        return (mask > 0).astype(np.uint8), "pydicom_seg", align_mode

    mask = decode_seg_with_highdicom(seg_ds, segment_number, ct_info)
    if mask is not None:
        align_mode = "decoder_native"
        if tuple(mask.shape) != tuple(target_shape):
            mask = resample_3d_nearest(mask, target_shape)
            align_mode = "resampled_3d"
        return (mask > 0).astype(np.uint8), "highdicom", align_mode

    mask, align_mode = decode_seg_with_pydicom_manual(seg_ds, segment_number, ct_info)
    if tuple(mask.shape) != tuple(target_shape):
        if not force_resample:
            raise RuntimeError(
                f"manual decoded mask shape {mask.shape} != CT shape {target_shape}; use --force-resample"
            )
        mask = resample_3d_nearest(mask, target_shape)
        align_mode = "resampled_3d"
    return (mask > 0).astype(np.uint8), "pydicom_manual", align_mode


def save_mask_nifti(mask_zyx, affine_ras_yxz, output_path):
    mask_yxz = np.transpose(mask_zyx.astype(np.uint8), (1, 2, 0))
    nii = nib.Nifti1Image(mask_yxz, affine_ras_yxz)
    nii.set_data_dtype(np.uint8)
    nib.save(nii, str(output_path))
    return output_path


def build_metadata_dict(
    subject_id,
    series_instance_uid,
    spacing_zyx,
    affine_ras_yxz,
    segment_record,
    voxel_count,
    ct_info,
    seg_ds,
    decode_backend,
    align_mode,
):
    meta = {
        "SubjectID": subject_id,
        "SeriesInstanceUID": series_instance_uid,
        "spacing": [float(x) for x in spacing_zyx],
        "affine": [[float(v) for v in row] for row in affine_ras_yxz.tolist()],
        "segment_number": int(segment_record["segment_number"]),
        "voxel_count": int(voxel_count),
        "segment_label": segment_record.get("segment_label", ""),
        "segment_description": segment_record.get("segment_description", ""),
        "shape_zyx": [int(x) for x in ct_info["ct_shape_zyx"]],
        "origin_lps": [float(x) for x in ct_info["origin_lps"]],
        "direction_lps": ct_info["direction_lps"],
        "decode_backend": decode_backend,
        "align_mode": align_mode,
        "seg_series_uid": str(getattr(seg_ds, "SeriesInstanceUID", "")).strip(),
        "seg_sop_instance_uid": str(getattr(seg_ds, "SOPInstanceUID", "")).strip(),
        "coordinate_system": "RAS",
    }
    return meta


def write_metadata_json(metadata, output_path):
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return output_path


def run_stage5(case_dir, output_dir, seg_file, ct_series_uid, segment_number, segment_label, subject_id, force_resample, verbose):
    headers = scan_case_headers(case_dir)
    if not headers:
        raise RuntimeError(f"no .dcm file found in case_dir: {case_dir}")

    seg_path, seg_ds = load_seg_dataset(case_dir, seg_file, headers)
    segment_record = choose_segment_number(seg_ds, segment_number, segment_label)
    referenced_uids = collect_referenced_series_uids(seg_ds)
    series_map = get_ct_series_map(headers)
    chosen_ct_uid = choose_ct_series_uid(series_map, referenced_uids, ct_series_uid)
    ct_info = load_ct_geometry(series_map[chosen_ct_uid])

    mask_zyx, decode_backend, align_mode = decode_tumor_mask(
        seg_ds=seg_ds,
        segment_number=segment_record["segment_number"],
        ct_info=ct_info,
        force_resample=force_resample,
    )
    voxel_count = int(mask_zyx.sum())
    if voxel_count <= 0:
        raise RuntimeError("mask_tumor is empty after decode/alignment")

    subject = subject_id.strip()
    if not subject:
        subject = ct_info.get("patient_id", "").strip()
    if not subject:
        subject = str(Path(case_dir).name)

    mask_path = Path(output_dir) / "mask_tumor.nii.gz"
    meta_path = Path(output_dir) / "metadata.json"
    save_mask_nifti(mask_zyx, ct_info["affine_ras_yxz"], mask_path)
    metadata = build_metadata_dict(
        subject_id=subject,
        series_instance_uid=ct_info["ct_series_uid"],
        spacing_zyx=ct_info["spacing_zyx"],
        affine_ras_yxz=ct_info["affine_ras_yxz"],
        segment_record=segment_record,
        voxel_count=voxel_count,
        ct_info=ct_info,
        seg_ds=seg_ds,
        decode_backend=decode_backend,
        align_mode=align_mode,
    )
    write_metadata_json(metadata, meta_path)

    if verbose:
        print(f"case_dir: {case_dir}")
        print(f"seg_file: {seg_path}")
        print(f"segment_number: {segment_record['segment_number']}")
        print(f"segment_label: {segment_record.get('segment_label', '')}")
        print(f"ct_series_uid: {ct_info['ct_series_uid']}")
        print(f"ct_shape_zyx: {ct_info['ct_shape_zyx']}")
        print(f"spacing_zyx: {ct_info['spacing_zyx']}")
        print(f"decode_backend: {decode_backend}")
        print(f"align_mode: {align_mode}")
        print(f"voxel_count: {voxel_count}")

    return {
        "mask_path": str(mask_path),
        "metadata_path": str(meta_path),
        "voxel_count": voxel_count,
        "segment_number": int(segment_record["segment_number"]),
        "ct_series_uid": ct_info["ct_series_uid"],
        "decode_backend": decode_backend,
        "align_mode": align_mode,
    }


DEFAULT_MANIFEST_CSV = Path("output/patient_manifest.csv")
DEFAULT_METADATA_CSV = Path("data/manifest-1622561851074/metadata.csv")
DEFAULT_CASE_ROOT = Path("data/manifest-1622561851074/NSCLC Radiogenomics")
DEFAULT_OUTPUT_ROOT = Path("output/stage5")


def read_segment_map(segment_map_csv):
    out = {}
    if not segment_map_csv:
        return out
    path = Path(segment_map_csv)
    if not path.exists():
        raise RuntimeError(f"segment_map_csv not found: {path}")

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("patient_id") or "").strip()
            if not pid:
                continue
            seg_num_raw = (row.get("segment_number") or "").strip()
            seg_label = (row.get("segment_label") or "").strip()
            seg_num = None
            if seg_num_raw:
                try:
                    seg_num = int(seg_num_raw)
                except Exception:
                    raise RuntimeError(
                        f"invalid segment_number for patient_id={pid}: {seg_num_raw}"
                    )
            out[pid] = {"segment_number": seg_num, "segment_label": seg_label}
    return out


def load_patient_ids_from_manifest(manifest_csv, require_seg):
    path = Path(manifest_csv)
    if not path.exists():
        return []

    out = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("patient_id") or "").strip()
            if not pid:
                continue
            has_ct = (row.get("has_ct") or "").strip() == "1"
            has_seg = (row.get("has_seg") or "").strip() == "1"
            if not has_ct:
                continue
            if require_seg and not has_seg:
                continue
            out.append(pid)
    return sorted(set(out))


def load_patient_ids_from_metadata(metadata_csv, require_seg):
    path = Path(metadata_csv)
    if not path.exists():
        raise RuntimeError(f"metadata_csv not found: {path}")

    modality_by_pid = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("Collection") or "").strip() != "NSCLC Radiogenomics":
                continue
            pid = (row.get("Subject ID") or "").strip()
            if not pid:
                continue
            mod = (row.get("Modality") or "").strip()
            if pid not in modality_by_pid:
                modality_by_pid[pid] = set()
            modality_by_pid[pid].add(mod)

    out = []
    for pid, mods in modality_by_pid.items():
        if "CT" not in mods:
            continue
        if require_seg and "SEG" not in mods:
            continue
        out.append(pid)
    return sorted(set(out))


def resolve_patient_ids(manifest_csv, metadata_csv, require_seg):
    from_manifest = load_patient_ids_from_manifest(manifest_csv, require_seg)
    if from_manifest:
        return from_manifest, "manifest"
    from_meta = load_patient_ids_from_metadata(metadata_csv, require_seg)
    return from_meta, "metadata"


def ensure_output_root(output_root):
    Path(output_root).mkdir(parents=True, exist_ok=True)


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Stage 5 tumor mask provider over NSCLC Radiogenomics cases.",
        allow_abbrev=False,
    )
    parser.add_argument("--manifest-csv", type=str, default=str(DEFAULT_MANIFEST_CSV))
    parser.add_argument("--metadata-csv", type=str, default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--case-root", type=str, default=str(DEFAULT_CASE_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--segment-map-csv", type=str, default="", help="Optional CSV: patient_id,segment_number,segment_label")
    parser.add_argument("--segment-number", type=int, default=1, help="Default segment number for all cases.")
    parser.add_argument("--segment-label", type=str, default="", help="Default segment label when number is not provided.")
    parser.add_argument("--subject-id-from-patient", action="store_true", help="Use patient_id as SubjectID in metadata.")
    parser.add_argument("--ct-series-uid", type=str, default="", help="Optional fixed CT series UID for all cases (rarely needed).")
    parser.add_argument("--require-seg", action="store_true", help="Only process cases with CT+SEG in manifest/metadata.")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="0 means process all cases; >0 means process first N cases.",
    )
    parser.add_argument("--force-resample", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"ignore_unknown_args: {unknown}")
    return args


def choose_segment_for_patient(patient_id, segment_map, default_segment_number, default_segment_label):
    if patient_id in segment_map:
        item = segment_map[patient_id]
        return item.get("segment_number"), item.get("segment_label", ""), "map_csv"
    return default_segment_number, default_segment_label, "default"


def run_batch():
    args = parse_args()

    if args.max_cases < 0:
        raise SystemExit("--max-cases must be >= 0 (0 means all cases)")

    case_root = Path(args.case_root)
    output_root = Path(args.output_root)
    ensure_output_root(output_root)

    if not case_root.exists():
        raise SystemExit(f"case_root not found: {case_root}")

    segment_map = read_segment_map(args.segment_map_csv.strip())
    patient_ids, source_name = resolve_patient_ids(
        manifest_csv=args.manifest_csv,
        metadata_csv=args.metadata_csv,
        require_seg=bool(args.require_seg),
    )
    if not patient_ids:
        raise SystemExit("no patient found from manifest/metadata under current filters")

    total_cases = len(patient_ids)
    if args.max_cases > 0:
        patient_ids = patient_ids[: min(args.max_cases, total_cases)]

    print(f"[start] patient_source={source_name}")
    print(
        f"[start] selected_cases={len(patient_ids)} total_cases={total_cases} "
        f"full_cases={1 if args.max_cases == 0 else 0}"
    )
    print(f"[start] output_root={output_root}")
    print(f"[start] default_segment_number={args.segment_number}")
    print(f"[start] require_seg={bool(args.require_seg)}")

    summary_rows = []
    ok_count = 0
    fail_count = 0
    miss_case_dir_count = 0

    total = len(patient_ids)
    for i, patient_id in enumerate(patient_ids, start=1):
        case_dir = case_root / patient_id
        patient_out = output_root / patient_id
        seg_num, seg_label, seg_source = choose_segment_for_patient(
            patient_id=patient_id,
            segment_map=segment_map,
            default_segment_number=args.segment_number,
            default_segment_label=args.segment_label.strip(),
        )

        row = {
            "patient_id": patient_id,
            "case_dir": str(case_dir),
            "output_dir": str(patient_out),
            "segment_number": "" if seg_num is None else int(seg_num),
            "segment_label": seg_label,
            "segment_source": seg_source,
            "status": "",
            "error": "",
            "mask_path": "",
            "metadata_path": "",
            "voxel_count": 0,
            "ct_series_uid": "",
            "decode_backend": "",
            "align_mode": "",
        }
        print(f"[batch] {i}/{total} patient_id={patient_id}")

        if not case_dir.exists():
            row["status"] = "case_dir_not_found"
            row["error"] = "case directory missing"
            summary_rows.append(row)
            fail_count += 1
            miss_case_dir_count += 1
            continue

        try:
            patient_out.mkdir(parents=True, exist_ok=True)
            result = run_stage5(
                case_dir=case_dir,
                output_dir=patient_out,
                seg_file="",
                ct_series_uid=args.ct_series_uid.strip(),
                segment_number=seg_num,
                segment_label=seg_label,
                subject_id=patient_id if args.subject_id_from_patient else "",
                force_resample=bool(args.force_resample),
                verbose=bool(args.verbose),
            )
            row["status"] = "ok"
            row["mask_path"] = result["mask_path"]
            row["metadata_path"] = result["metadata_path"]
            row["voxel_count"] = int(result["voxel_count"])
            row["ct_series_uid"] = result["ct_series_uid"]
            row["decode_backend"] = result["decode_backend"]
            row["align_mode"] = result["align_mode"]
            ok_count += 1
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            fail_count += 1

        summary_rows.append(row)

    summary_csv = output_root / "stage5_batch_summary.csv"
    write_csv(
        summary_csv,
        [
            "patient_id",
            "case_dir",
            "output_dir",
            "segment_number",
            "segment_label",
            "segment_source",
            "status",
            "error",
            "mask_path",
            "metadata_path",
            "voxel_count",
            "ct_series_uid",
            "decode_backend",
            "align_mode",
        ],
        summary_rows,
    )

    print(f"wrote: {summary_csv}")
    print(f"processed_cases: {len(patient_ids)}")
    print(f"ok_cases: {ok_count}")
    print(f"failed_cases: {fail_count}")
    print(f"case_dir_not_found: {miss_case_dir_count}")
    print("complete")


if __name__ == "__main__":
    run_batch()
