"""

作用：
- 5.1 CT normalization：选主 CT 序列、读取体数据、重采样、HU 裁剪归一化（依赖 numpy/pydicom/scipy）。
- 5.2 Tumor segmentation：读取 DICOM-SEG 并生成肿瘤 mask（依赖 numpy/pydicom）。
- 5.3 Tumor ROI token：从肿瘤 ROI 计算 token（依赖 numpy）。
- 5.4 Semantic annotation token：解析 AIM XML 并生成语义 token（仅标准库可运行）。

输入：
- output/patient_manifest.csv
- data/manifest-1622561851074/metadata.csv
- data/AIM_files_updated-11-10-2020/*.xml

输出：
- output/preprocessed/imaging_preprocess_summary.csv
- output/preprocessed/semantic_tokens.csv
- output/preprocessed/roi_tokens.csv
- output/preprocessed/ct_norm/*.npz（依赖满足时）
- output/preprocessed/seg_masks/*.npz（依赖满足时）
"""
import numpy as np
import pydicom as pydicom
import scipy.ndimage as ndimage
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET


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
    """Why: 预处理会产出多个文件，需要提前保证输出目录存在。

    Content: 创建 preprocessed 根目录和子目录。
    Input: 无（使用模块常量目录）。
    Output: 目录创建完成（已存在则不报错）。
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_CT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SEG_DIR.mkdir(parents=True, exist_ok=True)


def check_imaging_dependencies():
    """Why: CT/SEG/ROI 步骤依赖第三方库，需先判断当前环境是否可跑。

    Content: 检查 numpy、pydicom、scipy.ndimage 是否可导入。
    Input: 无。
    Output: 字典，包含 ready 标记、缺失库、已导入模块对象。
    """
    deps = {"ready": False, "missing": [], "np": None, "pydicom": None, "ndimage": None}
    try:
        import numpy as np  # noqa: WPS433
    except Exception:
        deps["missing"].append("numpy")
        np = None
    try:
        import pydicom  # noqa: WPS433
    except Exception:
        deps["missing"].append("pydicom")
        pydicom = None
    try:
        from scipy import ndimage  # noqa: WPS433
    except Exception:
        deps["missing"].append("scipy")
        ndimage = None

    deps["np"] = np
    deps["pydicom"] = pydicom
    deps["ndimage"] = ndimage
    deps["ready"] = len(deps["missing"]) == 0
    return deps


def load_patient_ids():
    """Why: 预处理应按项目病人清单逐例执行，而不是扫描全盘。

    Content: 从 patient_manifest.csv 读取 patient_id 列。
    Input: PATIENT_MANIFEST_CSV。
    Output: patient_id 列表（按文件顺序）。
    """
    manifest_path = PATIENT_MANIFEST_CSV if PATIENT_MANIFEST_CSV.exists() else LEGACY_PATIENT_MANIFEST_CSV
    patient_ids = []
    with manifest_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_ids.append(row["patient_id"].strip())
    return patient_ids


def load_metadata_rows():
    """Why: 影像序列选择依赖 metadata.csv 里的模态、描述和路径信息。

    Content: 读取 metadata.csv 全部行到内存。
    Input: METADATA_CSV。
    Output: metadata 行字典列表。
    """
    rows = []
    with METADATA_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_series_index(metadata_rows):
    """Why: 需要按病人快速查找 CT/SEG/AIM 对应序列。

    Content: 将 metadata 行按 Subject ID 建索引。
    Input: metadata 行列表。
    Output: patient_id -> 系列行列表 的字典。
    """
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
    """Why: 每个病人通常有多个 CT 序列，需要可复用的主序列打分规则。

    Content: 根据描述关键词和切片数计算启发式分数。
    Input: series_description（字符串），num_images（整数）。
    Output: 分数（越高越优先）。
    """
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
    """Why: metadata 里的路径是相对路径，后续读取文件需要绝对路径。

    Content: 将 metadata File Location 转为本地目录路径。
    Input: file_location（metadata 中的路径字符串）。
    Output: Path 对象（序列目录）。
    """
    rel = (file_location or "").strip()
    if rel.startswith("./"):
        rel = rel[2:]
    return RADIOGENOMICS_ROOT / rel


def pick_primary_ct_series(patient_id, series_index):
    """Why: 5.1 需要先确定一个主 CT 序列作为标准输入。

    Content: 从该病人的 CT 系列中按启发式打分挑选最高分。
    Input: patient_id，series_index。
    Output: 最佳 CT 系列行；若不存在返回 None。
    """
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
    """Why: 5.2 需要定位肿瘤分割序列（优先 DICOM-SEG）。

    Content: 在该病人序列中寻找 Modality=SEG 的系列。
    Input: patient_id，series_index。
    Output: SEG 系列行；若不存在返回 None。
    """
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
    """Why: 5.4 语义 token 需要 AIM XML 文件路径。

    Content: 在 AIM 目录中按 patient_id 匹配 xml（支持 v1 后缀）。
    Input: patient_id。
    Output: AIM 文件路径字符串；若不存在返回空字符串。
    """
    direct = AIM_DIR / f"{patient_id}.xml"
    if direct.exists():
        return str(direct)
    candidates = sorted(AIM_DIR.glob(f"{patient_id}*.xml"))
    if candidates:
        return str(candidates[0])
    return ""


def load_ct_volume_and_normalize(ct_series_dir, deps):
    """Why: 5.1 需要生成归一化 CT 体数据用于后续分割/特征提取。

    Content: 读取 DICOM 序列，转换 HU，重采样到目标 spacing，裁剪并归一化。
    Input: ct_series_dir（CT 目录），deps（依赖模块字典）。
    Output: 包含状态、体数据、spacing、错误信息的字典。
    """
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

    def slice_sort_key(ds):
        if hasattr(ds, "ImagePositionPatient"):
            try:
                return float(ds.ImagePositionPatient[2])
            except Exception:
                pass
        if hasattr(ds, "InstanceNumber"):
            try:
                return float(ds.InstanceNumber)
            except Exception:
                pass
        return 0.0

    slices.sort(key=slice_sort_key)
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

    if len(slices) > 1 and hasattr(slices[0], "ImagePositionPatient") and hasattr(slices[1], "ImagePositionPatient"):
        z0 = float(slices[0].ImagePositionPatient[2])
        z1 = float(slices[1].ImagePositionPatient[2])
        slice_spacing = abs(z1 - z0) if abs(z1 - z0) > 0 else float(getattr(first, "SliceThickness", 1.0))
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

    return {
        "status": "ok",
        "volume_original": volume,
        "volume_norm_native": normalized_native,
        "volume_norm": normalized,
        "spacing_original": current_spacing,
        "spacing_target": TARGET_SPACING,
    }


def load_tumor_mask(seg_series_dir, ct_shape, deps):
    """Why: 5.2 需要得到和 CT 对齐的肿瘤二值 mask。

    Content: 读取 DICOM-SEG 像素，做基础形状对齐检查并生成二值 mask。
    Input: seg_series_dir（SEG 目录），ct_shape（CT 体形状），deps。
    Output: 包含状态、mask、错误信息的字典。
    """
    if not deps["ready"]:
        return {"status": "blocked_missing_dependency", "error": ",".join(deps["missing"])}
    if seg_series_dir is None:
        return {"status": "seg_not_found", "error": "no seg series"}

    np = deps["np"]
    pydicom = deps["pydicom"]
    ndimage = deps["ndimage"]
    seg_files = sorted(seg_series_dir.glob("*.dcm"))
    if not seg_files:
        return {"status": "seg_not_found", "error": "no seg dcm file"}

    try:
        seg_ds = pydicom.dcmread(str(seg_files[0]), force=True)
        arr = seg_ds.pixel_array
    except Exception as exc:
        return {"status": "seg_read_failed", "error": str(exc)}

    if arr.ndim == 4:
        # Some SEG files store multiple segment channels; merge as one tumor mask.
        arr = arr.max(axis=0)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        return {"status": "seg_shape_mismatch", "error": f"unsupported ndim={arr.ndim}"}

    align_method = "direct"
    source_shape = tuple(arr.shape)
    if tuple(arr.shape) != tuple(ct_shape):
        if min(arr.shape) <= 0:
            return {"status": "seg_shape_mismatch", "error": f"invalid seg shape={arr.shape}"}
        zoom_factors = (
            ct_shape[0] / arr.shape[0],
            ct_shape[1] / arr.shape[1],
            ct_shape[2] / arr.shape[2],
        )
        arr = ndimage.zoom(arr.astype("float32"), zoom=zoom_factors, order=0)
        align_method = "resampled_nn"
        if tuple(arr.shape) != tuple(ct_shape):
            fixed = np.zeros(ct_shape, dtype=arr.dtype)
            z = min(ct_shape[0], arr.shape[0])
            y = min(ct_shape[1], arr.shape[1])
            x = min(ct_shape[2], arr.shape[2])
            fixed[:z, :y, :x] = arr[:z, :y, :x]
            arr = fixed

    mask = (arr > 0).astype("uint8")
    if int(mask.sum()) == 0:
        return {"status": "seg_empty", "error": "mask is empty"}
    return {"status": "ok", "mask": mask, "align_method": align_method, "source_shape": source_shape}


def compute_roi_token(volume_norm, tumor_mask, deps):
    """Why: 5.3 需要把肿瘤 ROI 压缩成固定维度 token 用于建模。

    Content: 从肿瘤区域提取统计特征并投影到固定维度向量。
    Input: volume_norm（归一化 CT），tumor_mask（二值 mask），deps。
    Output: 包含状态、token、错误信息的字典。
    """
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
    """Why: 5.4 需要把 AIM 语义注释转成可量化特征。

    Content: 解析 XML，收集标签名、属性值和文本内容作为语义特征字符串。
    Input: aim_xml_path（AIM 文件路径）。
    Output: 特征字符串列表；解析失败返回空列表。
    """
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
    """Why: 5.4 最终需要固定维度语义 token 作为模型输入。

    Content: 使用哈希技巧将任意数量文本特征映射到固定维度向量并归一化。
    Input: feature_texts（字符串列表），token_dim（目标维度）。
    Output: token 浮点列表；若输入为空则返回空列表。
    """
    if not feature_texts:
        return []
    vec = [0.0 for _ in range(token_dim)]
    for text in feature_texts:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % token_dim
        sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
        vec[idx] += sign

        # 尝试提取文本中的数字，额外编码到 token。
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
    """Why: 归一化 CT 结果需要持久化，供训练阶段重复使用。

    Content: 将归一化 CT 与 spacing 保存为压缩 npz。
    Input: patient_id，ct_result，deps。
    Output: 输出文件路径字符串。
    """
    np = deps["np"]
    out_path = OUTPUT_CT_DIR / f"{patient_id}.npz"
    np.savez_compressed(
        out_path,
        ct_volume=ct_result["volume_norm"],
        spacing_target=np.asarray(ct_result["spacing_target"], dtype="float32"),
    )
    return str(out_path)


def save_seg_npz(patient_id, mask, deps):
    """Why: 肿瘤 mask 需要落盘，便于后续 ROI 和训练直接读取。

    Content: 将 mask 保存为压缩 npz。
    Input: patient_id，mask，deps。
    Output: 输出文件路径字符串。
    """
    np = deps["np"]
    out_path = OUTPUT_SEG_DIR / f"{patient_id}.npz"
    np.savez_compressed(out_path, mask=mask.astype("uint8"))
    return str(out_path)


def process_patient(patient_id, series_index, deps):
    """Why: 把 5.1~5.4 串在一起，形成单病人端到端处理单元。

    Content: 选择 CT/SEG/AIM，执行预处理，返回摘要和 token 结果。
    Input: patient_id，series_index，deps。
    Output: 包含 summary/semantic_token/roi_token 的字典。
    """
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
        "seg_align_method": "",
        "seg_source_shape": "",
        "roi_token_status": "",
        "roi_token_dim": 0,
        "semantic_status": "",
        "semantic_token_dim": 0,
        "aim_xml_path": aim_xml,
    }

    semantic_row = {"patient_id": patient_id, "token_json": ""}
    roi_row = {"patient_id": patient_id, "token_json": ""}

    # 5.4 Semantic annotation token (标准库可跑)
    feature_texts = parse_aim_feature_texts(aim_xml)
    semantic_token = build_semantic_token(feature_texts, SEMANTIC_TOKEN_DIM)
    if semantic_token:
        semantic_row["token_json"] = json.dumps(semantic_token)
        summary["semantic_status"] = "ok"
        summary["semantic_token_dim"] = len(semantic_token)
    else:
        summary["semantic_status"] = "aim_missing_or_empty"
        summary["semantic_token_dim"] = 0

    # 5.1 CT normalization
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

    # 5.2 Tumor segmentation
    seg_result = None
    if seg_row is None:
        summary["seg_status"] = "seg_series_not_found"
    else:
        seg_dir = resolve_series_dir(seg_row.get("File Location"))
        summary["seg_series_dir"] = str(seg_dir)
        if ct_result is None or ct_result.get("status") != "ok":
            summary["seg_status"] = "blocked_ct_unavailable"
        else:
            seg_result = load_tumor_mask(seg_dir, ct_result["volume_original"].shape, deps)
            summary["seg_status"] = seg_result["status"]
            if seg_result["status"] == "ok":
                summary["seg_mask_npz"] = save_seg_npz(patient_id, seg_result["mask"], deps)
                summary["seg_align_method"] = seg_result.get("align_method", "")
                source_shape = seg_result.get("source_shape")
                if source_shape:
                    summary["seg_source_shape"] = "x".join(str(v) for v in source_shape)

    # 5.3 Tumor ROI token
    if ct_result is None or ct_result.get("status") != "ok":
        summary["roi_token_status"] = "blocked_ct_unavailable"
    elif seg_result is None or seg_result.get("status") != "ok":
        summary["roi_token_status"] = "blocked_seg_unavailable"
    else:
        # Tumor mask is aligned to original CT space, so ROI token uses native-space normalized CT.
        roi_result = compute_roi_token(ct_result["volume_norm_native"], seg_result["mask"], deps)
        summary["roi_token_status"] = roi_result["status"]
        if roi_result["status"] == "ok":
            roi_row["token_json"] = json.dumps(roi_result["token"])
            summary["roi_token_dim"] = len(roi_result["token"])

    return {"summary": summary, "semantic_row": semantic_row, "roi_row": roi_row}


def write_csv(path, fieldnames, rows):
    """Why: 预处理会写多张表，统一 CSV 写入逻辑可减少重复代码。

    Content: 按给定字段顺序写入 CSV。
    Input: path、fieldnames、rows。
    Output: CSV 文件写入完成。
    """
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(max_cases):
    """Why: 统一调度所有病人的 5.1~5.4 处理流程。

    Content: 加载输入、遍历病人、执行预处理并写出结果文件。
    Input: max_cases（最多处理多少病人，None 表示全部）。
    Output: 终端统计 + 产出 CSV/NPZ 文件。
    """
    ensure_output_dirs()
    deps = check_imaging_dependencies()
    patient_ids = load_patient_ids()
    metadata_rows = load_metadata_rows()
    series_index = build_series_index(metadata_rows)

    if max_cases is not None:
        patient_ids = patient_ids[:max_cases]

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
        "seg_align_method",
        "seg_source_shape",
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
    """Why: 便于按批次调试，不必每次处理全量病例。

    Content: 解析命令行参数。
    Input: 命令行。
    Output: argparse 参数对象。
    """
    parser = argparse.ArgumentParser(description="Run imaging preprocessing pipeline (5.1~5.4).")
    parser.add_argument("--max-cases", type=int, default=None, help="Only process first N patients.")
    return parser.parse_args()


def main():
    """Why: 提供脚本入口，一条命令执行整套流程。

    Content: 读取参数并启动 pipeline。
    Input: 命令行参数。
    Output: 预处理结果文件和终端日志。
    """
    args = parse_args()
    run_pipeline(args.max_cases)


if __name__ == "__main__":
    main()
