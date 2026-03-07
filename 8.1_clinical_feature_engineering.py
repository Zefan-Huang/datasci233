"""
作用：
- 实现 project.md Stage 8.1：Clinical feature engineering（clinical.csv -> x_ehr）。
- 连续变量：标准化（z-score）+ 缺失指示（missing indicator）。
- 类别变量：支持 one-hot（默认）或 index 编码（供后续 embedding 使用）。
- 明确排除结局/未来日期相关字段，以及依赖 CT 之后时间点的间隔字段，避免标签泄漏。

输入：
- output/clean_data/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv（优先）
- data/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv（回退）
- output/patient_manifest.csv（优先，用于 patient 过滤与对齐）
- output/clean_data/patient_manifest.csv（回退）

输出：
- output/stage8/8.1_clinical_feature_engineering/x_ehr_features.npz
- output/stage8/8.1_clinical_feature_engineering/x_ehr_features.csv
- output/stage8/8.1_clinical_feature_engineering/categorical_vocab.csv
- output/stage8/8.1_clinical_feature_engineering/continuous_stats.csv
- output/stage8/8.1_clinical_feature_engineering/clinical_feature_summary.json
"""
import argparse
import csv
import json
import re
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None


PRIMARY_CLINICAL_CSV = Path("output/clean_data/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")
FALLBACK_CLINICAL_CSV = Path("data/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")

PRIMARY_MANIFEST_CSV = Path("output/patient_manifest.csv")
FALLBACK_MANIFEST_CSV = Path("output/clean_data/patient_manifest.csv")

DEFAULT_OUTPUT_ROOT = Path("output/stage8/8.1_clinical_feature_engineering")

PATIENT_ID_COL = "Case ID"

MISSING_VALUES = {
    "",
    "na",
    "n/a",
    "null",
    "not collected",
    "not recorded in database",
}

# Stage 8.1 输入特征中明确排除：结局标签、未来日期，以及依赖 post-CT 时间点的字段。
EXCLUDED_COLUMNS = {
    "Recurrence",
    "Recurrence Location",
    "Date of Recurrence",
    "Date of Last Known Alive",
    "Survival Status",
    "Date of Death",
    "Time to Death (days)",
    "CT Date",
    "PET Date",
    "Days between CT and surgery",
}

# 默认连续字段（其余非排除字段自动视作类别字段）。
DEFAULT_CONTINUOUS_COLUMNS = [
    "Age at Histological Diagnosis",
    "Weight (lbs)",
    "Pack Years",
    "Quit Smoking Year",
    "%GG",
]


def check_dependencies():
    """Why: Stage 8.1 的矩阵计算依赖 numpy，缺失时应提前失败。

    Content: 检查 numpy 是否可用。
    Input: 无。
    Output: 缺失依赖列表。
    """
    missing = []
    if np is None:
        missing.append("numpy")
    return missing


def resolve_input_path(primary_path, fallback_path, label):
    """Why: 项目有 output/data 两套目录布局，脚本需稳定定位输入文件。

    Content: 优先使用 primary_path，不存在则回退到 fallback_path。
    Input: primary_path、fallback_path、label。
    Output: 可用 Path。
    """
    if primary_path.exists():
        return primary_path
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(
        f"{label} not found in either '{primary_path}' or '{fallback_path}'"
    )


def ensure_output_dir(output_root):
    """Why: Stage 8.1 需要输出多个文件，先创建目录可避免写文件失败。

    Content: 创建输出目录（含父目录）。
    Input: output_root。
    Output: 目录创建完成。
    """
    output_root.mkdir(parents=True, exist_ok=True)


def normalize_missing_text(value):
    """Why: 原始临床数据缺失写法不统一，需要归一化处理。

    Content: 去除空白并把缺失语义映射为空字符串。
    Input: value（任意字符串或 None）。
    Output: 清洗后字符串（缺失返回空字符串）。
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in MISSING_VALUES:
        return ""
    return text


def canonical_category(value):
    """Why: 类别值可能大小写/空白不一致，需先规范化避免同义分裂。

    Content: 将类别值转为小写并压缩空白；缺失映射到 __MISSING__。
    Input: value。
    Output: 规范化类别字符串。
    """
    text = normalize_missing_text(value)
    if not text:
        return "__MISSING__"
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def parse_numeric(value):
    """Why: 连续特征可能带百分号或区间文本，需要稳健解析为数值。

    Content: 支持纯数字、百分比、以及区间（取中点）；解析失败返回 None。
    Input: value。
    Output: float 或 None。
    """
    text = normalize_missing_text(value)
    if not text:
        return None

    text = text.replace(",", "")
    text = text.replace("%", "")
    text = text.replace("<", "")
    text = text.replace(">", "")
    text = text.strip()

    # 区间：例如 "25 - 50"、"75 - 100"
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$", text)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        return (a + b) / 2.0

    try:
        return float(text)
    except Exception:
        return None


def safe_feature_token(text):
    """Why: 特征名将写入 CSV 列头，需要保证稳定且可读。

    Content: 将文本转为安全 token（仅保留字母数字下划线）。
    Input: text。
    Output: token 字符串。
    """
    token = re.sub(r"[^0-9a-zA-Z_]+", "_", str(text)).strip("_")
    return token if token else "empty"


def make_unique_feature_names(feature_names):
    """Why: 不同字段/类别经安全化后可能重名，需要去重避免列冲突。

    Content: 对重复名称追加 __dupN 后缀。
    Input: feature_names 列表。
    Output: 唯一名称列表。
    """
    out = []
    seen = {}
    for name in feature_names:
        if name not in seen:
            seen[name] = 0
            out.append(name)
            continue
        seen[name] += 1
        out.append(f"{name}__dup{seen[name]}")
    return out


def load_clinical_rows(clinical_csv_path):
    """Why: 8.1 需要病人级临床原始记录作为输入源。

    Content: 读取 clinical CSV 并按 patient_id 索引。
    Input: clinical_csv_path。
    Output: (header, patient_to_row)。
    """
    with clinical_csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        patient_to_row = {}
        for row in reader:
            pid = normalize_missing_text(row.get(PATIENT_ID_COL))
            if not pid:
                continue
            patient_to_row[pid] = row
    if PATIENT_ID_COL not in header:
        raise RuntimeError(f"clinical csv missing required column: {PATIENT_ID_COL}")
    return header, patient_to_row


def load_manifest_patient_ids(manifest_csv_path):
    """Why: Stage 8.1 通常应与项目 cohort 对齐，而非使用所有临床行。

    Content: 读取 manifest 的 patient_id 列；若文件不存在返回空列表。
    Input: manifest_csv_path。
    Output: patient_id 列表。
    """
    if not manifest_csv_path.exists():
        return []

    ids = []
    with manifest_csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "patient_id" not in (reader.fieldnames or []):
            return []
        for row in reader:
            pid = normalize_missing_text(row.get("patient_id"))
            if pid:
                ids.append(pid)
    return ids


def choose_patient_ids(patient_to_row, manifest_patient_ids, use_manifest_filter, max_patients):
    """Why: 训练前需要稳定的病人顺序与过滤规则，便于复现。

    Content: 可选按 manifest 过滤；支持 max_patients 截断调试。
    Input: patient_to_row、manifest_patient_ids、use_manifest_filter、max_patients。
    Output: 选中的 patient_id 列表。
    """
    if max_patients < 0:
        raise RuntimeError("max_patients must be >= 0")

    if use_manifest_filter and manifest_patient_ids:
        patient_ids = [pid for pid in manifest_patient_ids if pid in patient_to_row]
    else:
        patient_ids = sorted(patient_to_row.keys())

    if max_patients > 0:
        patient_ids = patient_ids[: min(max_patients, len(patient_ids))]
    return patient_ids


def resolve_feature_columns(header, continuous_columns):
    """Why: 8.1 需要明确哪些列做连续处理、哪些列做类别处理。

    Content: 先选连续列，再把其余非排除列归入类别列。
    Input: header、continuous_columns（候选列表）。
    Output: (continuous_cols, categorical_cols, missing_cont_cols)。
    """
    continuous_cols = []
    missing_cont_cols = []
    for col in continuous_columns:
        if col in EXCLUDED_COLUMNS or col == PATIENT_ID_COL:
            continue
        if col in header:
            continuous_cols.append(col)
        else:
            missing_cont_cols.append(col)

    categorical_cols = []
    for col in header:
        if col == PATIENT_ID_COL:
            continue
        if col in EXCLUDED_COLUMNS:
            continue
        if col in continuous_cols:
            continue
        categorical_cols.append(col)

    return continuous_cols, categorical_cols, missing_cont_cols


def build_continuous_features(patient_rows, continuous_cols, drop_constant_features):
    """Why: 连续变量需要标准化并补充缺失指示，以提升模型鲁棒性。

    Content: 对每个连续列计算 z-score 和 missing indicator，可选去掉常量特征。
    Input: patient_rows、continuous_cols、drop_constant_features。
    Output: 特征矩阵、特征名、统计行、保留列、丢弃列。
    """
    n = len(patient_rows)
    c = len(continuous_cols)
    if c == 0:
        return (
            np.zeros((n, 0), dtype=np.float32),
            [],
            [],
            [],
            [],
        )

    raw = np.full((n, c), np.nan, dtype=np.float32)
    for i, row in enumerate(patient_rows):
        for j, col in enumerate(continuous_cols):
            val = parse_numeric(row.get(col))
            if val is not None:
                raw[i, j] = float(val)

    means = np.zeros((c,), dtype=np.float32)
    stds = np.zeros((c,), dtype=np.float32)
    for j in range(c):
        valid = raw[~np.isnan(raw[:, j]), j]
        if valid.size == 0:
            means[j] = 0.0
            stds[j] = 0.0
        else:
            means[j] = float(valid.mean())
            stds[j] = float(valid.std())
    safe_stds = np.where(stds > 1e-8, stds, 1.0).astype(np.float32)

    missing = np.isnan(raw).astype(np.float32)
    z = (raw - means[None, :]) / safe_stds[None, :]
    z[np.isnan(z)] = 0.0
    z = z.astype(np.float32)

    feature_parts = []
    feature_names = []
    stats_rows = []
    kept_cols = []
    dropped_cols = []

    for j, col in enumerate(continuous_cols):
        z_col = z[:, j]
        m_col = missing[:, j]
        miss_rate = float(m_col.mean()) if len(m_col) > 0 else 0.0
        raw_non_missing = raw[~np.isnan(raw[:, j]), j]

        base_token = safe_feature_token(col)
        z_name = f"cont::{base_token}"
        m_name = f"miss::{base_token}"

        z_const = bool(np.allclose(z_col, z_col[0])) if len(z_col) > 0 else True
        m_const = bool(np.allclose(m_col, m_col[0])) if len(m_col) > 0 else True

        if drop_constant_features and z_const and m_const:
            dropped_cols.append(col)
            continue

        kept_cols.append(col)
        stats_rows.append(
            {
                "column_name": col,
                "mean_raw": float(means[j]),
                "std_raw": float(stds[j]),
                "safe_std_used": float(safe_stds[j]),
                "missing_rate": miss_rate,
                "non_missing_count": int(raw_non_missing.shape[0]),
            }
        )

        if not (drop_constant_features and z_const):
            feature_parts.append(z_col[:, None].astype(np.float32))
            feature_names.append(z_name)
        if not (drop_constant_features and m_const):
            feature_parts.append(m_col[:, None].astype(np.float32))
            feature_names.append(m_name)

    if not feature_parts:
        return (
            np.zeros((n, 0), dtype=np.float32),
            [],
            stats_rows,
            kept_cols,
            dropped_cols,
        )

    x = np.concatenate(feature_parts, axis=1).astype(np.float32)
    return x, feature_names, stats_rows, kept_cols, dropped_cols


def build_categorical_onehot_features(patient_rows, categorical_cols, drop_constant_features):
    """Why: one-hot 是最直接可用的类别编码，输出可直接拼接进 x_ehr。

    Content: 按列构建 one-hot，可选删除常量列/常量特征。
    Input: patient_rows、categorical_cols、drop_constant_features。
    Output: 特征矩阵、特征名、词表行、保留列、丢弃列。
    """
    n = len(patient_rows)
    if len(categorical_cols) == 0:
        return (
            np.zeros((n, 0), dtype=np.float32),
            [],
            [],
            [],
            [],
        )

    feature_parts = []
    feature_names = []
    vocab_rows = []
    kept_cols = []
    dropped_cols = []

    for col in categorical_cols:
        values = [canonical_category(row.get(col)) for row in patient_rows]
        vocab = sorted(set(values))

        if drop_constant_features and len(vocab) <= 1:
            dropped_cols.append(col)
            continue

        kept_cols.append(col)
        col_token = safe_feature_token(col)
        for cat in vocab:
            cat_token = safe_feature_token(cat)
            vec = np.asarray([1.0 if x == cat else 0.0 for x in values], dtype=np.float32)
            if drop_constant_features and bool(np.allclose(vec, vec[0])):
                continue
            feature_parts.append(vec[:, None])
            feature_names.append(f"cat::{col_token}::{cat_token}")
            vocab_rows.append(
                {
                    "column_name": col,
                    "encoding": "onehot",
                    "category_value": cat,
                    "category_index": "",
                }
            )

    if not feature_parts:
        return (
            np.zeros((n, 0), dtype=np.float32),
            [],
            vocab_rows,
            kept_cols,
            dropped_cols,
        )

    x = np.concatenate(feature_parts, axis=1).astype(np.float32)
    return x, feature_names, vocab_rows, kept_cols, dropped_cols


def build_categorical_index_features(patient_rows, categorical_cols, drop_constant_features):
    """Why: index 编码适合后续 embedding 层，避免 one-hot 维度膨胀。

    Content: 每个类别列映射到整数 id（缺失也作为类别）。
    Input: patient_rows、categorical_cols、drop_constant_features。
    Output: 特征矩阵、特征名、词表行、保留列、丢弃列。
    """
    n = len(patient_rows)
    if len(categorical_cols) == 0:
        return (
            np.zeros((n, 0), dtype=np.float32),
            [],
            [],
            [],
            [],
        )

    feature_parts = []
    feature_names = []
    vocab_rows = []
    kept_cols = []
    dropped_cols = []

    for col in categorical_cols:
        values = [canonical_category(row.get(col)) for row in patient_rows]
        vocab = sorted(set(values))

        if drop_constant_features and len(vocab) <= 1:
            dropped_cols.append(col)
            continue

        kept_cols.append(col)
        idx_map = {cat: i for i, cat in enumerate(vocab)}
        vec = np.asarray([idx_map[v] for v in values], dtype=np.float32)
        if drop_constant_features and bool(np.allclose(vec, vec[0])):
            dropped_cols.append(col)
            kept_cols = [x for x in kept_cols if x != col]
            continue

        feature_parts.append(vec[:, None])
        feature_names.append(f"cat_idx::{safe_feature_token(col)}")
        for cat, idx in idx_map.items():
            vocab_rows.append(
                {
                    "column_name": col,
                    "encoding": "index",
                    "category_value": cat,
                    "category_index": int(idx),
                }
            )

    if not feature_parts:
        return (
            np.zeros((n, 0), dtype=np.float32),
            [],
            vocab_rows,
            kept_cols,
            dropped_cols,
        )

    x = np.concatenate(feature_parts, axis=1).astype(np.float32)
    return x, feature_names, vocab_rows, kept_cols, dropped_cols


def write_csv(path, fieldnames, rows):
    """Why: 工程产物需要可读可追踪，CSV 是最通用格式。

    Content: 按给定字段写 CSV。
    Input: path、fieldnames、rows。
    Output: CSV 文件。
    """
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_x_ehr_csv(path, patient_ids, feature_names, x_ehr):
    """Why: 便于直接人工检查每个病人的 EHR 特征向量。

    Content: 写 patient_id + 全部 x_ehr 列到 CSV。
    Input: path、patient_ids、feature_names、x_ehr。
    Output: x_ehr_features.csv。
    """
    fieldnames = ["patient_id"] + list(feature_names)
    rows = []
    for i, pid in enumerate(patient_ids):
        row = {"patient_id": str(pid)}
        for j, name in enumerate(feature_names):
            row[name] = float(x_ehr[i, j])
        rows.append(row)
    write_csv(path, fieldnames, rows)


def write_summary_json(path, summary):
    """Why: 8.1 需要保留关键统计，便于复现和审计。

    Content: 写 summary JSON。
    Input: path、summary。
    Output: json 文件。
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args():
    """Why: Stage 8.1 需要可调参数以支持全量/调试运行。

    Content: 解析参数并忽略 IDE 注入的未知参数。
    Input: 命令行参数。
    Output: 参数对象。
    """
    parser = argparse.ArgumentParser(
        description="Stage 8.1 Clinical feature engineering: clinical.csv -> x_ehr",
        allow_abbrev=False,
    )
    parser.add_argument("--clinical-csv", type=str, default="")
    parser.add_argument("--manifest-csv", type=str, default="")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="0 means all selected patients; >0 means first N patients.",
    )
    parser.add_argument(
        "--no-manifest-filter",
        action="store_true",
        help="If set, do not filter patients by patient_manifest.csv.",
    )
    parser.add_argument(
        "--categorical-encoding",
        type=str,
        default="onehot",
        choices=["onehot", "index"],
        help="onehot for direct numeric x_ehr; index for downstream embedding models.",
    )
    parser.add_argument(
        "--continuous-columns",
        type=str,
        default=",".join(DEFAULT_CONTINUOUS_COLUMNS),
        help="Comma-separated continuous columns.",
    )
    parser.add_argument(
        "--keep-constant-features",
        action="store_true",
        help="If set, keep constant features (default behavior is dropping constants).",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"ignore_unknown_args: {unknown}")
    return args


def main():
    """Why: 一条命令完成 Stage 8.1 并输出 x_ehr。

    Content: 读 clinical/manifest，做特征工程，导出 x_ehr 与统计文件。
    Input: 命令行参数。
    Output: npz/csv/json 结果文件。
    """
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

    clinical_csv_path = (
        Path(args.clinical_csv)
        if args.clinical_csv.strip()
        else resolve_input_path(PRIMARY_CLINICAL_CSV, FALLBACK_CLINICAL_CSV, "clinical csv")
    )

    if args.manifest_csv.strip():
        manifest_csv_path = Path(args.manifest_csv)
    else:
        manifest_csv_path = (
            PRIMARY_MANIFEST_CSV if PRIMARY_MANIFEST_CSV.exists() else FALLBACK_MANIFEST_CSV
        )

    output_root = Path(args.output_root)
    ensure_output_dir(output_root)

    use_manifest_filter = not bool(args.no_manifest_filter)
    drop_constant_features = not bool(args.keep_constant_features)

    continuous_columns_arg = [
        x.strip() for x in args.continuous_columns.split(",") if x.strip()
    ]

    print(f"[start] clinical_csv={clinical_csv_path}")
    print(f"[start] manifest_csv={manifest_csv_path}")
    print(
        f"[start] max_patients={args.max_patients} "
        f"full_patients={1 if args.max_patients == 0 else 0} "
        f"use_manifest_filter={1 if use_manifest_filter else 0}"
    )
    print(
        f"[start] categorical_encoding={args.categorical_encoding} "
        f"drop_constant_features={1 if drop_constant_features else 0}"
    )
    print(f"[deps] numpy={np.__version__}")

    header, patient_to_row = load_clinical_rows(clinical_csv_path)
    manifest_patient_ids = load_manifest_patient_ids(manifest_csv_path)

    patient_ids = choose_patient_ids(
        patient_to_row=patient_to_row,
        manifest_patient_ids=manifest_patient_ids,
        use_manifest_filter=use_manifest_filter,
        max_patients=args.max_patients,
    )
    if len(patient_ids) == 0:
        raise RuntimeError("no patient selected after filtering")

    patient_rows = [patient_to_row[pid] for pid in patient_ids]

    continuous_cols, categorical_cols, missing_cont_cols = resolve_feature_columns(
        header=header,
        continuous_columns=continuous_columns_arg,
    )

    print(
        f"[columns] continuous_selected={len(continuous_cols)} "
        f"categorical_selected={len(categorical_cols)} "
        f"excluded={len(EXCLUDED_COLUMNS)}"
    )
    if missing_cont_cols:
        print(f"[warn] continuous columns not found and skipped: {missing_cont_cols}")

    print("[stage] build continuous features")
    x_cont, cont_feature_names, cont_stats_rows, cont_kept, cont_dropped = build_continuous_features(
        patient_rows=patient_rows,
        continuous_cols=continuous_cols,
        drop_constant_features=drop_constant_features,
    )

    print("[stage] build categorical features")
    if args.categorical_encoding == "onehot":
        (
            x_cat,
            cat_feature_names,
            vocab_rows,
            cat_kept,
            cat_dropped,
        ) = build_categorical_onehot_features(
            patient_rows=patient_rows,
            categorical_cols=categorical_cols,
            drop_constant_features=drop_constant_features,
        )
    else:
        (
            x_cat,
            cat_feature_names,
            vocab_rows,
            cat_kept,
            cat_dropped,
        ) = build_categorical_index_features(
            patient_rows=patient_rows,
            categorical_cols=categorical_cols,
            drop_constant_features=drop_constant_features,
        )

    x_ehr = np.concatenate([x_cont, x_cat], axis=1).astype(np.float32)
    feature_names = cont_feature_names + cat_feature_names
    feature_names = make_unique_feature_names(feature_names)

    if x_ehr.shape[1] != len(feature_names):
        raise RuntimeError(
            f"x_ehr feature dim mismatch: matrix={x_ehr.shape[1]} names={len(feature_names)}"
        )

    out_npz = output_root / "x_ehr_features.npz"
    out_csv = output_root / "x_ehr_features.csv"
    out_vocab_csv = output_root / "categorical_vocab.csv"
    out_cont_stats_csv = output_root / "continuous_stats.csv"
    out_summary_json = output_root / "clinical_feature_summary.json"

    np.savez_compressed(
        out_npz,
        x_ehr=x_ehr.astype(np.float32),
        patient_ids=np.asarray(patient_ids).astype(str),
        feature_names=np.asarray(feature_names).astype(str),
        categorical_encoding=np.asarray([args.categorical_encoding]).astype(str),
        continuous_columns_kept=np.asarray(cont_kept).astype(str),
        categorical_columns_kept=np.asarray(cat_kept).astype(str),
    )
    write_x_ehr_csv(out_csv, patient_ids, feature_names, x_ehr)
    write_csv(
        out_vocab_csv,
        ["column_name", "encoding", "category_value", "category_index"],
        vocab_rows,
    )
    write_csv(
        out_cont_stats_csv,
        ["column_name", "mean_raw", "std_raw", "safe_std_used", "missing_rate", "non_missing_count"],
        cont_stats_rows,
    )

    summary = {
        "clinical_csv_path": str(clinical_csv_path),
        "manifest_csv_path": str(manifest_csv_path),
        "use_manifest_filter": 1 if use_manifest_filter else 0,
        "max_patients": int(args.max_patients),
        "selected_patient_count": int(len(patient_ids)),
        "clinical_row_count": int(len(patient_to_row)),
        "categorical_encoding": args.categorical_encoding,
        "drop_constant_features": 1 if drop_constant_features else 0,
        "x_ehr_shape": [int(x_ehr.shape[0]), int(x_ehr.shape[1])],
        "feature_count_total": int(x_ehr.shape[1]),
        "continuous_feature_count": int(x_cont.shape[1]),
        "categorical_feature_count": int(x_cat.shape[1]),
        "continuous_columns_requested": continuous_columns_arg,
        "continuous_columns_found": continuous_cols,
        "continuous_columns_kept": cont_kept,
        "continuous_columns_dropped": cont_dropped,
        "categorical_columns_found": categorical_cols,
        "categorical_columns_kept": cat_kept,
        "categorical_columns_dropped": cat_dropped,
        "continuous_columns_not_found": missing_cont_cols,
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
    }
    write_summary_json(out_summary_json, summary)

    print(f"wrote: {out_npz}")
    print(f"wrote: {out_csv}")
    print(f"wrote: {out_vocab_csv}")
    print(f"wrote: {out_cont_stats_csv}")
    print(f"wrote: {out_summary_json}")
    print(f"x_ehr_shape: {tuple(x_ehr.shape)}")
    print("complete")


if __name__ == "__main__":
    main()
