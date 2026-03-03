"""这个文件用于生成 patient_manifest 总表。

作用：
- 汇总每个病人的模态可用性（CT/PET/SEG/AIM/RNA）。
- 根据临床字段构建 OS/复发相关标签字段。

输入：
- output/clean_data/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv（优先）
- data/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv（回退）
- data/manifest-1622561851074/metadata.csv
- data/AIM_files_updated-11-10-2020/*.xml
- output/clean_data/GSE103584_series_matrix.txt（优先）
- data/GSE103584_series_matrix.txt（回退）

输出：
- output/patient_manifest.csv
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path


DATA_DIR = Path("data")
OUTPUT_CLEAN_DIR = Path("output") / "clean_data"
PRIMARY_CLINICAL_CSV = OUTPUT_CLEAN_DIR / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
LEGACY_CLINICAL_CSV = DATA_DIR / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
METADATA_CSV = DATA_DIR / "manifest-1622561851074" / "metadata.csv"
AIM_DIR = DATA_DIR / "AIM_files_updated-11-10-2020"
PRIMARY_SERIES_MATRIX_TXT = OUTPUT_CLEAN_DIR / "GSE103584_series_matrix.txt"
LEGACY_SERIES_MATRIX_TXT = DATA_DIR / "GSE103584_series_matrix.txt"
OUTPUT_CSV = Path("output") / "patient_manifest.csv"

MISSING_STRINGS = {"", "n/a", "na", "not collected", "not recorded in database", "null",}


def normalize_missing(value):
    """Why: 各数据字段缺失写法不一致，需要统一成一个规则。

    Content: 去掉前后空格，并把缺失语义文本转为空字符串。
    Input: value（字符串或 None）。
    Output: 清理后的字符串；若缺失则返回空字符串。
    """
    if value is None:
        return ""
    cleaned = value.strip()
    if cleaned.lower() in MISSING_STRINGS:
        return ""
    return cleaned


def resolve_input_path(primary_path, fallback_path, label):
    """Why: 当前项目有 output/data 两套目录，需要稳定选择可用输入文件。

    Content: 优先使用 primary_path；不存在时回退 fallback_path；都不存在则报错。
    Input: primary_path、fallback_path、label。
    Output: 可读取的 Path。
    """
    if primary_path.exists():
        return primary_path
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(
        f"{label} not found in either '{primary_path}' or '{fallback_path}'"
    )


def parse_date(value):
    """Why: 生存/复发时间都依赖日期差，必须先稳定解析日期。

    Content: 先清理缺失值，再按支持格式解析日期，失败返回 None。
    Input: value（日期字符串或 None）。
    Output: date 对象或 None。
    """
    cleaned = normalize_missing(value)
    if not cleaned:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            pass
    return None


def days_between(start, end):
    """Why: 统一计算 time-to-event 天数，避免各处实现不一致。

    Content: 计算 end-start 的非负天数；无效输入返回 None。
    Input: start（起始日期），end（结束日期）。
    Output: 天数整数或 None。
    """
    if start is None or end is None:
        return None
    delta = (end - start).days
    if delta < 0:
        return None
    return delta


def load_modality_flags():
    """Why: manifest 需要知道每个病人的模态可用性（CT/PET/SEG）。

    Content: 读取影像 metadata，按病人聚合 has_ct/has_pet/has_seg。
    Input: METADATA_CSV。
    Output: 以 patient_id 为键的字典，值为三种可用性标记。
    """
    flags = defaultdict(
        lambda: {
            "has_ct": 0,
            "has_pet": 0,
            "has_seg": 0,
        }
    )
    with METADATA_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Collection") != "NSCLC Radiogenomics":
                continue
            pid = row["Subject ID"].strip()
            modality = (row.get("Modality") or "").strip()
            if modality == "CT":
                flags[pid]["has_ct"] = 1
            if modality == "PT":
                flags[pid]["has_pet"] = 1
            sop_uid = (row.get("SOP Class UID") or "").strip()
            sop_name = (row.get("SOP Class Name") or "").strip()
            if sop_uid == "1.2.840.10008.5.1.4.1.1.66.4" or "Segmentation Storage" in sop_name:
                flags[pid]["has_seg"] = 1
    return flags


def load_aim_cases():
    """Why: manifest 需要知道语义注释（AIM）是否存在。

    Content: 扫描 AIM XML 文件并标准化病例 ID（去掉 v1 这类后缀）。
    Input: AIM_DIR 下的 XML 文件。
    Output: 有 AIM 的病例 ID 集合。
    """
    out = set()
    for xml_file in AIM_DIR.glob("*.xml"):
        # R01-023v1.xml 规范化为 R01-023
        case_id = re.sub(r"v\d+$", "", xml_file.stem)
        out.add(case_id)
    return out


def load_rna_mapping(series_matrix_path):
    """Why: 多模态训练需要把 RNA 样本（GSM）映射到病例 ID。

    Content: 从 series_matrix 的标题行和 GEO accession 行提取映射。
    Input: series_matrix_path。
    Output: case_id 到 gsm_id 的映射字典。
    """
    case_to_gsm = {}
    sample_titles = None
    sample_geo = None

    with series_matrix_path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("!Sample_title\t"):
                sample_titles = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession\t"):
                sample_geo = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]

    if sample_titles is None or sample_geo is None:
        raise RuntimeError("Failed to parse !Sample_title / !Sample_geo_accession from series_matrix")
    if len(sample_titles) != len(sample_geo):
        raise RuntimeError("series_matrix has mismatched title vs geo lengths")

    for gsm, case_id in zip(sample_geo, sample_titles):
        case_to_gsm[case_id] = gsm

    return case_to_gsm


def build_patient_manifest(clinical_csv_path, series_matrix_path):
    """Why: 训练前需要一张病人级总表，把标签和模态状态放在一起。

    Content: 合并临床、影像模态、AIM、RNA 映射，生成每病人一行的 manifest。
    Input: clinical_csv_path + METADATA_CSV + AIM_DIR + series_matrix_path。
    Output: 排序后的 manifest 行列表（字典列表）。
    """
    modality_flags = load_modality_flags()
    aim_cases = load_aim_cases()
    case_to_gsm = load_rna_mapping(series_matrix_path)

    output_rows = []
    with clinical_csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row["Case ID"].strip()

            ct_date = parse_date(row.get("CT Date"))
            pet_date = parse_date(row.get("PET Date"))
            date_last_alive = parse_date(row.get("Date of Last Known Alive"))
            date_death = parse_date(row.get("Date of Death"))
            date_recurrence = parse_date(row.get("Date of Recurrence"))

            survival_status = normalize_missing(row.get("Survival Status"))
            survival_status_lower = survival_status.lower()
            if survival_status_lower == "dead":
                event_os = 1
                os_label_known = 1
                os_anchor_date = date_death if date_death is not None else date_last_alive
            elif survival_status_lower == "alive":
                event_os = 0
                os_label_known = 1
                os_anchor_date = date_last_alive
            else:
                event_os = ""
                os_label_known = 0
                os_anchor_date = None
            time_os = days_between(ct_date, os_anchor_date)

            rec_raw = normalize_missing(row.get("Recurrence")).lower()
            rec_location = normalize_missing(row.get("Recurrence Location")).lower()
            if rec_raw == "yes":
                event_rec = 1
                rec_label_known = 1
                time_rec = days_between(ct_date, date_recurrence)
                rec_location_class = rec_location
            elif rec_raw == "no":
                event_rec = 0
                rec_label_known = 1
                time_rec = days_between(ct_date, date_last_alive)
                rec_location_class = ""
            else:
                event_rec = ""
                rec_label_known = 0
                time_rec = None
                rec_location_class = ""

            manifest_row = {
                "patient_id": patient_id,
                "has_ct": modality_flags[patient_id]["has_ct"],
                "has_pet": modality_flags[patient_id]["has_pet"],
                "has_seg": modality_flags[patient_id]["has_seg"],
                "has_aim": 1 if patient_id in aim_cases else 0,
                "has_rnaseq": 1 if patient_id in case_to_gsm else 0,
                "gsm_id": case_to_gsm.get(patient_id, ""),
                "event_os": event_os,
                "time_os": "" if time_os is None else time_os,
                "os_label_known": os_label_known,
                "event_rec": event_rec,
                "time_rec": "" if time_rec is None else time_rec,
                "rec_label_known": rec_label_known,
                "rec_location_class": rec_location_class,
                "ct_date": "" if ct_date is None else ct_date.isoformat(),
                "pet_date": "" if pet_date is None else pet_date.isoformat(),
                "date_last_known_alive": "" if date_last_alive is None else date_last_alive.isoformat(),
                "date_death": "" if date_death is None else date_death.isoformat(),
                "date_recurrence": "" if date_recurrence is None else date_recurrence.isoformat(),
                "survival_status": survival_status,
            }
            output_rows.append(manifest_row)

    output_rows.sort(key=lambda x: str(x["patient_id"]))
    return output_rows


def write_manifest(rows):
    """Why: 把内存中的 manifest 结果落盘，方便后续流程直接读取。

    Content: 按固定字段顺序写 CSV。
    Input: rows（build_patient_manifest 产出的行列表）。
    Output: 在 OUTPUT_CSV 写出 patient_manifest.csv。
    """
    if not rows:
        raise RuntimeError("No rows generated for patient manifest")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "has_ct",
        "has_pet",
        "has_seg",
        "has_aim",
        "has_rnaseq",
        "gsm_id",
        "event_os",
        "time_os",
        "os_label_known",
        "event_rec",
        "time_rec",
        "rec_label_known",
        "rec_location_class",
        "ct_date",
        "pet_date",
        "date_last_known_alive",
        "date_death",
        "date_recurrence",
        "survival_status",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    """Why: 快速确认生成结果是否符合预期规模。

    Content: 打印关键字段计数（模态可用性与标签可用性）。
    Input: rows（manifest 行列表）。
    Output: 终端摘要文本。
    """
    n = len(rows)
    has = lambda col: sum(1 for r in rows if int(r[col]) == 1)  # noqa: E731
    print(f"wrote: {OUTPUT_CSV}")
    print(f"rows: {n}")
    print(f"has_ct: {has('has_ct')}")
    print(f"has_pet: {has('has_pet')}")
    print(f"has_seg: {has('has_seg')}")
    print(f"has_aim: {has('has_aim')}")
    print(f"has_rnaseq: {has('has_rnaseq')}")
    print(f"os_label_known: {has('os_label_known')}")
    print(f"rec_label_known: {has('rec_label_known')}")


def main():
    """Why: 提供一键执行入口，减少手动调用步骤。

    Content: 依次构建 manifest、写文件、打印摘要。
    Input: 无（使用模块常量指定的数据路径）。
    Output: 生成 patient_manifest.csv，并输出统计信息。
    """
    clinical_csv_path = resolve_input_path(
        PRIMARY_CLINICAL_CSV,
        LEGACY_CLINICAL_CSV,
        "clinical csv",
    )
    series_matrix_path = resolve_input_path(
        PRIMARY_SERIES_MATRIX_TXT,
        LEGACY_SERIES_MATRIX_TXT,
        "series matrix",
    )
    print(f"using_clinical_csv: {clinical_csv_path}")
    print(f"using_series_matrix: {series_matrix_path}")
    rows = build_patient_manifest(clinical_csv_path, series_matrix_path)
    write_manifest(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
