"""这个文件用于做 Label construction 和 time zero。

作用：
- 以 CT Date 作为 t0（time zero）。
- 构建 OS 标签（event_os, time_os_days）。
- 构建复发标签（event_rec, time_rec_days, rec_location_class）。

输入：
- output/clean_data/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv（优先）
- data/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv（回退）

输出：
- output/labels_time_zero.csv
"""

from __future__ import annotations

import csv
from datetime import datetime, date
from pathlib import Path


PRIMARY_INPUT_CLINICAL = Path("output/clean_data/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")
LEGACY_INPUT_CLINICAL = Path("data/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")
OUTPUT_LABELS = Path("output/labels_time_zero.csv")

MISSING = {"", "n/a", "na", "not collected", "not recorded in database", "null"}


def clean(value):
    """Why: 统一处理缺失值文本，避免每个字段重复写判断逻辑。

    Content: 清理字符串两端空格，并把缺失语义文本转成空字符串。
    Input: value（任意字符串或 None）。
    Output: 清理后的字符串；若视为缺失则返回空字符串。
    """
    if value is None:
        return ""
    text = value.strip()
    return "" if text.lower() in MISSING else text


def resolve_input_path(primary_path, fallback_path, label):
    """Why: 兼容 output/data 两种目录布局，避免路径迁移后脚本失效。

    Content: 优先用 primary_path；不存在时回退 fallback_path；都不存在则报错。
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
    """Why: 标签构建依赖日期差，需要先把日期文本稳定解析成 date。

    Content: 先做缺失清理，再按两种日期格式解析；失败返回 None。
    Input: value（日期字符串或 None）。
    Output: date 对象或 None。
    """
    text = clean(value)
    if not text:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def day_diff(start, end):
    """Why: OS/复发时间标签统一用“事件日期 - t0”计算。

    Content: 计算非负天数差；缺失或负值返回 None。
    Input: start（起始日期），end（结束日期）。
    Output: 天数整数或 None。
    """
    if start is None or end is None:
        return None
    d = (end - start).days
    if d < 0:
        return None
    return d


def build_rows(clinical_path):
    """Why: 把原始临床表转换成可直接训练的 time-zero 标签表。

    Content: 以 CT Date 作为 t0，构建 OS 与复发的 event/time/known 字段。
    Input: clinical_path（临床 CSV 路径）。
    Output: 标签行列表（每行是一个 patient 的字典）。
    """
    rows = []
    with clinical_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            patient_id = r["Case ID"].strip()
            t0 = parse_date(r.get("CT Date"))
            last_alive = parse_date(r.get("Date of Last Known Alive"))
            death = parse_date(r.get("Date of Death"))
            recurrence_date = parse_date(r.get("Date of Recurrence"))

            # OS label
            survival = clean(r.get("Survival Status")).lower()
            if survival == "dead":
                event_os = 1
                os_label_known = 1
                os_anchor = death if death is not None else last_alive
            elif survival == "alive":
                event_os = 0
                os_label_known = 1
                os_anchor = last_alive
            else:
                event_os = ""
                os_label_known = 0
                os_anchor = None
            time_os = day_diff(t0, os_anchor)

            # Recurrence label
            rec = clean(r.get("Recurrence")).lower()
            rec_location = clean(r.get("Recurrence Location")).lower()
            if rec == "yes":
                event_rec = 1
                rec_label_known = 1
                rec_location_class = rec_location
                time_rec = day_diff(t0, recurrence_date)
                rec_censored = 0
            elif rec == "no":
                event_rec = 0
                rec_label_known = 1
                rec_location_class = ""
                time_rec = day_diff(t0, last_alive)
                rec_censored = 1
            else:
                event_rec = ""
                rec_label_known = 0
                rec_location_class = ""
                time_rec = None
                rec_censored = ""

            rows.append(
                {
                    "patient_id": patient_id,
                    "t0_ct_date": "" if t0 is None else t0.isoformat(),
                    "event_os": event_os,
                    "time_os_days": "" if time_os is None else time_os,
                    "os_label_known": os_label_known,
                    "event_rec": event_rec,
                    "time_rec_days": "" if time_rec is None else time_rec,
                    "rec_label_known": rec_label_known,
                    "rec_censored": rec_censored,
                    "rec_location_class": rec_location_class,
                    "date_of_recurrence": "" if recurrence_date is None else recurrence_date.isoformat(),
                    "date_of_last_known_alive": "" if last_alive is None else last_alive.isoformat(),
                    "date_of_death": "" if death is None else death.isoformat(),
                }
            )
    rows.sort(key=lambda x: str(x["patient_id"]))
    return rows


def write_rows(rows):
    """Why: 固化标签结果，供后续训练代码稳定读取。

    Content: 按固定列顺序把标签行写入 CSV。
    Input: rows（build_rows 生成的标签行列表）。
    Output: 在 OUTPUT_LABELS 位置写出 labels_time_zero.csv。
    """
    if not rows:
        raise RuntimeError("No label rows generated.")
    OUTPUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "t0_ct_date",
        "event_os",
        "time_os_days",
        "os_label_known",
        "event_rec",
        "time_rec_days",
        "rec_label_known",
        "rec_censored",
        "rec_location_class",
        "date_of_recurrence",
        "date_of_last_known_alive",
        "date_of_death",
    ]
    with OUTPUT_LABELS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Why: 提供一键执行入口，避免手动拼步骤。

    Content: 依次构建标签、写文件、打印摘要统计。
    Input: 无（读取模块常量定义的输入文件）。
    Output: 生成标签 CSV，并在终端输出统计信息。
    """
    clinical_path = resolve_input_path(
        PRIMARY_INPUT_CLINICAL,
        LEGACY_INPUT_CLINICAL,
        "clinical csv",
    )
    print(f"using_clinical_csv: {clinical_path}")
    rows = build_rows(clinical_path)
    write_rows(rows)
    n = len(rows)
    print(f"wrote: {OUTPUT_LABELS}")
    print(f"rows: {n}")
    print(f"os_label_known: {sum(int(r['os_label_known']) for r in rows)}")
    print(f"rec_label_known: {sum(int(r['rec_label_known']) for r in rows)}")


if __name__ == "__main__":
    main()
