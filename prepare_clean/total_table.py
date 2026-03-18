
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

    if value is None:
        return ""
    cleaned = value.strip()
    if cleaned.lower() in MISSING_STRINGS:
        return ""
    return cleaned

def resolve_input_path(primary_path, fallback_path, label):

    if primary_path.exists():
        return primary_path
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(
        f"{label} not found in either '{primary_path}' or '{fallback_path}'"
    )

def parse_date(value):

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

    if start is None or end is None:
        return None
    delta = (end - start).days
    if delta < 0:
        return None
    return delta

def load_modality_flags():

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

    out = set()
    for xml_file in AIM_DIR.glob("*.xml"):
        case_id = re.sub(r"v\d+$", "", xml_file.stem)
        out.add(case_id)
    return out

def load_rna_mapping(series_matrix_path):

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

    n = len(rows)
    has = lambda col: sum(1 for r in rows if int(r[col]) == 1)
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
