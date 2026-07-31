"""Build the pooled multi-dataset manifest CSV (docs/58 sec 2.1, TODO item 5).

One row per subject across the 5 pooling sources (CMRxRecon2023/2024/2025, ACDC, M&Ms-1).
Geometry (n_z/pitch_mm/num_phases) is read directly off each subject's own converted
`sax/3d_recon/sax_frame_00.nii.gz` on disk — not from any provenance JSON/CSV — so the
manifest can never drift from what `MRIDataset` actually loads.

Demographic/scanner metadata is joined per-source from whatever ground truth exists for
that source (see docs/58 sec 2 + the per-dataset README files); fields that are genuinely
undocumented for a source are left blank rather than guessed. In particular
CMRxRecon2025 is NOT single-vendor/healthy like 2023/2024 — it is joined against
`_provenance/CMRxRecon2025_TaskR1_TaskR2_Disease_Info.xlsx` (see
tools/scan_cmrx2025_disease_info.py, which this script imports) and turns out to be
majority-diseased (317 diseased / 34 healthy / 7 unknown / 1 unmatched of 359 SAX subjects)
across 3 vendors (Siemens/UIH/Philips) and 8 centres. Philips (12 subjects) appears only in
the challenge's own official test splits — zero in the on-disk train/val subjects.

`split` (the pooled train/val/test assignment) is filled in by the separate
`tools/build_pooled_split.py` (docs/58 sec 13), which reads this manifest and writes the
column back in place — this script only produces the manifest with `split` blank.

Usage:
    python tools/build_manifest.py [--out training/splits/manifest.csv]
"""

import argparse
import csv
import json
import os
import re

import nibabel as nib

import scan_cmrx2025_disease_info as cmrx25_disease

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO, "scratch", "data")

COLUMNS = [
    "id", "source", "rel_path", "official_split",
    "n_z", "pitch_mm", "z_extent_mm", "T_native", "num_phases",
    "vendor", "scanner_model", "field_strength_t", "centre",
    "pathology_label", "pathology_detail",
    "age", "sex", "height_cm", "weight_kg",
    "ed", "es", "split", "source_file",
]

CMRX25_DIR_RE = re.compile(r"^CMRx25_(?P<split>[A-Za-z0-9]+)_(?P<centre>Center\d+)_(?P<scanner>.+)_P(?P<pid>\d+)$")
CMRX_DIR_RE = re.compile(r"^CMRx\d\d_(?P<split>[A-Za-z]+)_P\d+")


def read_geometry(sax_dir):
    """(n_z, pitch_mm, num_phases) read straight off the on-disk NIfTI stack."""
    recon_dir = os.path.join(sax_dir, "3d_recon")
    frames = sorted(f for f in os.listdir(recon_dir) if f.startswith("sax_frame_"))
    img = nib.load(os.path.join(recon_dir, frames[0]))
    n_z = int(img.shape[2])
    pitch_mm = float(img.header.get_zooms()[2])
    return n_z, pitch_mm, len(frames)


def read_field_strength(sax_dir):
    path = os.path.join(sax_dir, "cine_sax_info.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for row in csv.reader(f):
            if row and row[0] == "FieldStrength":
                return float(row[1])
    return None


def rows_cmrx(source, subdir):
    root = os.path.join(DATA_ROOT, source, "Cine_combined")
    disease_by_key = None
    if source == "CMRxRecon2025":
        sheets = cmrx25_disease.read_xlsx(cmrx25_disease.XLSX)
        disease_by_key = {}
        for rows in sheets.values():
            for r in rows:
                key = (r.get("Center", ""), r.get("Manufacturer", ""), r.get("AnonPatientID", ""))
                disease_by_key[key] = r

    out = []
    for subj_id in sorted(os.listdir(root)):
        sax_dir = os.path.join(root, subj_id, "sax")
        if not os.path.isdir(os.path.join(sax_dir, "3d_recon")):
            continue
        n_z, pitch_mm, num_phases = read_geometry(sax_dir)
        fs = read_field_strength(sax_dir)
        row = {c: "" for c in COLUMNS}
        row.update({
            "id": subj_id,
            "source": source,
            "rel_path": os.path.relpath(os.path.join(root, subj_id), DATA_ROOT),
            "n_z": n_z, "pitch_mm": pitch_mm, "z_extent_mm": (n_z - 1) * pitch_mm,
            "T_native": 12, "num_phases": num_phases,
            "field_strength_t": fs if fs is not None else "",
            "ed": 0,  # ED-anchored convention (docs/58 sec 7) — always frame 0 for CMRx
            "es": "",  # needs cardiac_phase.csv regeneration (docs/58 sec 8.4) — blocked
        })

        if source == "CMRxRecon2023":
            row.update(vendor="Siemens", scanner_model="MAGNETOM Vida", centre="Fudan",
                       pathology_label="healthy", pathology_detail="healthy")
            m = CMRX_DIR_RE.match(subj_id)
            row["official_split"] = m.group("split") if m else ""
        elif source == "CMRxRecon2024":
            # README: "single center", vendor inferred from cine_sax_info SoftwareVersion
            # ("syngo MR ..." = Siemens); neither the specific model nor the centre name
            # is documented anywhere on disk, so both stay blank rather than guessed.
            row.update(vendor="Siemens", pathology_label="healthy", pathology_detail="healthy")
            m = CMRX_DIR_RE.match(subj_id)
            row["official_split"] = m.group("split") if m else ""
        elif source == "CMRxRecon2025":
            m = CMRX25_DIR_RE.match(subj_id)
            if m:
                row.update(centre=m.group("centre"), scanner_model=m.group("scanner"),
                           official_split=m.group("split"))
                vendor = m.group("scanner").split("_")[0]
                row["vendor"] = vendor
                key = (m.group("centre"), m.group("scanner"), f"P{m.group('pid')}")
                d = disease_by_key.get(key)
                if d is not None:
                    label = cmrx25_disease.label_of(d)
                    row["pathology_label"] = "" if label == "blank" else label
                    row["pathology_detail"] = "; ".join(cmrx25_disease.diseases_of(d))
                    row["age"] = d.get("Age", "")
                    row["sex"] = d.get("Gender", "")
        out.append(row)
    return out


def read_mnms_demographics():
    """{external_code: row} from the official M&Ms CSV (age/sex/height/weight; vendor/
    centre/pathology/ed/es come from convert_meta.json instead, already join-verified by
    the converter)."""
    path = os.path.join(DATA_ROOT, "MNMs", "MNMs1",
                         "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    with open(path) as f:
        return {row["External code"]: row for row in csv.DictReader(f)}


def rows_converted(source, subdir_name):
    root = os.path.join(DATA_ROOT, subdir_name)
    mnms_demo = read_mnms_demographics() if source == "MNMs" else None
    out = []
    for subj_id in sorted(os.listdir(root)):
        sax_dir = os.path.join(root, subj_id, "sax")
        meta_path = os.path.join(sax_dir, "convert_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        n_z, pitch_mm, num_phases = read_geometry(sax_dir)
        row = {c: "" for c in COLUMNS}
        row.update({
            "id": subj_id, "source": source,
            "rel_path": os.path.relpath(os.path.join(root, subj_id), DATA_ROOT),
            "official_split": meta.get("official_split", ""),
            "n_z": n_z, "pitch_mm": pitch_mm, "z_extent_mm": (n_z - 1) * pitch_mm,
            "T_native": meta.get("native_T", ""), "num_phases": num_phases,
            "ed": 0,  # ED-anchored (frame_indices_native[0] == ed_native by construction)
            "es": meta.get("es_frame_on_12grid_advisory", ""),
            "source_file": meta.get("source_file", ""),
        })

        if source == "ACDC":
            group = meta.get("group", "")
            row.update(vendor="Siemens", centre="Dijon",
                       pathology_label="healthy" if group == "NOR" else "diseased",
                       pathology_detail=group)
            cfg_path = os.path.join(DATA_ROOT, os.path.dirname(meta["source_file"]), "Info.cfg")
            if os.path.exists(cfg_path):
                cfg = {}
                for line in open(cfg_path):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        cfg[k.strip()] = v.strip()
                row["height_cm"] = cfg.get("Height", "")
                row["weight_kg"] = cfg.get("Weight", "")
        elif source == "MNMs":
            pathology = meta.get("pathology", "")
            row.update(vendor=meta.get("vendor", ""), centre=meta.get("centre", ""),
                       pathology_label="healthy" if pathology == "NOR" else "diseased",
                       pathology_detail=pathology)
            code = subj_id[len("MNMs_"):]
            demo = mnms_demo.get(code)
            if demo is not None:
                row.update(age=demo.get("Age", ""), sex=demo.get("Sex", ""),
                           height_cm=demo.get("Height", ""), weight_kg=demo.get("Weight", ""))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "training", "splits", "manifest.csv"))
    args = ap.parse_args()

    all_rows = []
    all_rows += rows_cmrx("CMRxRecon2023", "CMRxRecon2023")
    all_rows += rows_cmrx("CMRxRecon2024", "CMRxRecon2024")
    all_rows += rows_cmrx("CMRxRecon2025", "CMRxRecon2025")
    all_rows += rows_converted("ACDC", "ACDC_sax")
    all_rows += rows_converted("MNMs", "MNMs_sax")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(all_rows)

    by_source = {}
    for r in all_rows:
        by_source.setdefault(r["source"], 0)
        by_source[r["source"]] += 1
    print(f"wrote {len(all_rows)} rows -> {args.out}")
    for s, n in by_source.items():
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
