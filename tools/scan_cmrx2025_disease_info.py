"""Summarise CMRxRecon2025's per-subject disease/demographic table and join it to what we extracted.

Source: `_provenance/CMRxRecon2025_TaskR1_TaskR2_Disease_Info.xlsx` (pulled from the V2 Drive
folder). Five sheets -- TrainingSet, {Validation,Test}Set_Task{R1,R2} -- keyed by
Center / Manufacturer / Split / AnonPatientID, i.e. exactly the center+scanner+P### key the
recon has to use (2025 reuses patient IDs across centres).

Parsed with zipfile+ElementTree rather than pandas/openpyxl: openpyxl is not in the `svr` env and
an xlsx is just a zip of XML, so this needs no new dependency.

Usage:
    python tools/scan_cmrx2025_disease_info.py
    python tools/scan_cmrx2025_disease_info.py --csv-out /tmp/cmrx2025_subjects.csv
"""

import argparse
import csv as csvmod
import glob
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "scratch", "data", "CMRxRecon2025")
XLSX = os.path.join(ROOT, "_provenance", "CMRxRecon2025_TaskR1_TaskR2_Disease_Info.xlsx")


def read_xlsx(path):
    """-> {sheet_name: [row_dicts]} using the first row as the header."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
    names = [s.get("name") for s in ET.fromstring(z.read("xl/workbook.xml")).iter(f"{NS}sheet")]
    out = {}
    for i, name in enumerate(names, 1):
        member = f"xl/worksheets/sheet{i}.xml"
        if member not in z.namelist():
            continue
        rows = []
        for row in ET.fromstring(z.read(member)).iter(f"{NS}row"):
            cells = {}
            for c in row.findall(f"{NS}c"):
                col = re.match(r"[A-Z]+", c.get("r")).group()
                v = c.find(f"{NS}v")
                if v is None:
                    inline = c.find(f"{NS}is")
                    val = "".join(x.text or "" for x in inline.iter(f"{NS}t")) if inline is not None else ""
                else:
                    val = shared[int(v.text)] if c.get("t") == "s" else v.text
                cells[col] = (val or "").strip()
            rows.append(cells)
        if not rows:
            continue
        hdr = rows[0]
        out[name] = [{hdr.get(k, k): v for k, v in r.items()} for r in rows[1:]]
    return out


def diseases_of(row):
    ds = [row.get(f"Disease{i}", "") for i in range(1, 6)]
    return [d.strip().lower() for d in ds if d and d.strip()]


def label_of(row):
    """healthy / unknown / diseased. The healthy label is a PREFIX match on purpose --
    the sheets spell it several ways ('normal control, healthy', '... healthy volunteer')."""
    ds = diseases_of(row)
    if not ds:
        return "blank"
    if all(d.startswith("normal control") for d in ds):
        return "healthy"
    if all(d == "unknown" for d in ds):
        return "unknown"
    return "diseased"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-out", default="/tmp/cmrx2025_subjects.csv")
    args = ap.parse_args()

    sheets = read_xlsx(XLSX)
    print(f"sheets: { {k: len(v) for k, v in sheets.items()} }\n")

    all_rows = []
    for sheet, rows in sheets.items():
        centers = sorted({r.get("Center", "") for r in rows})
        scanners = sorted({r.get("Manufacturer", "") for r in rows})
        dc = Counter(d for r in rows for d in diseases_of(r))
        lab = Counter(label_of(r) for r in rows)
        print(f"=== {sheet}: {len(rows)} subjects ===")
        print(f"  centers ({len(centers)}): {centers}")
        print(f"  scanners ({len(scanners)}): {scanners}")
        print(f"  labels: {dict(lab)}")
        print(f"  top diagnoses: {dc.most_common(8)}")
        print()
        for r in rows:
            r["_sheet"] = sheet
            all_rows.append(r)

    print(f"TOTAL subjects in table: {len(all_rows)}")
    print(f"centers overall : {len(sorted({r.get('Center','') for r in all_rows}))}")
    print(f"scanners overall: {len(sorted({r.get('Manufacturer','') for r in all_rows}))}")

    # ---- join against what is actually extracted on disk ----
    on_disk = set()
    for mat in glob.glob(os.path.join(ROOT, "*_extracted", "**", "cine_sax.mat"), recursive=True):
        parts = mat.split("/")
        on_disk.add((parts[-4], parts[-3], parts[-2]))  # center, scanner, P###
    in_table = {(r.get("Center", ""), r.get("Manufacturer", ""), r.get("AnonPatientID", "")): r
                for r in all_rows}

    print(f"\n--- join: extracted SAX vs disease table ---")
    print(f"extracted (center,scanner,P###) : {len(on_disk)}")
    print(f"table rows                      : {len(in_table)}")
    matched = on_disk & set(in_table)
    print(f"matched                         : {len(matched)}")
    print(f"extracted but NOT in table      : {len(on_disk - set(in_table))}")
    print(f"in table but NOT extracted      : {len(set(in_table) - on_disk)}  "
          f"(expected: table covers all views, we extracted SAX only)")

    lbl = Counter()
    for k in sorted(matched):
        lbl[label_of(in_table[k])] += 1
    print(f"\nlabels for the {len(matched)} extracted SAX subjects: {dict(lbl)}")

    by_center = defaultdict(Counter)
    for k in sorted(matched):
        by_center[k[0]][label_of(in_table[k])] += 1
    print("\nper-center label mix (extracted SAX only):")
    for c in sorted(by_center):
        print(f"  {c}: {dict(by_center[c])}")

    with open(args.csv_out, "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(["center", "scanner", "pid", "sheet", "extracted_sax", "age", "gender", "diseases"])
        for k, r in sorted(in_table.items()):
            w.writerow([k[0], k[1], k[2], r["_sheet"], k in on_disk,
                        r.get("Age", ""), r.get("Gender", ""), "; ".join(diseases_of(r))])
    print(f"\ncsv -> {args.csv_out}")


if __name__ == "__main__":
    main()
