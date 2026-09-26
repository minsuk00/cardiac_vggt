"""Paper tables for the af12 (main) / hrv12 (appendix) arms: one row per method, one column per metric.

Sources (all read-only):
  image      evaluation/metric_results/test/<src>_<arm>/<method>.json       (unit-peak PSNR, SSIM, NCC; heart ROI)
  function   evaluation/metric_results/test/<src>_<arm>/ef/<method>.json    (EDV/ESV/EF/LVM MAE, VTC error)
  Dice/HD95  _paper_figures/seg_allphase_dice_hd95.py output                         (mean over LV, MYO, RV x all 12 phases)
  Resp. EPE  common_set.py (evaluation/src/analysis/motion_epe/) re-run on _motion_epe/<arm>/*.json slice dumps
             at full precision (its dz block = the checked-in common_set_9methods.txt at 2 decimals; dz, demeaned, mm)
  runtime    docs/120 §6 (af12, s per subject = full 12-phase cine; GPU 4x L40S, CPU 16 cores Xeon Gold 6154);
             the three 4-GPU arms (Ours, Dangi, CiNeVol) recomputed from their per-subject timing files, n = 180
  Disp.      ablation tables: render_ablation_motion.subject_deform (mean in-plane |delta| in the heart ROI after
             removing each slice's mean shift, ED target), mean over subjects

VTC error = per subject mean_t |V_pred(t) - V_gt(t)| / EDV_gt x 100 (stored lv_curve_nmae_breath).
Pooled means over subjects; Dice/HD95 pooled over (subject, structure, phase), NaN skipped (as ef_dice.py).
Numbers print with 4 digits, a leading 0 counted (0.883, 7.490, 24.50, 167.4), as in the paper tables.

    python _paper_figures/paper_results_table.py --seg <seg_allphase.json> [--json out.json]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/evaluation/metric_results/test"
SRCS = ["acdc", "cmrx2023", "cmrx2024", "cmrx2025", "mnms"]
# display name -> (af12 arm, hrv12 arm, EPE row name af12, EPE row name hrv12)
BASELINES = {
    "SVRTK": ("svrtk3d_debug_scatter", "svrtk3d_debug_scatter", "svrtk3d", "svrtk3d"),
    "NiftyMIC": ("niftymic_scatter", "niftymic_scatter", "niftymic", "niftymic"),
    "NeSVoR": ("nesvor_4gpu_scatter", "nesvor_scatter", "nesvor", "nesvor"),
    "Dangi": ("dangi_scatter_4gpu", "dangi_scatter", None, None),
    "Fetal CMR 4D†": ("fetal_cmr_4d", "fetal_cmr_4d", "fetal_cmr_4d", "fetal_cmr_4d"),
    "CiNeVol†": ("cinevol_motion_masked", "cinevol_masked", "cinevol_motion", "cinevol"),
}
VGGT = {
    "Ours (diff1000)": "vggt_final518_diff1000_ep300",
    "base": "vggt_final518_base_ep300",
    "nogather": "vggt_final518_nogather_ep300",
    "hw0": "vggt_final518_hw0_ep300",
}
# docs/120 §6: af12 seconds per subject (full cine). NeSVoR = the 126 gl1706 subjects (docs/120 headline). The
# CPU methods' numbers are the 6154-node subsets (docs/120 §6c), so they stay as recorded there.
RUNTIME_AF12 = {"SVRTK": 167.4, "NiftyMIC": 232.5, "NeSVoR": 323.1, "Fetal CMR 4D†": 517.0}
# 4-GPU timing arms (docs/120 §6) -> per-subject seconds: (arm dir, reader)
RUNTIME_FILES = {
    "Ours (diff1000)": ("vggt_final518_diff1000_ep300_4gpu_v2/timing.json", lambda t: json.loads(t)["breath"]["total_sec"]),
    "Dangi": ("dangi_scatter_4gpu/timing.json", lambda t: json.loads(t)["breath"]["total_sec"]),
    "CiNeVol†": ("cinevol_4gpu/recon_breath/provenance.txt",       # stored to 0.1 s per subject
                 lambda t: float(re.search(r"total ([0-9.]+)", t).group(1))),
}
COLS = [("NCC", 1), ("PSNR", 1), ("SSIM", 1), ("Resp. EPE", -1), ("EDV", -1), ("ESV", -1), ("EF", -1), ("LVM", -1),
        ("VTC err.", -1), ("Dice", 1), ("HD95", -1), ("Runtime (s)", -1), ("Disp.", 0)]


def ok(v):
    return v is not None and np.isfinite(v)


def four(v):
    """4 digits counting a leading 0: 0.883, 7.490, 24.50, 167.4."""
    ints = len(str(int(abs(v)))) if abs(v) >= 1 else 1
    return f"{v:.{max(4 - ints, 0)}f}"


def epe_rows(arm):
    """Full-precision common-set dz EPE: common_set.py on the arm's per-method slice dumps -> {row: demeaned EPE}."""
    d = f"{RES}/_motion_epe/{arm}"
    dumps = sorted(p for p in glob.glob(f"{d}/*.json") if not p.endswith("rows.json"))
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        subprocess.run([sys.executable, f"{ROOT}/evaluation/src/analysis/motion_epe/common_set.py", *dumps,
                        "--json", f.name], check=True, stdout=subprocess.DEVNULL)
        return {k: v["epe_demeaned"] for k, v in json.load(open(f.name))["dz"].items()}


def runtimes():
    out = dict(RUNTIME_AF12)
    for n, (path, read) in RUNTIME_FILES.items():
        v = [read(open(p).read()) for p in glob.glob(f"{ROOT}/evaluation/volumes/*_af12/out/*/{path}")]
        assert len(v) == 180, (n, len(v))
        out[n] = float(np.mean(v))
    return out


def _deform(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from render_ablation_motion import subject_deform
    return subject_deform(*args)


def disp(arm, method):
    """Mean over subjects of the per-subject heart in-plane deformation (render_ablation_motion.subject_deform)."""
    subs = sorted(glob.glob(f"{ROOT}/evaluation/volumes/*_{arm}/out/*"))
    with Pool(16) as p:
        v = p.map(_deform, [(s, method.removeprefix("vggt_final518_").removesuffix("_ep300")) for s in subs])
    return float(np.mean(v))


def row(arm, method, seg, epe_key, epe):
    img, ef = [], []
    for s in SRCS:
        img += json.load(open(f"{RES}/{s}_{arm}/{method}.json"))["per_subject"]
        ef += json.load(open(f"{RES}/{s}_{arm}/ef/{method}.json"))["per_subject"]
    mean = lambda xs: float(np.mean([x for x in xs if ok(x)]))
    err = lambda k: mean([abs(r[f"{k}_breath"] - r[f"{k}_gt"]) for r in ef
                          if ok(r.get(f"{k}_breath")) and ok(r.get(f"{k}_gt"))])
    d, h = [], []
    for r in seg:
        if r["arm"] == arm and method in r["m"]:
            for st in ("LV", "MYO", "RV"):
                d += r["m"][method][f"dice_{st}"]
                h += r["m"][method][f"hd95_{st}"]
    return {"NCC": mean([r["breath_ncc"] for r in img]), "PSNR": mean([r["breath_psnr_unit_peak"] for r in img]),
            "SSIM": mean([r["breath_ssim"] for r in img]), "Resp. EPE": epe.get(epe_key) if epe_key else None,
            "EDV": err("edv"), "ESV": err("esv"), "EF": err("ef"), "LVM": err("lvm"),
            "VTC err.": mean([r.get("lv_curve_nmae_breath") for r in ef]),
            "Dice": mean(d), "HD95": mean(h), "n_img": len(img), "n_ef": len(ef)}


def table(arm, seg, names, runtime):
    epe = epe_rows(arm)
    rows = {}
    for n in names:
        if n in BASELINES:
            a = BASELINES[n]
            rows[n] = row(arm, a[0] if arm == "af12" else a[1], seg, a[2] if arm == "af12" else a[3], epe)
        else:
            rows[n] = row(arm, VGGT[n], seg, VGGT[n], epe)
        rows[n]["Runtime (s)"] = runtime.get(n)
    return rows


def show(title, rows, cols):
    best = {}
    for c, sgn in cols:
        vs = [r[c] for r in rows.values() if r.get(c) is not None]
        if vs and sgn:
            best[c] = max(vs) if sgn > 0 else min(vs)
    print(f"\n### {title}\n")
    print("| method | " + " | ".join(c for c, _ in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for n, r in rows.items():
        cells = []
        for c, _ in cols:
            v = r.get(c)
            s = "--" if v is None else four(v)
            cells.append(f"**{s}**" if v is not None and v == best.get(c) else s)
        print(f"| {n} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True, help="_paper_figures/seg_allphase_dice_hd95.py output")
    ap.add_argument("--json")
    a = ap.parse_args()
    seg = json.load(open(a.seg))
    main_names = ["SVRTK", "NiftyMIC", "NeSVoR", "Dangi", "Fetal CMR 4D†", "CiNeVol†", "Ours (diff1000)"]
    out = {"af12_main": table("af12", seg, main_names, runtimes()),
           "hrv12_appendix": table("hrv12", seg, main_names, {}),
           "af12_ablation": table("af12", seg, list(VGGT), {}),
           "hrv12_ablation": table("hrv12", seg, list(VGGT), {})}
    for k in ("af12_ablation", "hrv12_ablation"):
        for n, r in out[k].items():
            r["Disp."] = disp(k.split("_")[0], VGGT[n])
    main_cols = [c for c in COLS if c[0] != "Disp."]
    show("af12 — main (Tables 2, 3)", out["af12_main"], main_cols)
    show("hrv12 — main (Tables 2, 3)", out["hrv12_appendix"], [c for c in main_cols if c[0] != "Runtime (s)"])
    abl = [c for c in COLS if c[0] in ("PSNR", "SSIM", "VTC err.", "Dice", "Resp. EPE", "Disp.")]
    show("af12 — ablation (Table 4)", out["af12_ablation"], abl)
    show("hrv12 — ablation (Table 5)", out["hrv12_ablation"], abl)
    for k, rows in out.items():
        print(k, "n_img/n_ef:", {n: (r["n_img"], r["n_ef"]) for n, r in rows.items()})
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
