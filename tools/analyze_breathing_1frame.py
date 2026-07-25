"""Breathing-estimation deep-dive across models, from artifacts run_vggt.py already wrote.

No GPU. Sources, per <dataset>/out/<subject>/<method>/:
  resp_diag.json  -> per-slot predicted vs applied Dz (mm) at ED, for BOTH a breath arm and a
                     clean arm (applied == 0 on every slot = negative control).
  ed_dvf.npz      -> delta (S,518,518,3) normalized; lets us ask whether the predicted Dz is a
                     rigid slab shift (low within-slot std) or a smear (high std).
  metrics.json    -> clean/breath PSNR -> the breathing COST in dB, i.e. what estimation is FOR.

Why more than slope/corr/EPE: those are saturated for 5 of 6 runs and collapse three different
failure modes (bias, amplitude under-correction, deep-breath collapse) into one number.

Slot 0 is the reference anchor and IS included by the trainer metric and by resp_diag; we report
with and without it, because tools/exp_4wok_analysis.py excludes it and the two are otherwise
not comparable.
"""
import glob
import json
import os
import sys
import numpy as np

MM_PER_NORM_Z = (12 - 1) / 2.0 * 12.0  # 66 mm per normalized z unit; matches loss.py:624
BINS = [(0, 2), (2, 8), (8, 12), (12, 40)]  # applied |SI| mm; bin edges from tools/exp_4wok_analysis.py


def collect(dataset, method):
    """Per-subject records for one (dataset, method)."""
    root = f"/home/minsukc/vggt/scratch/eval/{dataset}/out"
    recs = []
    for f in sorted(glob.glob(f"{root}/*/{method}/resp_diag.json")):
        d = os.path.dirname(f)
        subj = f.split("/")[-3]
        rd = json.load(open(f))
        try:
            mm = json.load(open(os.path.join(d, "metrics.json")))
        except FileNotFoundError:
            mm = None
        rec = {"subject": subj, "method": method, "dataset": dataset,
               "pred": np.array(rd["breath"]["pred_dz_mm"], dtype=float),
               "appl": np.array(rd["breath"]["applied_dz_mm"], dtype=float),
               "clean_pred": np.array(rd["clean"]["pred_dz_mm"], dtype=float),
               "breath_epe": rd["breath"].get("epe_dz_mm"),
               "clean_epe": rd["clean"].get("epe_dz_mm"),
               "slope": rd["breath"].get("slope"), "corr": rd["breath"].get("corr")}
        if mm:
            rec["cost_db"] = mm["clean_psnr_mean"] - mm["breath_psnr_mean"]
            rec["clean_psnr"] = mm["clean_psnr_mean"]
            rec["breath_psnr"] = mm["breath_psnr_mean"]
            rec["max_disp_mm"] = mm["breath_max_disp_mm"]
        npz = os.path.join(d, "ed_dvf.npz")
        if os.path.exists(npz):
            z = np.load(npz)
            dz = z["delta"][..., 2].astype(np.float32) * MM_PER_NORM_Z  # (S,518,518) mm
            rec["within_slot_std_mm"] = float(dz.reshape(dz.shape[0], -1).std(axis=1).mean())
        recs.append(rec)
    return recs


def fit(x, y):
    """OLS slope/corr. x = applied (EXACT sim GT, zero measurement error) => the slope is
    unbiased regardless of noise in y; slope<1 is real under-correction, not attenuation."""
    if len(x) < 2 or np.std(x) < 1e-6:
        return None, None
    return float(np.polyfit(x, y, 1)[0]), float(np.corrcoef(x, y)[0, 1])


def summarize(recs, drop_slot0=False):
    P, A = [], []
    for r in recs:
        p, a = r["pred"], r["appl"]
        if drop_slot0 and len(p) > 1:
            p, a = p[1:], a[1:]
        P.append(p); A.append(a)
    p, a = np.concatenate(P), np.concatenate(A)
    sl, co = fit(a, p)
    out = {"n_subj": len(recs), "n_slots": len(p), "slope": sl, "corr": co,
           "epe_mm": float(np.abs(p - a).mean()),
           "bias_mm": float((p - a).mean())}
    out["bins"] = []
    for lo, hi in BINS:
        m = (np.abs(a) >= lo) & (np.abs(a) < hi)
        out["bins"].append({"bin": f"[{lo},{hi})", "n": int(m.sum()),
                            "applied_mean": float(np.abs(a[m]).mean()) if m.any() else None,
                            "pred_mean": float(p[m].mean()) if m.any() else None,
                            "recovered_frac": float(p[m].mean() / np.abs(a[m]).mean()) if m.any() and np.abs(a[m]).mean() > 1e-6 else None})
    return out


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cmrxrecon"
    methods = sys.argv[2:] or ["vggt_20260713_gather05"]
    allout = {}
    for meth in methods:
        recs = collect(dataset, meth)
        if not recs:
            print(f"{meth}: no data"); continue
        s = summarize(recs)
        s_no0 = summarize(recs, drop_slot0=True)
        clean = np.concatenate([r["clean_pred"] for r in recs])
        s["clean_control"] = {"mean_signed_mm": float(clean.mean()), "mean_abs_mm": float(np.abs(clean).mean()),
                              "max_abs_mm": float(np.abs(clean).max()), "n_slots": int(clean.size)}
        # GUARD: predicted Δz = breathing-response + the model's constant relocation offset (the clean-arm
        # signal). recovered_frac / epe_mm / bias_mm are NOT baseline-subtracted, so they are only valid
        # when the clean offset is small. On OOD (MIITT) the offset is ~8 mm and poisons these (e.g.
        # "1941% recovered"). Only `slope` is offset-robust (cov/var is shift-invariant). Warn loudly.
        if abs(clean.mean()) > 1.0:
            print(f"  ** WARNING: clean-arm offset {clean.mean():+.2f} mm is large — recovered_frac/EPE/bias "
                  f"are relocation-contaminated and NOT reliable here; trust only `slope`. **")
        s["slope_no_slot0"] = s_no0["slope"]; s["epe_no_slot0_mm"] = s_no0["epe_mm"]
        wss = [r["within_slot_std_mm"] for r in recs if "within_slot_std_mm" in r]
        if wss:
            s["within_slot_dz_std_mm"] = float(np.mean(wss))
        if all("cost_db" in r for r in recs):
            e = np.array([r["breath_epe"] for r in recs]); c = np.array([r["cost_db"] for r in recs])
            _, rr = fit(e, c)
            s["H1_epe_vs_cost_r"] = rr
            s["H1_epe_vs_cost_R2"] = rr * rr if rr is not None else None
            s["cost_db_mean"] = float(c.mean())
        allout[meth] = s
        print(f"\n=== {dataset} / {meth}  (n={s['n_subj']} subj, {s['n_slots']} slots)")
        print(f"  slope {s['slope']:.3f} (no slot0 {s['slope_no_slot0']:.3f})  corr {s['corr']:.3f}  "
              f"EPE {s['epe_mm']:.2f} mm  bias {s['bias_mm']:+.2f} mm")
        cc = s["clean_control"]
        print(f"  CLEAN control (applied=0): mean|pred| {cc['mean_abs_mm']:.2f} mm  max {cc['max_abs_mm']:.2f} mm  "
              f"signed bias {cc['mean_signed_mm']:+.2f} mm")
        if "within_slot_dz_std_mm" in s:
            print(f"  within-slot Dz std {s['within_slot_dz_std_mm']:.2f} mm (low=rigid shift, high=smear)")
        if "H1_epe_vs_cost_r" in s and s["H1_epe_vs_cost_r"] is not None:
            print(f"  H1: per-subject EPE vs breathing cost  r={s['H1_epe_vs_cost_r']:+.3f}  R2={s['H1_epe_vs_cost_R2']:.3f}  (cost mean {s['cost_db_mean']:.2f} dB)")
        print("  amplitude response (applied |SI| bin -> mean predicted Dz):")
        for b in s["bins"]:
            if b["n"]:
                print(f"    {b['bin']:>8s} mm  n={b['n']:4d}  applied {b['applied_mean']:5.1f} -> pred {b['pred_mean']:5.1f}  "
                      f"({100*b['recovered_frac']:.0f}% recovered)")

    od = "/home/minsukc/vggt/result/1frame_series"
    os.makedirs(od, exist_ok=True)
    p = os.path.join(od, f"breathing_{dataset}.json")
    json.dump(allout, open(p, "w"), indent=1)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
