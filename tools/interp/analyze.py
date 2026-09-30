"""Aggregate statistics for the interpretability campaign (docs/126).

Reads temp/interp/{e1,e1b,e2,e5,e7} (+ scratch/interp/e4/e4_curves.json if present) and prints, per
condition, the subject mean with a 95% bootstrap CI, the paired difference vs the full model, a
Wilcoxon signed-rank p-value, and the fraction of subjects moving in the direction of the mean.

Primary metrics
  cardiac transfer  G = mean_t NCC(V_t, GT_t) - mean_{t!=t'} NCC(V_t, GT_t') on COMPANION planes
                    (reference plane excluded); reported raw and as % of the full model's G.
  breathing         through-plane EPE (mm, per-subject demeaned, companions) and slope of predicted
                    vs applied dz.
  patching          state index = (s(V_patch)-s(V_src)) / (s(V_donor)-s(V_src)), s = NCC(.,GT_don)-NCC(.,GT_src)
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "temp/interp")
RNG = np.random.default_rng(0)


def boot_ci(x, n=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    m = RNG.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired(a, b):
    """a = condition, b = reference (full). Returns mean diff, CI, wilcoxon p, frac same sign as mean."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    if len(d) < 3 or np.allclose(d, 0):
        return np.mean(d) if len(d) else np.nan, (np.nan, np.nan), np.nan, np.nan
    p = wilcoxon(d).pvalue
    return d.mean(), boot_ci(d), p, np.mean(np.sign(d) == np.sign(d.mean()))


def fmt_row(name, vals, ref=None):
    m = np.nanmean(vals); lo, hi = boot_ci(vals)
    s = f"  {name:<18} {m:8.4f} [{lo:7.4f},{hi:7.4f}]"
    if ref is not None:
        dm, (dlo, dhi), p, fr = paired(vals, ref)
        s += f"   Δ={dm:+.4f} [{dlo:+.4f},{dhi:+.4f}] p={p:.1e} same-sign={fr:.2f}"
    return s


# ─────────────────────────── E1 / E1b ───────────────────────────
def gap(M):
    T = M.shape[0]
    return np.mean(np.diag(M)) - (M.sum() - np.trace(M)) / (T * T - T)


def e1_metrics(rec, cond):
    rows = rec["cond"][cond]
    T = rec["T"]
    M = np.array([r["ncc_comp"] for r in rows])
    G = np.mean(np.diag(M)) - (M.sum() - np.trace(M)) / (T * T - T)
    # far planes (|z - z_ref| >= 2): free of the reference slice's own trilinear-splat leakage
    G_far = gap(np.array([r["ncc_far"] for r in rows])) if rows[0].get("ncc_far") is not None else np.nan
    G_common = gap(np.array([r["ncc_common"] for r in rows])) if rows[0].get("ncc_common") is not None else np.nan
    G_conly = gap(np.array([r["ncc_conly"] for r in rows])) if rows[0].get("ncc_conly") is not None else np.nan
    G_conly_common = (gap(np.array([r["ncc_conly_common"] for r in rows]))
                      if rows[0].get("ncc_conly_common") is not None else np.nan)
    Mr = np.array([r["ncc_ref"] for r in rows])
    Gr = np.mean(np.diag(Mr)) - (Mr.sum() - np.trace(Mr)) / (T * T - T)
    diag = np.mean(np.diag(M))
    # rows may carry their own slot-ordered displacement (permuted runs, e.g. E1b ref_offmid)
    appl = np.array(rows[0].get("disp_slot", rec["disp"]))[:, 0]
    dz = np.array([r["disp_fov"] for r in rows])[:, :, 0]
    err = dz[:, 1:] - appl[None, 1:]
    err = err - np.nanmean(err)
    epe = np.nanmean(np.abs(err))
    slope = nan_slope(appl[1:], np.nanmean(dz[:, 1:], 0))
    return {"G": G, "G_far": G_far, "G_common": G_common, "G_conly": G_conly, "G_conly_common": G_conly_common,
            "G_ref": Gr, "ncc_match": diag, "epe": epe, "slope": slope}


def nan_slope(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    ok = ~(np.isnan(a) | np.isnan(p))
    a, p = a[ok], p[ok]
    return np.polyfit(a, p, 1)[0] if len(a) > 2 and a.std() > 1e-3 else np.nan


def e1_table(dirname, label):
    fs = sorted(glob.glob(f"{D}/{dirname}/s*.json"))
    recs = [json.load(open(f)) for f in fs]
    recs = [r for r in recs if r.get("cond")]
    if not recs:
        return None
    conds = list(recs[0]["cond"])
    recs = [r for r in recs if all(c in r["cond"] for c in conds)]     # keep arrays paired
    vals = {c: {k: [] for k in ("G", "G_far", "G_common", "G_conly", "G_conly_common", "G_ref", "ncc_match",
                                "epe", "slope")} for c in conds}
    for r in recs:
        for c in conds:
            m = e1_metrics(r, c)
            for k in m:
                vals[c][k].append(m[k])
    print(f"\n=== {label}: n={len(recs)} subjects ===")
    for key, desc in (("G_conly", "cardiac transfer G — COMPANION-ONLY splat (primary; no reference leakage)"),
                      ("G_conly_common", "G companion-only splat, planes excl. mid AND off-mid (for ref_offmid)"),
                      ("G", "cardiac transfer G (companion planes)"), ("G_far", "cardiac transfer G (far planes, |z-zref|>=2)"),
                      ("G_common", "cardiac transfer G (planes excl. mid AND off-mid; same set for all conds)"),
                      ("epe", "breathing EPE mm (companions)"),
                      ("slope", "breathing slope"), ("G_ref", "reference-plane G")):
        if np.all(np.isnan(vals["full"][key])):
            continue
        print(f" -- {desc}")
        for c in conds:
            print(fmt_row(c, vals[c][key], None if c == "full" else vals["full"][key]))
    for key in ("G_conly", "G", "G_far"):
        Gf = np.nanmean(vals["full"][key])
        if np.isnan(Gf):
            continue
        print(f" -- cardiac transfer as % of full (mean {key} ratio)")
        print("  " + "  ".join(f"{c}:{100 * np.nanmean(vals[c][key]) / Gf:.0f}%" for c in conds))
    return vals


def e4_table():
    p = os.path.join(ROOT, "scratch/interp/e4/e4_curves.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    print(f"\n=== E4 LV volume curves (nnU-Net): n={len(d['per_subject'])} ===")
    for scope in ("all", "comp", "far"):
        if scope not in d["summary"]["full"]:
            continue
        print(f" -- {scope} planes: VTC% / EF err / swing ratio (NaN counts shown; NaN EF = empty LV seg)")
        ref = {k: np.array(d["summary"]["full"][scope])[:, i] for i, k in enumerate(("vtc", "ef", "swing", "r"))}
        for arm, v in d["summary"].items():
            v = np.array(v[scope], float)
            for i, k in enumerate(("vtc", "ef", "swing")):
                nn = int(np.isnan(v[:, i]).sum())
                print(fmt_row(f"{arm}.{k}" + (f" (nan={nn})" if nn else ""), v[:, i], None if arm == "full" else ref[k]))
    return d


# ─────────────────────────── E2 ───────────────────────────
def e2_table():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/e2/s*.json"))]
    if not recs:
        return None
    conds = list(recs[0]["A"][0]["pred"])
    out = {c: {"epe": [], "slope": [], "epe_y": [], "slope_y": []} for c in conds}
    for r in recs:
        appl = np.array([row["disp"] for row in r["A"]])
        for c in conds:
            pred = np.array([row["pred"][c] for row in r["A"]])
            for ax, sfx in ((0, ""), (1, "_y")):
                a, p = appl[..., ax][:, 1:].ravel(), pred[..., ax][:, 1:].ravel()
                ok = ~np.isnan(p); a, p = a[ok], p[ok]
                e = p - a; e = e - e.mean()
                out[c]["epe" + sfx].append(np.mean(np.abs(e)))
                out[c]["slope" + sfx].append(np.polyfit(a, p, 1)[0] if a.std() > 1e-3 else np.nan)
    print(f"\n=== E2 breathing pathway: n={len(recs)} subjects x {len(recs[0]['A'])} draws ===")
    for key in ("epe", "slope", "epe_y", "slope_y"):
        print(f" -- {key}")
        for c in conds:
            print(fmt_row(c, out[c][key], None if c == "full" else out["full"][key]))
    # dose-response / z-lie / common-mode
    DEL = np.array([-24, -18, -12, -6, -3, 0, 3, 6, 12, 18, 24])
    print(" -- B single-slot content shift: probe dz (mean over subjects) per δ")
    for c in recs[0]["B"]:
        ys = np.array([[row[r["probe_slot"]][0] for row in r["B"][c]] for r in recs])
        print(f"  {c:<10}" + " ".join(f"{v:6.1f}" for v in np.nanmean(ys, 0)))
    ys = np.array([[row[r["probe_slot"]][0] for row in r["C"]["full"]] for r in recs])
    print(f"  {'z-lie':<10}" + " ".join(f"{v:6.1f}" for v in np.nanmean(ys, 0)))
    for v in recs[0]["D_"]:
        ys = np.array([[np.nanmean(np.array(row)[1:, 0]) for row in r["D_"][v]] for r in recs])
        print(f"  {'cm-' + v:<10}" + " ".join(f"{x:6.1f}" for x in np.nanmean(ys, 0)))
    print(f"  {'δ':<10}" + " ".join(f"{d:6d}" for d in DEL))
    return out


# ─────────────────────────── E5 ───────────────────────────
def e5_table():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/e5/s*.json"))]
    recs = [r for r in recs if r.get("dirs")]
    if not recs:
        return None
    names = list(recs[0]["dirs"][0]["patch"])
    # primary: companion-only splat (element 2, no reference-slice leakage) when present
    j = 2 if len(recs[0]["dirs"][0]["src_ncc"]) > 2 else 0
    print(f"  [metric source: {'companion-only splat' if j == 2 else 'full splat, companion planes'}]")
    idx = {n: [] for n in names}
    selfdiff = []
    for r in recs:
        per = {n: [] for n in names}
        for d in r["dirs"]:
            def s(n):
                return n[j][1] - n[j][0] if d["src"] == r["a"] else n[j][0] - n[j][1]
            s0, s1 = s(d["src_ncc"]), s(d["don_ncc"])
            selfdiff.append(d["selfpatch_maxdiff"])
            for n in names:
                per[n].append((s(d["patch"][n]["ncc"]) - s0) / (s1 - s0) if abs(s1 - s0) > 1e-6 else np.nan)
        for n in names:
            idx[n].append(np.nanmean(per[n]))
    print(f"\n=== E5 activation patching (companions read donor reference K/V): n={len(recs)} ===")
    print(f"  self-patch max |Δwp| over all runs: {np.max(selfdiff):.2e}")
    held = np.array([r["idx"] not in SELECTION_IDX for r in recs])
    for n in names:
        v = np.array(idx[n])
        if n.startswith("L3_h") or n in ("L3_top4", "L3_other12"):
            v = v[held]                      # head subsets: held-out subjects only (TOP4 chosen on idx 0-2)
            print(fmt_row(n + " (held-out)", v))
        else:
            print(fmt_row(n, v))
    return idx


SELECTION_IDX = {0, 1, 2}   # subjects used to pick TOP4 heads (e5_patching.TOP4)


# ─────────────────────────── E7 ───────────────────────────
def e7_table():
    recs = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{D}/e7/s*.pkl"))]
    recs = [r for r in recs if r]
    if not recs:
        return None
    print(f"\n=== E7 locality (layer-3 partial K/V patch): n={len(recs)} ===")
    # one value per subject (mean of the two half-patches) — the unit of replication is the subject
    same = [0.5 * (r["sets"]["half_L"]["d_left"] + r["sets"]["half_R"]["d_right"]) for r in recs]
    opp = [0.5 * (r["sets"]["half_L"]["d_right"] + r["sets"]["half_R"]["d_left"]) for r in recs]
    print(fmt_row("patched side mm", same))
    print(fmt_row("opposite side mm", opp, None))
    dm, ci, p, fr = paired(same, opp)
    print(f"  same - opposite: {dm:+.3f} mm {ci} p={p:.1e} frac>0={fr:.2f}")
    prof = np.array([r["sets"][f"blk{j}"]["prof"] for r in recs for j in range(4)], float)
    print("  block-patch influence vs distance (mm): " + " ".join(f"{b}:{v:.2f}" for b, v in zip(recs[0]['sets']['blk0']['bins'], np.nanmean(prof, 0))))
    return same, opp


def main():
    out = {}
    out["e1"] = e1_table("e1", "E1 reference pathway")
    out["e1b"] = e1_table("e1b", "E1b dissociation")
    out["e2"] = e2_table()
    out["e5"] = e5_table()
    out["e7"] = e7_table()
    out["e4"] = e4_table()


if __name__ == "__main__":
    main()
