"""Self-gating step 3/3 — read nnU-Net Task114 segs of every (slice, frame), build the per-slice
LV blood-pool AREA curve, detect ED/ES (Åkesson 2025 rule), and RE-ANCHOR the authors' x-f
cardiac phases so theta=0 sits at each slice's own detected ED.

This replaces fetal_cmr_4d's inter-slice-sync step (impossible single-orientation, doc 34) with a
per-slice absolute ED anchor (doc 35). It keeps the AUTHORS' x-f R-R intervals + phases verbatim
(cardsync/rrintervals.txt, cardphases_intraslice_cardsync.txt) — the ONLY change is the per-slice
phase OFFSET, previously identity (-> 106deg ED scatter, doc 34), now the LV-area-ED offset.

Outputs (in <recon>/Volunteer1/cardsync/):
  cardphases_lvanchor_cardsync.txt   -- re-anchored phases, SAME format, feeds reconstructCardiac
  selfgate_lvarea.json               -- per-slice ED/ES frames, scatter metrics
  selfgate_lvarea.png                -- LV-area curves + ED/ES + phase-scatter diagnostic

Validation metrics printed:
  ED scatter BEFORE (frame-0 anchored)  ~ should reproduce doc-34's 106deg (desync proof)
  ED scatter AFTER  (LV-anchored)       ~ 0 by construction (sanity)
  ES scatter AFTER  (LV-anchored)       = the non-circular test + doc-35 linear-vs-two-anchor gate

Run: micromamba run -n svr python baselines/fetal_cmr_4d/selfgate_lvarea_assemble.py Volunteer1
"""
import os, sys, json, glob, re
import numpy as np
import nibabel as nib
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import circmean, circstd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTROOT = os.path.join(_ROOT, "scratch/fetal_cmr_4d/recon")
FRAME_DT = 0.025          # s/frame (MIITT RT), matches miitt_preproc.m
LV_LABEL = 1              # Task114: 1=LV blood pool
ED_ORDER = 3             # Åkesson: larger than its 3 nearest frames each direction


def per_slice_area(segdir, vol, Z, T):
    """(Z, T) LV blood-pool area (voxel count) from the Task114 segs."""
    area = np.full((Z, T), np.nan)
    for s in glob.glob(os.path.join(segdir, f"{vol}__z*__f*.nii.gz")):
        m = re.search(r"__z(\d+)__f(\d+)", os.path.basename(s))
        z, f = int(m.group(1)), int(m.group(2))
        area[z, f] = int((np.asarray(nib.load(s).dataobj) == LV_LABEL).sum())
    return area


def detect_ed_es(a, rr_frames):
    """Robust per-slice ED/ES from the LV-area curve, using the known R-R to enforce ~one beat/peak.
    (Naive local-maxima over-fire on the noisy curve — doc 34/35: needs smoothing + a min-distance
    prior, not naive peak-picking.) ED = smoothed-area peaks >= 0.6*RR apart; ES = area min between
    consecutive EDs on the smoothed curve. Returns (ed, es) frame lists on the ORIGINAL index."""
    a = np.nan_to_num(a)
    if a.max() <= 0:
        return [], []
    w = int(max(5, round(rr_frames * 0.25)))     # savgol window ~1/4 cycle, odd
    w = min(w | 1, (len(a) // 2) * 2 - 1)
    sm = savgol_filter(a, w, 2) if len(a) > w else a
    dist = max(5, int(rr_frames * 0.6))          # >= 0.6 R-R between EDs -> one peak/beat
    prom = 0.25 * (sm.max() - sm.min())
    ed = find_peaks(sm, distance=dist, prominence=prom)[0].tolist()
    es = []
    for f0, f1 in zip(ed[:-1], ed[1:]):
        es.append(f0 + int(np.argmin(sm[f0:f1 + 1])))
    return ed, es


def main():
    vol = sys.argv[1] if len(sys.argv) > 1 else "Volunteer1"
    rd = os.path.join(OUTROOT, vol)
    meta = json.load(open(os.path.join(rd, "selfgate", "extract_meta.json")))
    Z, T = meta["Z"], meta["T"]
    segdir = os.path.join(rd, "selfgate", "lvseg_segs")

    area = per_slice_area(segdir, vol, Z, T)
    n_seg = int(np.isfinite(area).sum())
    if n_seg < Z * T:
        print(f"WARNING: only {n_seg}/{Z*T} segs found — nnU-Net may still be running", flush=True)

    # authors' x-f gating outputs (verbatim)
    rr = np.loadtxt(os.path.join(rd, "cardsync", "rrintervals.txt"))                       # (Z,)
    intra = np.loadtxt(os.path.join(rd, "cardsync", "cardphases_intraslice_cardsync.txt")).reshape(Z, T)

    # high-confidence = LV segmented on >=90% of frames (else nnU-Net dropout -> unreliable ES)
    COVERAGE_MIN = 0.90
    ed_ph_before, es_ph_after, sysfrac, per_slice = [], [], [], []
    new = intra.copy()
    for z in range(Z):
        a = np.nan_to_num(area[z])
        cover = float((a > 0).mean())
        ed, es = detect_ed_es(a, rr[z] / FRAME_DT)
        if not ed:
            # no LV / no ED (base/apex) -> keep identity offset, exclude from metrics
            per_slice.append({"z": z, "ed_frames": [], "es_frames": [], "reliable": False,
                              "high_conf": False, "coverage": cover, "n_lv_max": float(a.max())})
            continue
        phi = circmean(intra[z, ed], high=2 * np.pi, low=0.0)      # ED anchor offset
        new[z] = np.mod(intra[z] - phi, 2 * np.pi)                 # anchor ALL detected slices in the file
        es_p = circmean(new[z, es], high=2 * np.pi, low=0.0) if es else np.nan
        high_conf = cover >= COVERAGE_MIN
        if high_conf:
            ed_ph_before.append(phi)
            if es:
                es_ph_after.append(es_p)
                sf = np.mean([((e - d) * FRAME_DT / rr[z]) for d, e in zip(ed, es) if e > d])
                sysfrac.append(sf)
        per_slice.append({"z": z, "ed_frames": ed, "es_frames": es, "reliable": True,
                          "high_conf": high_conf, "coverage": cover,
                          "ed_anchor_rad": float(phi), "es_phase_after_rad": float(es_p) if es else None,
                          "n_lv_max": float(a.max())})

    ed_before = np.array(ed_ph_before)
    es_after = np.array([e for e in es_ph_after if np.isfinite(e)])
    m = {
        "vol": vol, "Z": Z, "T": T,
        "n_reliable_slices": int(sum(p["reliable"] for p in per_slice)),
        "n_high_conf_slices": len(ed_before),
        "ed_scatter_before_deg": float(np.degrees(circstd(ed_before, high=2*np.pi, low=0))) if len(ed_before) > 1 else None,
        "es_scatter_after_deg": float(np.degrees(circstd(es_after, high=2*np.pi, low=0))) if len(es_after) > 1 else None,
        "systolic_fraction_mean": float(np.mean(sysfrac)) if sysfrac else None,
        "systolic_fraction_std": float(np.std(sysfrac)) if sysfrac else None,
        "per_slice": per_slice,
    }
    json.dump(m, open(os.path.join(rd, "cardsync", "selfgate_lvarea.json"), "w"), indent=2)

    # write re-anchored cardphase file (same slice-major/frame-minor format, wrapped [0,2pi])
    with open(os.path.join(rd, "cardsync", "cardphases_lvanchor_cardsync.txt"), "w") as fh:
        fh.write(" ".join(f"{v:.6f}" for v in new.reshape(-1)) + " ")

    print(f"\n=== self-gate LV-area re-anchoring: {vol} ===")
    print(f"anchored slices: {m['n_reliable_slices']}/{Z}   high-conf (metric): {m['n_high_conf_slices']}/{Z}")
    print(f"ED scatter BEFORE (frame-0 anchored): {m['ed_scatter_before_deg']:.1f} deg   [doc-34 baseline ~106]")
    print(f"ED scatter AFTER  (LV-anchored):      ~0 by construction")
    if m["es_scatter_after_deg"] is not None:
        print(f"ES scatter AFTER  (LV-anchored):      {m['es_scatter_after_deg']:.1f} deg   [doc-35 gate: <15 -> linear theta OK]")
        print(f"systolic fraction: {m['systolic_fraction_mean']:.3f} +/- {m['systolic_fraction_std']:.3f}")

    _plot(rd, vol, area, per_slice, ed_before, es_after)
    print(f"-> cardsync/cardphases_lvanchor_cardsync.txt + selfgate_lvarea.{{json,png}}")


def _plot(rd, vol, area, per_slice, ed_before, es_after):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z = area.shape[0]
    fig = plt.figure(figsize=(16, 9))
    for z in range(Z):
        ax = fig.add_subplot(4, 4, z + 1)
        ax.plot(area[z], lw=0.8, color="k")
        ps = per_slice[z]
        for f in ps.get("ed_frames", []):
            ax.axvline(f, color="tab:red", lw=0.8, alpha=0.7)
        for f in ps.get("es_frames", []):
            ax.axvline(f, color="tab:blue", lw=0.8, alpha=0.7, ls="--")
        ax.set_title(f"z{z} {'' if ps['reliable'] else '(no ED)'}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(4, 4, 16, projection="polar")
    ax.plot(ed_before, np.ones_like(ed_before), "o", color="tab:red", label="ED before", ms=5)
    ax.plot(np.zeros_like(ed_before), np.ones_like(ed_before) * 0.6, "o", color="tab:green", label="ED after", ms=5)
    if len(es_after):
        ax.plot(es_after, np.ones_like(es_after) * 0.8, "s", color="tab:blue", label="ES after", ms=4)
    ax.set_yticks([]); ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("phase scatter", fontsize=8)
    fig.suptitle(f"{vol}: LV-area self-gating (red=ED, blue=ES)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(rd, "selfgate_lvarea.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
