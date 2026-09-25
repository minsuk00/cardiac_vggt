"""Screen af12 test subjects for a representative AF LV volume-time-curve (VTC) figure.

For every af12 subject: GT LV curve vs VGGT diff1000 curve (from the stored ef/ JSONs) plus the
reference plane's simulated cardiac position (manifest rhythm.ref_pos, T == ED/full). Flags the AF
signatures visible in a 12-frame (~1 s) window and renders the shortlisted candidates.

Run: micromamba run -n svr python tools/vtc_af12_candidates.py --out temp/vtc_candidates
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRCS = ["acdc", "cmrx2023", "cmrx2024", "cmrx2025", "mnms"]
METHOD = "vggt_final518_diff1000_ep300"
DT_MS = 1000.0 / 12  # nominal 1 s R-R, 12 frames


def features(pos, gt):
    pos = np.asarray(pos, float)
    gt = np.asarray(gt, float)
    held = pos >= 12.0 - 1e-6
    # longest run of consecutive frames sitting at full relaxation (diastasis hold)
    runs, r = [], 0
    for h in held:
        r = r + 1 if h else 0
        runs.append(r)
    hold = max(runs)
    # beat boundaries: position wraps/drops (new contraction starts)
    onsets = [f for f in range(1, 12) if pos[f] < pos[f - 1] - 1e-6]
    # a contraction starting from incomplete filling: the frame before the onset is not at full
    incomplete = [f for f in onsets if pos[f - 1] < 12.0 - 1e-6 and not held[f - 1]]
    edv, esv = gt.max(), gt.min()
    sv = edv - esv
    # one full cycle in the window: full -> empty -> full (or empty -> full -> empty), each
    # extreme within 40% of SV of the window's EDV/ESV -- a short-R-R beat refills only partway
    full, empty = gt >= edv - 0.4 * sv, gt <= esv + 0.4 * sv
    fi, ei = np.flatnonzero(full), np.flatnonzero(empty)
    cycle = any(ei[(ei > i)].size and fi[fi > ei[ei > i][0]].size for i in fi) or \
        any(fi[(fi > i)].size and ei[ei > fi[fi > i][0]].size for i in ei)
    # spikiness: at each interior local max, the smaller of the one-frame rise and fall, / SV
    spike = max([min(gt[f] - gt[f - 1], gt[f] - gt[f + 1]) / sv for f in range(1, 11)
                 if gt[f] >= gt[f - 1] and gt[f] >= gt[f + 1]] + [0.0])
    return dict(hold=hold, onsets=onsets, incomplete=incomplete,
                swing=sv / edv, edv=edv, esv=esv, cycle=bool(cycle), spike=spike)


def load():
    rows = []
    for s in SRCS:
        coh = f"{s}_af12"
        ef = json.load(open(f"evaluation/metric_results/test/{coh}/ef/{METHOD}.json"))["per_subject"]
        for e in ef:
            man = json.load(open(f"scratch/eval/{coh}/out/{e['subject']}/manifest.json"))["rhythm"]
            f = features(man["ref_pos"], e["lv_curve_gt"])
            rows.append(dict(cohort=coh, subject=e["subject"], pos=man["ref_pos"],
                             gt=e["lv_curve_gt"], pred=e["lv_curve_breath"],
                             nmae=e["lv_curve_nmae_breath"], ef_gt=e["ef_gt"],
                             rr=man["rr_per_plane"][man["ref_plane"]], **f))
    return rows


def plot(r, ax):
    t = np.arange(12) * DT_MS
    pos = np.asarray(r["pos"])
    for f in range(12):
        if pos[f] >= 12.0 - 1e-6:
            ax.axvspan(t[f] - DT_MS / 2, t[f] + DT_MS / 2, color="0.9", lw=0)
    for f in r["onsets"]:
        ax.axvline(t[f] - DT_MS / 2, color="0.5", ls=":", lw=1)
    ax.plot(t, r["gt"], "-o", color="k", ms=3, label="GT")
    ax.plot(t, r["pred"], "-s", color="#1f6fb4", ms=3, label="Ours")
    ax.set_title(f"{r['subject']} ({r['cohort'].split('_')[0]})\n"
                 f"VTC err {r['nmae']:.1f}% | GT EF {r['ef_gt']:.0f}% | hold {r['hold']}f | "
                 f"onsets {len(r['onsets'])} (incompl {len(r['incomplete'])})", fontsize=8)
    ax.set_xlabel("time (ms)", fontsize=8)
    ax.set_ylabel("LV volume (mL)", fontsize=8)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="temp/vtc_candidates")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load()
    nm = np.array([r["nmae"] for r in rows])
    lo, med, hi = np.percentile(nm, [30, 50, 70])
    print(f"n={len(rows)}  VTC err p30/p50/p70 = {lo:.2f}/{med:.2f}/{hi:.2f}")

    # AF-visible but not extreme, physiological GT EF, median-ish VGGT error (representative).
    # A: flat diastasis plateau of 2-3 frames (167-250 ms, no atrial-kick rise), one beat onset.
    # B: a short-R-R beat -- contraction starts before full filling (smaller EDV/SV), hold <= 3.
    base = [r for r in rows if r["hold"] <= 3 and len(r["onsets"]) <= 2
            and 40 <= r["ef_gt"] <= 70 and lo <= r["nmae"] <= hi]
    groups = {
        "A_plateau": [r for r in base if 2 <= r["hold"] and not r["incomplete"]],
        "B_short_rr_beat": [r for r in base if r["incomplete"]],
        # C: B + at least one full full->empty->full cycle + no sharp one-frame peak
        "C_short_rr_full_cycle": [r for r in base if r["incomplete"] and r["cycle"]
                                  and r["spike"] <= 0.2],
    }
    print(f"incomplete-beat subjects overall: {sum(bool(r['incomplete']) for r in rows)}/180")
    out = []
    for name, keep in groups.items():
        keep.sort(key=lambda r: abs(r["nmae"] - med))
        print(f"\n{name}: {len(keep)}")
        for r in keep:
            print(f"{r['cohort']:15s} {r['subject']:22s} nmae {r['nmae']:5.2f} EF {r['ef_gt']:4.1f} "
                  f"hold {r['hold']} onsets {r['onsets']} incompl {r['incomplete']} "
                  f"cycle {r['cycle']} spike {r['spike']:.2f} "
                  f"pos {np.round(r['pos'], 1).tolist()}")
        n = min(len(keep), 12)
        if not n:
            continue
        fig, axs = plt.subplots(3, 4, figsize=(16, 10))
        for ax in axs.flat[n:]:
            ax.axis("off")
        for r, ax in zip(keep[:n], axs.flat):
            plot(r, ax)
        axs.flat[0].legend(fontsize=7)
        fig.suptitle(f"af12 VTC candidates {name} (grey band = diastasis hold at full relaxation; "
                     "dotted = new contraction onset)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"candidates_{name}.png"), dpi=130)
        plt.close(fig)
        out += [dict(group=name, **{k: r[k] for k in ("cohort", "subject", "nmae", "ef_gt", "hold",
                                                      "onsets", "incomplete", "pos", "gt", "pred")})
                for r in keep]
    json.dump(out, open(os.path.join(a.out, "candidates.json"), "w"), indent=1)

    # every short-R-R subject, no other filter, sorted by VTC err, 20 per page
    allb = sorted([r for r in rows if r["incomplete"]], key=lambda r: r["nmae"])
    for p in range(0, len(allb), 20):
        fig, axs = plt.subplots(4, 5, figsize=(20, 14))
        page = allb[p:p + 20]
        for ax in axs.flat[len(page):]:
            ax.axis("off")
        for r, ax in zip(page, axs.flat):
            plot(r, ax)
        axs.flat[0].legend(fontsize=7)
        fig.suptitle(f"af12, all short-R-R subjects ({len(allb)}), sorted by VTC err "
                     f"(median over all 180 = {med:.2f}%) -- page {p // 20 + 1}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"all_short_rr_p{p // 20 + 1}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
