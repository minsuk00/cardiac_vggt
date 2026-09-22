"""EF / Dice for the frames-per-slice probes (tools/cinevol_nframes_probe.py,
tools/fetal4d_nframes_probe.py): the standing chain ef_dice dump -> run_seg.sh (nnU-Net 3d_fullres)
-> ef_dice score, run in-process against a probe root under temp/ with paths.VOLUMES redirected.

The shared GT seg cache (<subject>/seg_gt_3d_fullres) is symlinked in, so every subject is
`gt_cached` and nothing is written to scratch/eval -- the script aborts if any subject is not.
Per-frame segs are kept (project rule) under <probe>/ef/n<N>/seg/.

    PYTHONPATH=training:. python tools/nframes_probe_ef.py --probe cinevol --n 48   # needs a GPU
    PYTHONPATH=training:. python tools/nframes_probe_ef.py --probe fetal --n 12
    PYTHONPATH=training:. python tools/nframes_probe_ef.py --probe cinevol --report
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("tools", "evaluation", "evaluation/src/score", "evaluation/src/engine", "training"):
    sys.path.insert(0, str(ROOT / p))
os.environ.setdefault("SPLIT", "test")
os.environ.setdefault("SEG_CFG", "3d_fullres")
import paths                                   # noqa: E402
import ef_dice                                 # noqa: E402
from cinevol_nframes_probe import SUBJECTS     # noqa: E402

REAL = paths.VOLUMES.resolve()
PROBES = {"cinevol": (ROOT / "temp" / "cinevol_nframes_probe", "cinevol", (12, 24, 48)),
          "fetal": (ROOT / "temp" / "fetal4d_nframes_probe", "fetal_cmr_4d", (12, 24))}
COHORTS = [f"{src}_af24" for src in ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms")]
GT_CACHE = "seg_gt_3d_fullres"


def run(probe, n):
    root, method, _ = PROBES[probe]
    paths.VOLUMES = root / f"n{n}"
    for src, s in SUBJECTS:
        d = paths.subject_dir(f"{src}_af24", s)
        assert d.is_dir(), d
        if not (d / GT_CACHE).exists():
            os.symlink(REAL / f"{src}_af24" / "out" / s / GT_CACHE, d / GT_CACHE)
    ef_dir = root / "ef" / f"n{n}"
    out = ef_dir / "ef.json"
    if out.is_file():
        print(f"{probe} n={n}: {out} exists, skipping")
        return
    if ef_dir.exists():
        raise FileExistsError(f"{ef_dir} exists without ef.json -- inspect, then move it aside")
    inp, seg = ef_dir / "in", ef_dir / "seg"
    inp.mkdir(parents=True); seg.mkdir()
    ef_dice.dump(SimpleNamespace(input_dir=str(inp), method=method, cohorts=COHORTS, gt_only=False))
    man = json.load(open(inp / "ef_manifest.json"))["subjects"]
    bad = [r["subject"] for r in man if not r.get("gt_cached")]
    assert not bad, f"GT not cached for {bad} -- refusing (score would write the shared cache)"
    assert len(man) == len(SUBJECTS), (len(man), [r["subject"] for r in man])
    r = subprocess.run(["bash", str(ROOT / "evaluation/src/engine/run_seg.sh"), str(inp), str(seg)])
    if r.returncode:
        raise RuntimeError("run_seg.sh failed")
    ef_dice.score(SimpleNamespace(seg_dir=str(seg), input=str(inp), out=str(out)))
    shutil.rmtree(inp)          # the dumped crops are regenerable; the segs are kept


def report(probe):
    from scipy.stats import wilcoxon
    root, method, ns = PROBES[probe]
    rows = {}
    for n in ns:
        p = root / "ef" / f"n{n}" / "ef.json"
        if p.is_file():
            rows[n] = {r["subject"]: r for r in json.load(open(p))["per_subject"]}
    vg = {}
    for src in ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms"):
        p = ROOT / "temp/rhythm24_ef/af24" / f"vggt_final518_diff1000_ep300__{src}_af24.json"
        if p.is_file():
            vg.update({r["subject"]: r for r in json.load(open(p))["per_subject"]})
    ok = lambda r: r.get("ef_breath") is not None and r.get("ef_gt") is not None
    common = [s for _, s in SUBJECTS if all(s in rows[n] and ok(rows[n][s]) for n in rows)]
    print(f"{probe}: N={sorted(rows)}  subjects with EF at every N: {len(common)}/{len(SUBJECTS)}")
    err = {n: np.array([abs(rows[n][s]["ef_breath"] - rows[n][s]["ef_gt"]) for s in common]) for n in rows}
    dice = {n: np.array([[rows[n][s].get(f"dice_breath_LV_{ph}", np.nan) for ph in ("ED", "ES")] for s in common]) for n in rows}
    for n in sorted(rows):
        print(f"  N={n:2d}  EF MAE {err[n].mean():6.2f} pp   EF bias {np.mean([rows[n][s]['ef_breath'] - rows[n][s]['ef_gt'] for s in common]):+6.2f}"
              f"   Dice LV ED/ES {np.nanmean(dice[n][:, 0]):.3f}/{np.nanmean(dice[n][:, 1]):.3f}")
    base = 24
    for n in sorted(rows):
        if n == base:
            continue
        d = err[n] - err[base]
        p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d) else np.nan
        print(f"  N={n} - N={base}  EF MAE {d.mean():+.2f} pp  (worse in {int((d > 0).sum())}/{len(d)})  p={p:.3f}")
    vs = [s for s in common if s in vg and ok(vg[s])]
    if vs:
        ve = np.array([abs(vg[s]["ef_breath"] - vg[s]["ef_gt"]) for s in vs])
        print(f"  VGGT diff1000 on the same {len(vs)}: EF MAE {ve.mean():.2f} pp")
        for n in sorted(rows):
            e = np.array([abs(rows[n][s]["ef_breath"] - rows[n][s]["ef_gt"]) for s in vs])
            print(f"    {method} N={n:2d} - VGGT: {np.mean(e - ve):+.2f} pp  (VGGT better in {int((e > ve).sum())}/{len(vs)})")
    for s in common:
        print(f"    {s:48} GT {rows[sorted(rows)[0]][s]['ef_gt']:5.1f}  " + "  ".join(f"N{n}={rows[n][s]['ef_breath']:5.1f}" for n in sorted(rows))
              + (f"  VGGT={vg[s]['ef_breath']:5.1f}" if s in vg and ok(vg[s]) else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", required=True, choices=list(PROBES))
    ap.add_argument("--n", type=int)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(a.probe)
    else:
        run(a.probe, a.n)


if __name__ == "__main__":
    main()
