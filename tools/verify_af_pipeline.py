#!/usr/bin/env python
"""Standalone correctness checks for the docs/110 rhythm-cohort pipeline. Read-only.

Four checks, each paired with fault injections that MUST fire — a checker nobody has broken on
purpose is not evidence (it can pass by being vacuous).

  A. ORACLE GATES carry the truth. cardphase.txt must equal 2*pi*(pos_per_plane mod T)/T shifted
     by ONE global constant, with theta[0][0] at the 710 count token's residue. The per-element
     constancy is what catches a self-gated theta substituted into the oracle arm.

  B. ref_phase READOUT selects the right volume. The heredoc is extracted from the SHIPPED
     run_fetal4d.sh (never a copy) and driven on a synthetic cine whose phase k holds the constant
     value k, so the returned value names the bin it picked. Expectation comes from the manifest
     truth, not from the cardphase.txt the readout reads.

  C. regular_frozen REPLICATES the live bundle. Its input stacks must be bit-identical to the base
     cohort's rolled/ stacks and its GT a byte-identical cyclic permutation of the base GT --
     the precondition for that arm being a replication anchor.

  D. ANALYSIS MATH. tools/af_pilot_report.gate_stats decomposition on synthetic theta with known
     answers, including the property the regular* rows depend on: two uniform ramps always give a
     within-slice residual of exactly zero.

Usage:
    PYTHONPATH=training:. python tools/verify_af_pipeline.py
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import paths  # noqa: E402

T = 12
PY = sys.executable
COHORTS = ["cmrx2024_regular_frozen", "cmrx2024_regular", "cmrx2024_hrv", "cmrx2024_af"]
SUBJECTS = ["CMRx24_Test_P016", "CMRx24_Test_P004", "CMRx24_Test_P017", "CMRx24_Train_P046"]
COUNT_TOKEN = 710.0


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def extract_readout(dst):
    """Pull the ref_phase heredoc out of the SHIPPED run_fetal4d.sh."""
    src = os.path.join(ROOT, "evaluation/src/engine/run_fetal4d.sh")
    lines = open(src).read().splitlines()
    s = next(i for i, l in enumerate(lines) if l.startswith("import sys, nibabel"))
    e = next(i for i in range(s, len(lines)) if lines[i] == "EOF")
    open(dst, "w").write("\n".join(lines[s:e]) + "\n")
    return e - s


# ------------------------------------------------------------------ A
def check_oracle_gates():
    print("\n--- A. oracle gates carry the truth ---")
    bad = 0
    for c in COHORTS:
        for s in SUBJECTS:
            p = paths.subject_dir(c, s) / "fetal_cmr_4d_oracle" / "gate" / "cardphase.txt"
            if not p.is_file():
                continue
            th = np.array(open(p).read().split(), float)
            man = json.load(open(paths.manifest(c, s)))
            pos = np.asarray(man["rhythm"]["pos_per_plane"], float)
            th = th.reshape(pos.shape)
            off = (th - 2 * np.pi * (pos % T) / T + np.pi) % (2 * np.pi) - np.pi
            const = np.allclose(off, off.flat[0], atol=1e-5)
            tok = abs((COUNT_TOKEN - th[0, 0] + np.pi) % (2 * np.pi) - np.pi) < 1e-3
            if not (const and tok):
                bad += 1
                print(f"   FAIL {c}/{s} const={const} token={tok}")
    print(f"   {'ok' if not bad else 'FAILED'}: {bad} bad gate(s)")

    # fault injection
    p = paths.subject_dir("cmrx2024_af", "CMRx24_Test_P016") / "fetal_cmr_4d_oracle" / "gate" / "cardphase.txt"
    th0 = np.array(open(p).read().split(), float)
    man = json.load(open(paths.manifest("cmrx2024_af", "CMRx24_Test_P016")))
    pos = np.asarray(man["rhythm"]["pos_per_plane"], float)

    def verdict(th):
        th = th.reshape(pos.shape)
        off = (th - 2 * np.pi * (pos % T) / T + np.pi) % (2 * np.pi) - np.pi
        return (bool(np.allclose(off, off.flat[0], atol=1e-5)),
                bool(abs((COUNT_TOKEN - th[0, 0] + np.pi) % (2 * np.pi) - np.pi) < 1e-3))

    faults = {
        "perturb one theta": lambda a: (a.__setitem__(37, a[37] + 0.02), a)[1],
        "global shift +0.3": lambda a: a + 0.3,
        "replace with uniform ramp (self-gated)": lambda a: np.tile(
            2 * np.pi * np.arange(T) / T, pos.shape[0]),
        "shuffle": lambda a: np.random.default_rng(0).permutation(a),
    }
    for name, fn in faults.items():
        v = verdict(fn(th0.copy()))
        print(f"   {'FIRES ' if not all(v) else 'MISSED'} fault: {name}")
    return bad == 0


# ------------------------------------------------------------------ B
def check_readout():
    print("\n--- B. ref_phase readout selects the right volume ---")
    tmp = tempfile.mkdtemp()
    rp = os.path.join(tmp, "readout.py")
    n = extract_readout(rp)
    print(f"   extracted {n} lines from the shipped run_fetal4d.sh")
    out = os.path.join(tmp, "out")
    os.makedirs(out, exist_ok=True)
    cine = os.path.join(tmp, "cine.nii.gz")

    def mk():
        a = np.zeros((4, 4, 3, T), np.float32)
        for k in range(T):
            a[..., k] = k
        nib.save(nib.Nifti1Image(a, np.eye(4)), cine)

    def blend(i):
        i = i % T
        r = int(round(i))
        if abs(i - r) < 1e-4:
            return float(r % T)
        lo = int(np.floor(i))
        w = np.float32(i - lo)
        return float((lo % T) * (np.float32(1.0) - w) + ((lo + 1) % T) * w)

    def run(cp, ref):
        mk()
        r = subprocess.run([PY, rp, cine, out, str(T), "/nonexistent", "ref_phase", cp, str(ref)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None
        return [float(np.asarray(nib.load(f"{out}/vol_t{f:02d}.nii.gz").dataobj).flat[0])
                for f in range(T)]

    bad = n_ok = 0
    for c in COHORTS:
        for s in SUBJECTS:
            for arm in ["fetal_cmr_4d", "fetal_cmr_4d_oracle"]:
                cp = str(paths.subject_dir(c, s) / arm / "gate" / "cardphase.txt")
                if not os.path.isfile(cp):
                    continue
                man = json.load(open(paths.manifest(c, s)))
                ref = int(man["rhythm"]["ref_plane"])
                pos = np.asarray(man["rhythm"]["pos_per_plane"], float)
                if arm.endswith("_oracle"):
                    const = (COUNT_TOKEN % (2 * np.pi)) * T / (2 * np.pi) - (pos[0, 0] % T)
                    idx = (pos[ref] + const) % T
                else:
                    g = json.load(open(paths.subject_dir(c, s) / arm / "gate" / "gate.json"))
                    ed = {q["z"]: q for q in g["per_slice"]}[ref].get("ed_frame")
                    idx = (np.arange(T) - (0 if ed is None else int(ed))) % T
                exp = np.array([blend(float(v)) for v in idx])
                got = run(cp, ref)
                if got is None or not np.allclose(got, exp, atol=2e-3):
                    bad += 1
                    print(f"   FAIL {c}/{s}/{arm}\n     exp {np.round(exp,3)}\n     got {got}")
                else:
                    n_ok += 1
    print(f"   {'ok' if not bad else 'FAILED'}: {n_ok} correct, {bad} wrong")

    # fault injection on one real case
    c, s, arm = "cmrx2024_af", "CMRx24_Test_P016", "fetal_cmr_4d_oracle"
    cp = str(paths.subject_dir(c, s) / arm / "gate" / "cardphase.txt")
    ref = int(json.load(open(paths.manifest(c, s)))["rhythm"]["ref_plane"])
    base = run(cp, ref)
    th = np.array(open(cp).read().split(), float)
    badf = os.path.join(tmp, "bad.txt")
    for name, mut, r2 in [
            ("wrong ref_plane", None, (ref + 1) % 9),
            ("perturb one ref theta", lambda a: (a.__setitem__(ref * T + 3, a[ref * T + 3] + .25), a)[1], ref),
            ("uniform ramp on ref slice", lambda a: (a.__setitem__(
                slice(ref * T, (ref + 1) * T), 2 * np.pi * np.arange(T) / T), a)[1], ref)]:
        if mut is None:
            got = run(cp, r2)
        else:
            open(badf, "w").write(" ".join(f"{v:.6f}" for v in mut(th.copy())))
            got = run(badf, r2)
        print(f"   {'FIRES ' if got != base else 'MISSED'} fault: {name}")
    shutil.rmtree(tmp, ignore_errors=True)
    return bad == 0


# ------------------------------------------------------------------ C
def check_replication():
    print("\n--- C. regular_frozen replicates the live rolled/ bundle ---")
    bad = 0
    for s in SUBJECTS:
        base = paths.subject_dir("cmrx2024", s)
        rf = paths.subject_dir("cmrx2024_regular_frozen", s)
        if not (base / "rolled").is_dir() or not (rf / "breath").is_dir():
            continue
        md = max(float(np.abs(
            np.asarray(nib.load(str(rf / "breath" / f"stack_t{f:02d}.nii.gz")).dataobj, np.float64)
            - np.asarray(nib.load(str(base / "rolled" / f"stack_t{f:02d}.nii.gz")).dataobj, np.float64)
        ).max()) for f in range(T))
        gm = json.load(open(paths.manifest("cmrx2024_regular_frozen", s)))["rhythm"]["gt_map"]
        gtok = all(gm[f] is not None
                   and sha(str(rf / "gt" / f"gt_t{f:02d}.nii.gz")) == sha(str(base / "gt" / f"gt_t{gm[f]:02d}.nii.gz"))
                   for f in range(T))
        good = md == 0.0 and gtok
        bad += not good
        print(f"   {'ok  ' if good else 'FAIL'} {s:20s} input max|diff|={md:.3e}  GT byte-perm={gtok}")
    print(f"   {'ok' if not bad else 'FAILED'}: {bad} mismatch(es)")
    return bad == 0


# ------------------------------------------------------------------ D
def check_analysis_math():
    print("\n--- D. analysis decomposition math ---")
    import af_pilot_report as R
    tmp = tempfile.mkdtemp()
    orig = paths.VOLUMES
    paths.VOLUMES = __import__("pathlib").Path(tmp)
    D, k = 5, 2 * np.pi / T

    def write(coh, arm, th):
        d = os.path.join(tmp, coh, "out", "S1", arm, "gate")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "cardphase.txt"), "w").write(
            " ".join(f"{v:.6f}" for v in np.asarray(th).ravel()))

    true = np.random.default_rng(0).random((D, T)) * 2 * np.pi
    off = np.array([1., 1., 1., 0., 0.]) * k
    exp_off = off - np.angle(np.mean(np.exp(1j * np.repeat(off, T))))
    w = np.zeros((D, T)); w[:, ::2] = .5 * k; w[:, 1::2] = -.5 * k
    cases = [("identical", true.copy(), dict(rms=0., const=0., resid=0.)),
             ("global +2.0 rad", (true + 2.) % (2 * np.pi), dict(rms=0., const=0., resid=0.)),
             ("per-slice const", (true + off[:, None]) % (2 * np.pi),
              dict(const=float(np.sqrt(((exp_off * T / (2 * np.pi)) ** 2).mean())), resid=0.)),
             ("within-slice +-0.5fr", (true + w) % (2 * np.pi), dict(const=0., resid=.5))]
    ok = True
    for name, th, exp in cases:
        shutil.rmtree(os.path.join(tmp, "cmrx2024_x"), ignore_errors=True)
        write("cmrx2024_x", "fetal_cmr_4d", th)
        write("cmrx2024_x", "fetal_cmr_4d_oracle", true)
        g = R.gate_stats("cmrx2024_x", "S1")
        bad = [f"{f}={g[f]:.4f}!={exp[kk]:.4f}" for kk, f in
               [("rms", "rms_frames"), ("const", "rms_slice_const"), ("resid", "rms_within_slice")]
               if kk in exp and abs(g[f] - exp[kk]) > 2e-3]
        ok &= not bad
        print(f"   {'ok  ' if not bad else 'FAIL'} {name:22s} rms={g['rms_frames']:.3f} "
              f"const={g['rms_slice_const']:.3f} resid={g['rms_within_slice']:.3f} {' '.join(bad)}")
    worst = 0.
    for t in range(200):
        r = np.random.default_rng(t)
        a = 2 * np.pi * ((np.arange(T)[None] - r.integers(0, T, D)[:, None]) % T) / T
        b = 2 * np.pi * ((np.arange(T)[None] - r.integers(0, T, D)[:, None]) % T) / T
        shutil.rmtree(os.path.join(tmp, "cmrx2024_y"), ignore_errors=True)
        write("cmrx2024_y", "fetal_cmr_4d", a)
        write("cmrx2024_y", "fetal_cmr_4d_oracle", b)
        worst = max(worst, R.gate_stats("cmrx2024_y", "S1")["rms_within_slice"])
    ok &= worst < 1e-6
    print(f"   {'ok  ' if worst < 1e-6 else 'FAIL'} uniform-ramp pair => residual 0 "
          f"(200 trials, worst={worst:.2e})")
    paths.VOLUMES = orig
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


if __name__ == "__main__":
    res = [check_oracle_gates(), check_readout(), check_replication(), check_analysis_math()]
    print("\n" + ("ALL CHECKS PASSED" if all(res) else "SOME CHECKS FAILED"))
    sys.exit(0 if all(res) else 1)
