"""Step 2/3 driver — run a classical SVR baseline (SVRTK / NeSVoR) over an eval cohort.

The recon shells (`run_svrtk3d.sh`, `run_nesvor.sh`) are per-(subject, variant) and take their
geometry via env vars; this is their only caller. Per subject it reads the frozen bundle's
`manifest.json` (split, T, dz) and resolves the PSF slice THICKNESS — which is NOT the pitch
`dz_mm` for most sources (docs/27, docs/83):

    cmrx2023/24 : 8.0            (documented: 8 mm thickness + 4 mm gap -> 12 mm pitch)
    cmrx2025    : dz - 4         (docs/54 par.10c: the +4 mm gap rule holds; thickness varies by
                                  centre; the per-subject csv SliceThickness is pitch-valued and
                                  wrong — e.g. Center006 says 12.0 where the measured pitch is 10.0)
    acdc        : 10 -> 5.0      (Bernard 2018: 5 mm thickness + 5 mm gap); else contiguous (= dz;
                                  5/8 documented contiguous, the one 7.0 subject is an assumption)
    mnms        : dz             (contiguous; Campello 2021 Table III reports per-centre thickness
                                  9.2-10 mm ~= the header spacing, no gap documented anywhere)
    miitt       : 8.0            (data author: 8 mm thickness + 2 mm gap)
    ocmr        : per-subject    (ISMRMRD acquisition headers -> source meta.json slice_thickness_mm,
                                  resolved via the rel_path -> convert_meta.json chain)

A subject whose `recon_<variant>/stamp.json` already exists is complete and is SKIPPED without
invoking the shell — the shells are phase-level idempotent, but a no-op re-invocation still
rewrites provenance.txt/total_wall.sec and would clobber the fair timing record.

Run (one method per invocation — SVRTK is CPU, NeSVoR needs a GPU):
    PYTHONPATH=training:. python evaluation/src/engine/run_baselines.py --method svrtk3d
    PYTHONPATH=training:. python evaluation/src/engine/run_baselines.py --method nesvor \
        --sources cmrx2024 --subjects CMRx24_Test_P012
    ... --shard 0 3        # this process handles every 3rd subject (SLURM array sharding)

Extra shell knobs (J, OMP, DEBUG, METHOD) pass through the environment; METHOD names the arm
(e.g. METHOD=svrtk3d_debug DEBUG=1 --input scatter -> arm svrtk3d_debug_scatter, its own dir,
never colliding with / skipped-as-done under the campaign's <method>_scatter arm).
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import paths  # noqa: E402

DATA_ROOT = os.path.join(ROOT, "scratch/data")
SHELLS = {"svrtk3d": "run_svrtk3d.sh", "nesvor": "run_nesvor.sh", "niftymic": "run_niftymic_v2.sh",
          "fetal_cmr_4d": "run_fetal4d.sh"}   # fetal_cmr_4d: rolled/ input only, self-gated (docs/105)


def ocmr_thickness(rel_path):
    """OCMR ships real per-subject thickness: sax/convert_meta.json names the source cine dir,
    whose meta.json carries slice_thickness_mm verbatim from the ISMRMRD acquisition headers."""
    cm = json.load(open(os.path.join(DATA_ROOT, rel_path, "sax", "convert_meta.json")))
    src_meta = os.path.join(DATA_ROOT, os.path.dirname(cm["source_file"]), "meta.json")
    return float(json.load(open(src_meta))["slice_thickness_mm"])


def thickness_mm(source, rel_path, dz):
    """PSF slice thickness (mm) for the recon shells' -thickness/--thicknesses. See module docstring."""
    if source in ("cmrx2023", "cmrx2024"):
        return 8.0
    if source == "cmrx2025":
        return dz - 4.0
    if source == "acdc":
        return 5.0 if abs(dz - 10.0) < 0.01 else dz
    if source == "mnms":
        return dz
    if source == "miitt":
        return 8.0
    if source == "ocmr":
        return ocmr_thickness(rel_path)
    raise ValueError(f"no thickness rule for source '{source}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", required=True, choices=sorted(SHELLS))
    ap.add_argument("--sources", nargs="+", default=list(paths.DATASETS))
    ap.add_argument("--subjects", nargs="+", default=None, help="restrict to these subject names")
    ap.add_argument("--split", default="val")
    ap.add_argument("--variant", default="breath", choices=paths.VARIANTS)
    ap.add_argument("--input", default="gated", choices=["gated", "scatter"],
                    help="which bundle stack feeds the recon: 'gated' = the variant's own phase-"
                         "consistent stack (baseline upper bound); 'scatter' = the SAME-INPUT stack "
                         "VGGT sees (scatter/, manifest['scatter']) -> arm <method>_scatter")
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"),
                    help="process only subjects where index %% N == I (SLURM array sharding)")
    ap.add_argument("--dry-run", action="store_true", help="print the work list, run nothing")
    args = ap.parse_args()

    shell = os.path.join(os.path.dirname(os.path.abspath(__file__)), SHELLS[args.method])
    if args.input == "scatter" and args.variant != "breath":
        sys.exit("--input scatter is built from the breath bundle only; use --variant breath")
    # Arm name: a user METHOD (e.g. svrtk3d_debug) is honored for scatter too, always with the
    # _scatter suffix appended, so a debug rerun lands in its own arm dir instead of silently
    # being skipped-as-already-stamped under the campaign's <method>_scatter arm.
    base = os.environ.get("METHOD") or args.method
    # GATE_ARM and METHOD are independent env vars, and mis-pairing them is silent: set one and not
    # the other and you get an arm named _oracle holding self-gated timing, or an oracle recon
    # written into (or skipped as already-stamped under) the plain arm dir. Require them to agree.
    gate_arm_env = os.environ.get("GATE_ARM")
    if gate_arm_env and gate_arm_env != base:
        sys.exit(f"GATE_ARM={gate_arm_env!r} but METHOD={base!r}: an arm must be named after the "
                 f"gate it consumes, or its recon is mis-attributed. Set METHOD={gate_arm_env}.")
    arm = base + "_scatter" if args.input == "scatter" else base
    work = []                                     # (source, subject, T, thick)
    for ds in args.sources:
        keep, dropped = paths.filter_by_split(ds, paths.subjects(ds), args.split)
        for s, why in dropped:
            if args.subjects and s in args.subjects:
                print(f"!! {ds}/{s} requested but excluded: {why}")
        for s in keep:
            if args.subjects and s not in args.subjects:
                continue
            m = json.load(open(paths.manifest(ds, s)))
            # Guard on the DIRECTORY, not the manifest key. A rhythm cohort (docs/110) carries the
            # scatter draw in its manifest (run_vggt.pin_scatter needs it) but writes no scatter/
            # stacks, so a key-only check passes here and then fails inside the container.
            if args.input == "scatter" and not (paths.subject_dir(ds, s) / "scatter").is_dir():
                raise FileNotFoundError(f"{ds}/{s}: no scatter/ stacks — run pooled.py --add-scatter "
                                        f"(a rhythm cohort does not build them)")
            if args.method == "fetal_cmr_4d":
                if args.input == "scatter" or args.variant != "breath":
                    sys.exit("fetal_cmr_4d reads the rolled/ stack only: use --variant breath --input gated")
                if "rolled" not in m:
                    raise KeyError(f"{ds}/{s}: bundle has no rolled/ stack — run pooled.py --add-rolled")
                gate_arm = os.environ.get("GATE_ARM", "fetal_cmr_4d")
                if not (paths.subject_dir(ds, s) / gate_arm / "gate" / "cardphase.txt").is_file():
                    raise FileNotFoundError(f"{ds}/{s}: no {gate_arm}/gate — run fetal4d_gate.py dump/seg/assemble"
                                            + (" --oracle" if gate_arm.endswith("_oracle") else ""))
            # m["source"], not the cohort dir name: a rhythm-arm cohort (docs/110) is named
            # `<source>_<arm>` but carries the BASE source in its manifest, and thickness is a
            # property of the scanner protocol, not of the simulated rhythm. thickness_mm() raises
            # on an unknown name, so passing the dir name would hard-fail those cohorts.
            thick = thickness_mm(m.get("source", ds), m["rel_path"], float(m["dz_mm"]))
            if thick <= 0 or thick > float(m["dz_mm"]) + 1e-6:
                raise ValueError(f"{ds}/{s}: implausible thickness {thick} for dz {m['dz_mm']}")
            work.append((ds, s, int(m["T"]), thick))
    if args.shard:
        i, n = args.shard
        work = [w for k, w in enumerate(work) if k % n == i]

    todo, done = [], 0
    for ds, s, T, thick in work:
        if paths.recon_stamp(ds, s, arm, args.variant).is_file():
            done += 1
        else:
            todo.append((ds, s, T, thick))
    print(f"[{arm}/{args.variant}] {len(work)} subjects in shard: "
          f"{done} already stamped, {len(todo)} to run")

    if args.dry_run:
        for ds, s, T, thick in todo:
            print(f"  {ds:9s} {s:50s} T={T:2d} THICK={thick:g}")
        return

    ok, failed = 0, []
    for k, (ds, s, T, thick) in enumerate(todo):
        print(f"--- [{k + 1}/{len(todo)}] {ds}/{s} (T={T}, thick={thick:g}mm) ---", flush=True)
        env = {**os.environ, "EVAL_DATASET": ds, "T": str(T), "THICK": f"{thick:g}",
               "MASK_FILE": paths.HEART_MASK_PAD, "METHOD": arm,   # +10 mm padded ROI (docs/107)
               "INPUT": "scatter" if args.input == "scatter" else args.variant}
        r = subprocess.run(["bash", shell, s, args.variant], env=env)
        if r.returncode == 0 and paths.recon_stamp(ds, s, arm, args.variant).is_file():
            ok += 1
        else:
            failed.append(f"{ds}/{s} (rc={r.returncode})")
            print(f"!! FAILED {ds}/{s}", flush=True)
    print(f"DONE [{arm}/{args.variant}]: {ok} ok, {len(failed)} failed, {done} pre-existing")
    for f in failed:
        print(f"  FAILED: {f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
