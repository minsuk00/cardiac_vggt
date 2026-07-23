"""Phase-0 read-only smoke: paths.py must resolve every existing subject/arm/recon/
summary that a raw glob finds. No writes. Exits non-zero on any mismatch.

Facts this test accounts for (discovered on first run, not paths.py bugs):
  - T is per-dataset (from manifest), not always 12.
  - Only vggt_* arms write metadata.json; baseline arms (svrtk3d/nesvor) do not.
  - paths.* returns paths through the evaluation/volumes symlink; compare via samefile.
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/home/minsukc/vggt/evaluation")
import paths  # noqa: E402

VGGT = "/home/minsukc/vggt"
ok = True


def check(label, cond, detail=""):
    global ok
    if not cond:
        ok = False
    print(f"  [{'OK ' if cond else 'FAIL'}] {label} {detail}")


def samefiles(a_paths, b_paths):
    """Set-equality of two path lists by inode identity (crosses the symlink)."""
    def keys(ps):
        return {os.path.realpath(str(p)) for p in ps}
    return keys(a_paths) == keys(b_paths)


for ds in paths.DATASETS:
    root = f"{VGGT}/scratch/eval/{ds}/out"
    print(f"\n=== {ds} ===  ({root})")
    if not os.path.isdir(root):
        print("  (no out/ — skipping)")
        continue

    raw_subj = sorted(os.path.basename(os.path.dirname(m))
                      for m in glob.glob(f"{root}/*/manifest.json"))
    check("subjects match raw glob", raw_subj == paths.subjects(ds),
          f"(paths={len(paths.subjects(ds))} raw={len(raw_subj)})")

    raw_arms = set()
    for s in raw_subj:
        for d in glob.glob(f"{root}/{s}/*"):
            if os.path.isdir(d) and os.path.basename(d) not in paths.BUNDLE_DIRS:
                raw_arms.add(os.path.basename(d))
    check("arms match raw union", raw_arms == set(paths.arms(ds)),
          f"(paths={len(paths.arms(ds))} raw={len(raw_arms)})")

    if not raw_subj or not raw_arms:
        continue

    # probe a (subject, arm) that has recons; verify resolvers against the real files.
    probe = next(((s, a) for s in paths.subjects(ds) for a in paths.arms(ds, s)
                  if paths.recon_dir(ds, s, a, "clean").is_dir()), None)
    check("found a probe (subject,arm) with recons", probe is not None)
    if probe:
        s, a = probe
        T = json.load(open(paths.manifest(ds, s)))["T"]
        check("manifest resolves", paths.manifest(ds, s).is_file())
        for kind in paths.BUNDLE_DIRS:
            check(f"bundle_stack[{kind}] t0 resolves", paths.bundle_stack(ds, s, kind, 0).is_file())
        check("fov_mask resolves", paths.fov_mask(ds, s).is_file())
        for var in paths.VARIANTS:
            rd = paths.recon_dir(ds, s, a, var)
            raw_vols = glob.glob(f"{rd}/vol_t*.nii.gz")
            p_vols = [paths.recon(ds, s, a, var, t) for t in range(T)]
            check(f"recon[{var}] resolves all T={T} & matches glob ({a})",
                  all(p.is_file() for p in p_vols) and samefiles(p_vols, raw_vols),
                  f"(paths={len(p_vols)} raw={len(raw_vols)})")
        check("metrics.json resolves", paths.metrics(ds, s, a).is_file())
        # metadata.json only for vggt_* arms
        vggt_arm = next((x for x in paths.arms(ds, s) if x.startswith("vggt_")), None)
        if vggt_arm:
            check("metadata.json resolves (vggt arm)", paths.metadata(ds, s, vggt_arm).is_file(),
                  vggt_arm)

    # legacy_summary must point at each existing *_summary.json (by inode identity).
    raw_sum = glob.glob(f"{root}/*_summary.json")
    all_match = all(
        paths.legacy_summary(ds, os.path.basename(f)[:-len("_summary.json")]).is_file()
        and os.path.samefile(paths.legacy_summary(ds, os.path.basename(f)[:-len("_summary.json")]), f)
        for f in raw_sum)
    check("legacy_summary points at existing summaries", all_match, f"(n={len(raw_sum)})")

print("\n=== canonical_arm doubling guard ===")
check("plain contz not doubled",
      paths.canonical_arm("1f_contz_ep99", date="20260719", continuous_z=True)
      == "vggt_20260719_1f_contz_ep99")
check("non-contz gets marker",
      paths.canonical_arm("1f_gather05_ep99", date="20260719", continuous_z=True)
      == "vggt_20260719_1f_gather05_ep99_contz")
check("slug form (no date)", paths.canonical_arm("gather05") == "vggt_gather05")
check("already-prefixed + date not re-prefixed",
      paths.canonical_arm("vggt_gather05", date="20260719") == "vggt_20260719_gather05")
check("already-prefixed, no date, no double",
      paths.canonical_arm("vggt_gather05") == "vggt_gather05")

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
