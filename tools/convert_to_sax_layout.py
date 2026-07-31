"""Convert ACDC and M&Ms-1 4D cines into the CMRxRecon on-disk layout.

Output per subject (mirrors `CMRxRecon2024/Cine_combined/<ID>/sax/`):

    <out_root>/<PREFIX>_<id>/sax/3d_recon/sax_frame_{00..11}.nii.gz
    <out_root>/<PREFIX>_<id>/sax/convert_meta.json

so `preprocess.build_data_dicts`, `MRIDataset._find_subjects` and
`tools/nnunet_mnms_eval/prep_one.py`'s **cmrx** branch all work unchanged — every
source then flows through one identical code path. See docs/58 §8.2.

Three jobs:

1. **Pick 12 frames.** `native_idx(j) = (ed + round(j*T/12)) % T`, nearest native frame,
   NO temporal interpolation (`V_gt` is the supervision target; blended targets teach
   blur). ED lands exactly on frame 0. ED comes from metadata (ACDC `Info.cfg`, M&Ms
   CSV) — never assumed 0: only 138/345 M&Ms subjects have ED=0, and 146 have ED=T-1.
   The `% T` wrap handles both. ES generally does NOT land on the grid; measured cost on
   40 ACDC subjects with whole-cine segs: median EF error 0.24 pts, mean 0.65, worst 4.1
   (19/40 hit ES exactly). The bias is one-directional and applies to GT and prediction
   identically, so slope/Spearman are unaffected.

2. **Re-frame the affine (lossless).** Array axis 2 is the acquired slice axis and is
   PINNED as through-plane. Only the two in-plane axes may be swapped, and any axis may
   be flipped, to bring the volume as close to LPS as a permute+flip allows. Then a clean
   axis-aligned LPS affine is stamped with the true voxel sizes. No rotation, no resample,
   no interpolation — the acquired LV donuts are preserved bit-for-bit.

   This is deliberately NOT `Orientationd(axcodes="LPS")`: that assigns array axes by
   dominant patient axis, and M&Ms' slice normal is dominated by R/P/L/A rather than S,
   so it would move the slice axis into an in-plane slot and smear the heart across
   planes (`scratch/data/MNMs/README.md`, proof fig `result/mnms_scan/`).

   Discarding the true rotation is consistent, not lossy in any way that matters: CMRx
   and ACDC ship no real orientation at all, so all three sources end up in the same
   acquisition frame.

3. **Stamp the true pitch.** `zooms[2]` is the true centre-to-centre pitch for both
   sources (ACDC: documented protocol, `scratch/data/ACDC/README.md`; M&Ms: empirically
   indicated by the non-round values 8.05/9.52/9.96, `scratch/data/MNMs/README.md`).
   This matters because under the native-z design (docs/58) the on-disk pitch BECOMES
   the reconstruction grid.

Usage:
    python tools/convert_to_sax_layout.py                      # dry run, both sources
    python tools/convert_to_sax_layout.py --source acdc --limit 5
    python tools/convert_to_sax_layout.py --apply
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import nibabel as nib
import numpy as np

ROOT = "/home/minsukc/vggt"
DATA = os.path.join(ROOT, "scratch/data")
NUM_PHASES = 12

# Target patient-space directions, expressed in nibabel's RAS convention.
# LPS: L = -x_RAS, P = -y_RAS, S = +z_RAS.
T_L = np.array([-1.0, 0.0, 0.0])
T_P = np.array([0.0, -1.0, 0.0])
T_S = np.array([0.0, 0.0, 1.0])


# ──────────────────────────────────────────────────────────────────────────────
# Subject discovery + ED/ES metadata
# ──────────────────────────────────────────────────────────────────────────────
def _acdc_cfg(d):
    c = {}
    for line in open(os.path.join(d, "Info.cfg")):
        if ":" in line:
            k, v = line.split(":", 1)
            c[k.strip()] = v.strip()
    return c


def find_acdc():
    """[(out_id, src_4d_path, ed, es, extra_meta), ...]  ed/es 0-based."""
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, "ACDC", "*", "patient*", "patient*_4d.nii.gz"))):
        d = os.path.dirname(p)
        sid = os.path.basename(d)
        c = _acdc_cfg(d)
        # Info.cfg is 1-based; compute_cardiac_phase.py makes the same conversion.
        ed, es = int(c["ED"]) - 1, int(c["ES"]) - 1
        out.append((f"ACDC_{sid}", p, ed, es,
                    {"official_split": os.path.basename(os.path.dirname(d)),
                     "group": c.get("Group", ""), "nbframe": int(c.get("NbFrame", 0))}))
    return out


def find_mnms():
    csv_path = glob.glob(os.path.join(DATA, "MNMs", "MNMs1", "*_opendataset.csv"))
    if not csv_path:
        raise FileNotFoundError("M&Ms diagnosis CSV not found under MNMs/MNMs1/")
    meta = {}
    for r in csv.DictReader(open(csv_path[0])):
        meta[r["External code"]] = r
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, "MNMs", "MNMs1", "**", "*_sa.nii.gz"),
                              recursive=True)):
        code = os.path.basename(p).replace("_sa.nii.gz", "")
        if code not in meta:
            raise KeyError(f"{code} present on disk but missing from the M&Ms CSV")
        r = meta[code]
        # CSV ED/ES are already 0-based indices into the 4D cine (README, verified: all < T).
        out.append((f"MNMs_{code}", p, int(r["ED"]), int(r["ES"]),
                    {"official_split": os.path.basename(os.path.dirname(os.path.dirname(p))),
                     "vendor": r["VendorName"], "centre": r["Centre"],
                     "pathology": r["Pathology"]}))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────────
def plan_reframe(affine):
    """Lossless permute+flip bringing the volume closest to LPS, with array axis 2 PINNED
    as the through-plane axis.

    Returns (perm, flips, diag) where `perm` maps output spatial axis -> source spatial
    axis (perm[2] is always 2) and `flips` marks which OUTPUT axes to reverse.
    """
    cols = affine[:3, :3].astype(np.float64)
    norms = np.linalg.norm(cols, axis=0)
    if np.any(norms < 1e-8):
        raise ValueError(f"degenerate affine (zero-length column): {norms}")
    d = cols / norms                       # d[:, k] = RAS direction of source axis k

    # In-plane assignment: try both, keep the one with the better total alignment.
    # Only axes 0 and 1 are candidates — axis 2 is the acquired slice axis.
    a = abs(d[:, 0] @ T_L) + abs(d[:, 1] @ T_P)   # axis0 -> L, axis1 -> P
    b = abs(d[:, 1] @ T_L) + abs(d[:, 0] @ T_P)   # swapped
    perm = [0, 1, 2] if a >= b else [1, 0, 2]

    flips = [bool(d[:, perm[o]] @ t < 0) for o, t in enumerate((T_L, T_P, T_S))]
    diag = {
        "src_axcodes": "".join(nib.aff2axcodes(affine)),
        "slice_dir_ras": [round(float(v), 4) for v in d[:, 2]],
        "abs_S_component": round(float(abs(d[:, 2] @ T_S)), 4),
        "inplane_align": round(float(max(a, b)) / 2.0, 4),
        "swapped_inplane": perm[0] != 0,
        "flips": flips,
    }
    return perm, flips, diag


def apply_reframe(vol_xyz, perm, flips):
    """Permute + flip a 3D array. Pure view/copy — no interpolation."""
    out = np.transpose(vol_xyz, perm)
    for ax, f in enumerate(flips):
        if f:
            out = np.flip(out, axis=ax)
    return np.ascontiguousarray(out)


def lps_affine(shape_xyz, spacing_xyz):
    """Axis-aligned affine mapping voxel -> RAS with axis0=+L, axis1=+P, axis2=+S,
    origin at the volume centre (CMRx/ACDC carry no meaningful absolute position)."""
    sx, sy, sz = spacing_xyz
    A = np.diag([-sx, -sy, sz, 1.0]).astype(np.float64)   # L,P,S expressed in RAS
    A[:3, 3] = -A[:3, :3] @ ((np.asarray(shape_xyz, float) - 1) / 2.0)
    return A


def pick_frames(T, ed):
    """The 12 native frame indices, ED-anchored, nearest native, no interpolation."""
    return [(ed + int(round(j * T / float(NUM_PHASES)))) % T for j in range(NUM_PHASES)]


# ──────────────────────────────────────────────────────────────────────────────
# Conversion
# ──────────────────────────────────────────────────────────────────────────────
def convert_one(out_id, src, ed, es, extra, out_root, apply):
    im = nib.load(src)
    if im.ndim != 4:
        raise ValueError(f"{out_id}: expected 4D cine, got {im.shape}")
    zooms = [float(z) for z in im.header.get_zooms()[:3]]
    # SAX through-plane spacing must exceed both in-plane spacings; if it does not, the
    # slice axis is not array axis 2 and every geometric assumption below is wrong.
    if not zooms[2] > max(zooms[0], zooms[1]):
        raise ValueError(f"{out_id}: zooms={zooms} — axis 2 is not the slice axis")

    T = im.shape[3]
    if not (0 <= ed < T and 0 <= es < T):
        raise ValueError(f"{out_id}: ED={ed}/ES={es} out of range for T={T}")
    idx = pick_frames(T, ed)

    perm, flips, geo = plan_reframe(im.affine)
    sp_out = [zooms[perm[0]], zooms[perm[1]], zooms[2]]
    shp_out = [im.shape[perm[0]], im.shape[perm[1]], im.shape[2]]
    aff_out = lps_affine(shp_out, sp_out)
    # Post-condition: the stamped affine is axis-aligned, so the slice axis is exactly S.
    sd = aff_out[:3, 2] / np.linalg.norm(aff_out[:3, 2])
    assert abs(sd @ T_S) > 0.9, f"{out_id}: re-framed slice axis is not S ({sd})"

    # ES on the 12-frame grid: nearest sampled j to ES's fractional position. Advisory
    # only — the authoritative ED/ES are re-derived from our own seg on these 12 frames.
    es_j = int(round(((es - ed) % T) / float(T) * NUM_PHASES)) % NUM_PHASES

    meta = {
        "id": out_id, "source_file": os.path.relpath(src, DATA),
        "native_T": T, "ed_native": ed, "es_native": es,
        "frame_indices_native": idx, "es_frame_on_12grid_advisory": es_j,
        "native_zooms_xyz": zooms, "out_spacing_xyz": sp_out, "out_shape_xyz": shp_out,
        "pitch_mm": zooms[2], "reframe": geo,
        "converter": os.path.basename(__file__), "num_phases": NUM_PHASES, **extra,
    }
    if not apply:
        return meta

    sax = os.path.join(out_root, out_id, "sax")
    rec = os.path.join(sax, "3d_recon")
    os.makedirs(rec, exist_ok=True)
    arr = np.asarray(im.dataobj)                       # (X, Y, Z, T), source dtype
    for j, t in enumerate(idx):
        v = apply_reframe(arr[..., t], perm, flips)
        # pid in the tmp name so two concurrent converters can never collide on it.
        tmp = os.path.join(rec, f".sax_frame_{j:02d}.{os.getpid()}.tmp.nii.gz")
        nib.save(nib.Nifti1Image(v, aff_out), tmp)     # atomic: write then rename
        os.replace(tmp, os.path.join(rec, f"sax_frame_{j:02d}.nii.gz"))
    with open(os.path.join(sax, "convert_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _job(a):
    """Pool worker. Returns (meta, None) or (None, (id, err)) — never raises, so one bad
    subject cannot kill the pool."""
    out_id, src, ed, es, extra, out_root, apply = a
    try:
        return convert_one(out_id, src, ed, es, extra, out_root, apply), None
    except Exception as e:
        return None, (out_id, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["acdc", "mnms", "both"], default="both")
    ap.add_argument("--apply", action="store_true", help="write files (default: dry run)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N subjects per source")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes; gz encode is CPU-bound, so >1 helps a lot on --apply")
    args = ap.parse_args()

    jobs = []
    if args.source in ("acdc", "both"):
        jobs.append(("acdc", find_acdc(), os.path.join(DATA, "ACDC_sax")))
    if args.source in ("mnms", "both"):
        jobs.append(("mnms", find_mnms(), os.path.join(DATA, "MNMs_sax")))

    for name, subs, out_root in jobs:
        if args.limit:
            subs = subs[: args.limit]
        print(f"\n=== {name}: {len(subs)} subjects -> {os.path.relpath(out_root, ROOT)} "
              f"({'APPLY' if args.apply else 'dry run'})")
        work = [(o, s, ed, es, x, out_root, args.apply) for o, s, ed, es, x in subs]
        metas, failed = [], []
        if args.workers > 1:
            import multiprocessing as mp
            with mp.Pool(args.workers) as pool:
                results = pool.imap_unordered(_job, work, chunksize=1)
                for i, (m, err) in enumerate(results, 1):
                    (metas if m else failed).append(m or err)
                    if i % 25 == 0:
                        print(f"  ... {i}/{len(work)}", flush=True)
        else:
            for i, a in enumerate(work, 1):
                m, err = _job(a)
                (metas if m else failed).append(m or err)
                if args.apply and i % 25 == 0:
                    print(f"  ... {i}/{len(work)}", flush=True)
        metas.sort(key=lambda m: m["id"])
        if metas:
            sc = np.array([m["reframe"]["abs_S_component"] for m in metas])
            ip = np.array([m["reframe"]["inplane_align"] for m in metas])
            print(f"  ok {len(metas)}   pitch {min(m['pitch_mm'] for m in metas):.2f}"
                  f"-{max(m['pitch_mm'] for m in metas):.2f} mm"
                  f"   D {min(m['out_shape_xyz'][2] for m in metas)}"
                  f"-{max(m['out_shape_xyz'][2] for m in metas)}")
            print(f"  src |slice_dir . S|  min {sc.min():.3f}  median {np.median(sc):.3f}"
                  f"  max {sc.max():.3f}      (1.0 = slice axis already S)")
            print(f"  in-plane alignment   min {ip.min():.3f}  median {np.median(ip):.3f}"
                  f"      (1.0 = already axis-aligned; 0.71 = 45 deg off)")
            print(f"  in-plane swapped     {sum(m['reframe']['swapped_inplane'] for m in metas)}"
                  f"/{len(metas)}")
            print(f"  ES on the 12-grid    {sorted(set(m['es_frame_on_12grid_advisory'] for m in metas))}")
            for m in metas[:3]:
                print(f"    {m['id']:<18} T={m['native_T']:<3} ED={m['ed_native']:<3}"
                      f" frames={m['frame_indices_native']}")
        for out_id, e in failed:
            print(f"  FAILED {out_id}: {e}")
        print(f"  {len(failed)} failed")


if __name__ == "__main__":
    main()
