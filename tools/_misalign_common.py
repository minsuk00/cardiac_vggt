"""Shared loader for the misalignment-figure scripts (render_misalignment_figure.py,
render_misalignment_cube.py). Not a standing tool on its own — no CLI.
"""
import json, os
import numpy as np, nibabel as nib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dhw(p):                       # nifti (X,Y,Z) -> (D,H,W) = (Z,Y,X)
    return np.asarray(nib.load(p).dataobj, dtype=np.float32).transpose(2, 1, 0)


def load_case(ds, subject, arm, t=0):
    """Load the 5 corruption-panel volumes + geometry for one (dataset, subject, arm, target
    phase). Returns a dict: vols (list of (name, (D,H,W) vol)), D, dz_mm, inplane_mm,
    zc/yc/xc (heart-mask centroid, voxel indices), y0/y1/x0/x1 (padded heart-mask XY bbox),
    z0/z1 (padded heart-mask Z bbox), t_of_z (the recorded slot draw), manifest.
    """
    sd = os.path.join(ROOT, "evaluation/volumes", ds, "out", subject)
    m = json.load(open(os.path.join(sd, "manifest.json")))
    D, dz = m["D"], m["dz_mm"]; inplane = m["spacing_xyz_mm"][0]
    draw = np.load(os.path.join(sd, arm, "ed_dvf.npz"))
    t_of_z = {int(round(z)): int(tt) for z, tt in zip(draw["slot_z"], draw["slot_t"])}
    assert sorted(t_of_z) == list(range(D)), t_of_z

    gt = load_dhw(f"{sd}/gt/gt_t{t:02d}.nii.gz")
    fb = load_dhw(f"{sd}/breath/stack_t{t:02d}.nii.gz")
    clean = {tt: load_dhw(f"{sd}/clean/stack_t{tt:02d}.nii.gz") for tt in set(t_of_z.values())}
    breath = {tt: load_dhw(f"{sd}/breath/stack_t{tt:02d}.nii.gz") for tt in set(t_of_z.values())}
    ung = np.stack([clean[t_of_z[z]][z] for z in range(D)])
    fbung = np.stack([breath[t_of_z[z]][z] for z in range(D)])
    ours = load_dhw(f"{sd}/{arm}/recon_breath/vol_t{t:02d}.nii.gz")
    vols = [("(a) Target", gt), ("(b) Respiratory misalignment", fb), ("(c) Cardiac mismatch", ung),
            ("(d) Combined corruption", fbung), ("(e) Motion-corrected (Ours)", ours)]

    hm = load_dhw(f"{sd}/mask_heart.nii.gz") > 0
    zc, yc, xc = [int(round(v)) for v in np.argwhere(hm).mean(0)]
    zs, ys, xs = np.where(hm)
    pad_xy, pad_z = 24, 1
    y0, y1 = max(ys.min() - pad_xy, 0), min(ys.max() + pad_xy, gt.shape[1])
    x0, x1 = max(xs.min() - pad_xy, 0), min(xs.max() + pad_xy, gt.shape[2])
    z0, z1 = max(zs.min() - pad_z, 0), min(zs.max() + pad_z + 1, D)

    return dict(vols=vols, D=D, dz_mm=dz, inplane_mm=inplane, zc=zc, yc=yc, xc=xc,
                y0=y0, y1=y1, x0=x0, x1=x1, z0=z0, z1=z1, t_of_z=t_of_z, manifest=m)
