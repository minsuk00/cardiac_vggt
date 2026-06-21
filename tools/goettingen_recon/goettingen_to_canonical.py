#!/usr/bin/env python
"""Make a Göttingen RT free-breathing recon consumable by the trained VGGT model via MRIDataset.

The joint-L1 model is **z-only** for inputs (use_t_pose_embedding=false): it conditions input slices
on z alone and is BLIND to input cardiac phase (only the target_t query is phase-conditioned). So we
do NOT need to gate / estimate the cardiac phase of input frames at all — we just feed real scattered
slices and sweep target_t for the reconstruction.

We still write a 12-"frame" bundle in the CMRxRecon layout
(<out_root>/<subj>/sax/3d_recon/sax_frame_{00..11}.nii.gz) purely as a vehicle so the existing
MRIDataset + canonical transforms resample to the 256x256x12 cube and build the batch. Each "frame"
slot is just a real, evenly-spaced Göttingen frame per slice — the model ignores the slot's phase
label, and because each SAX slice was acquired independently, sampling across slots/z gives the
realistic *scattered* (uncorrelated cardiac+respiratory) input the model is designed to correct.

NOTHING here alters the source recon; it only writes a new derived subject dir under <out_root>.
"""
import argparse, os
import numpy as np
import nibabel as nib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('nii'); ap.add_argument('out_root')
    ap.add_argument('--phases', type=int, default=12)
    args = ap.parse_args()

    subj = os.path.basename(args.nii).replace('.nii.gz', '')
    a = nib.load(args.nii).get_fdata().astype(np.float32)        # [X,Y,Z,T]
    X, Y, Z, T = a.shape
    P = args.phases

    # 12 evenly-spaced real frames across each slice's cine (a sampling vehicle; phase label unused)
    frame_idx = np.linspace(0, T - 1, P).round().astype(int)

    out_dir = os.path.join(args.out_root, subj, 'sax', '3d_recon')
    os.makedirs(out_dir, exist_ok=True)
    aff = np.diag([1.6, 1.6, 6.0, 1.0])                          # native Göttingen geometry
    for p in range(P):
        vol = a[:, :, :, frame_idx[p]]                          # [X,Y,Z] real frame across all slices
        nib.save(nib.Nifti1Image(vol, aff), os.path.join(out_dir, f'sax_frame_{p:02d}.nii.gz'))
    print(f'{subj}: wrote {P} frame-NIfTIs (idx {frame_idx.tolist()}) -> {out_dir}  (X={X},Y={Y},Z={Z},T={T})')


if __name__ == '__main__':
    main()
