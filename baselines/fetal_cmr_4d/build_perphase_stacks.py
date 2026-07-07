"""Build per-phase multi-stacks for the 3D-per-phase SVR baseline (doc 35 roster: SVRTK 3D+t).

For each of P output phases, and each z-slice, take the K frames nearest that phase (K≈#beats, so
~one frame per beat) -> K stacks of (X,Y,Z), each a proper parallel-slice stack at the true z
positions. `mirtk reconstruct` then SVR-reconstructs each phase's 3D volume from its K stacks
INDEPENDENTLY (no temporal PSF, no cross-phase coupling) -> preserves contraction (doc 35 §12).

Run: micromamba run -n svr python baselines/fetal_cmr_4d/build_perphase_stacks.py Volunteer1 [K] [P]
"""
import os, sys
import numpy as np
import nibabel as nib

RD = "scratch/fetal_cmr_4d/recon"


def main():
    vol = sys.argv[1] if len(sys.argv) > 1 else "Volunteer1"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    P = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    rd = os.path.join(RD, vol)
    im = nib.load(os.path.join(rd, "data", "s01_rlt_ab.nii.gz"))
    rlt = im.get_fdata().astype(np.float32)                 # (X,Y,Z,F)
    ph = np.loadtxt(os.path.join(rd, "cardsync", "cardphases_lvanchor_cardsync.txt"))
    X, Y, Z, F = rlt.shape
    ph = ph.reshape(Z, F)
    od = os.path.join(rd, "perphase_stacks"); os.makedirs(od, exist_ok=True)
    binw = 2 * np.pi / P
    for p in range(P):
        center = p * binw
        # per z, the K frames nearest this phase (wrapped)
        picks = np.zeros((Z, K), int)
        for z in range(Z):
            d = np.abs(np.angle(np.exp(1j * (ph[z] - center))))
            picks[z] = np.argsort(d)[:K]
        for k in range(K):
            st = np.stack([rlt[:, :, z, picks[z, k]] for z in range(Z)], axis=2)  # (X,Y,Z)
            nib.save(nib.Nifti1Image(st, im.affine, im.header),
                     os.path.join(od, f"stack_p{p:02d}_k{k}.nii.gz"))
    print(f"built {P} phases x {K} stacks -> {od}")


if __name__ == "__main__":
    main()
