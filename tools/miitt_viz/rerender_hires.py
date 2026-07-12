"""Re-render the 7-row gated-sweep GIFs at higher dpi using the ORIGINAL matplotlib render_7row
(white background, proper subplot alignment) — from the saved npz (CPU only, no model rerun).
Usage: python rerender_hires.py [npz ...]   (default: all sweep npz)
"""
import os, sys, glob, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "training")
from tools.miitt_viz.gated_gather05_7row import render_7row

DPI = 130   # native ~256px/panel; the original layout, just higher resolution


def title_for(f, rd):
    p = f.split("/"); model, ds, subj = p[2], p[3], p[4]
    cond = os.path.basename(f).replace("_7row.npz", "")
    si = np.abs(rd[1:, 0])
    alab = ("clean (no breathing)" if cond == "clean" or si.max() < 1e-6
            else f"{cond}: SI breathing mean {si.mean():.1f} max {si.max():.1f}mm, {int((si > 8).sum())} planes>8mm (ref clean)")
    return f"{model} 1frame | {ds}/{subj} | {alab}"


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob("result/gated_model_sweep/*/*/*/*.npz"))
    for f in files:
        try:
            d = np.load(f)
        except Exception:
            print("SKIP corrupt", f, flush=True); continue
        cap = dict(GT=d["gt"], RE=d["recon"], IN=d["inp"], CO=d["cov"], DV=d["dvf"],
                   rd=d["applied_disp"], sop=list(d["sop"]), has_slot=list(d["has_slot"]),
                   z_mid=int(d["ref_zmid"]))
        gif = f.replace(".npz", ".gif")
        render_7row(cap, title_for(f, d["applied_disp"]), gif, dpi=DPI)
        print(f"rendered {gif}  {os.path.getsize(gif)//1024} KB", flush=True)
    print("DONE", flush=True)
