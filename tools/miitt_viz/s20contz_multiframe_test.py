"""Test s20contz in its NATIVE multiframe regime on MIITT — does the recon move non-reference
planes (vs the near-frozen 1-frame-contz result)? Measures per-plane aliveness (temporal std of
recon vs GT over the cardiac cycle) + renders 7-row GIFs. Inline GPU run.
Out: result/gated_model_sweep/s20contz_multiframe/miitt/<subj>/<cond>_7row.{gif,npz}
"""
import os, sys, glob, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "training")
from inference.adapters import MIITTGatedAdapter
from inference.inference import load_rtfb_model_reference
from inference.run_gated_ood import load_rcfg
from tools.miitt_viz.gated_gather05_7row import capture, render_7row

DEV="cuda"; OUT="result/gated_model_sweep/s20contz_multiframe"; SEED=72; MAXPH=18
CKPT=glob.glob("scratch/logs/216949414_*s20contz*/ckpts/checkpoint_last.pt")[0]
SUBJS=[f"Volunteer{i}" for i in (1,2,3)]      # start with 3

def alive(RE, GT):
    out=[]
    for p in range(GT.shape[1]):
        m=GT[:,p].max(0)>1e-4
        out.append(float(RE[:,p][:,m].std(0).mean()) if m.sum() else 0.0)
    return np.array(out)

def main():
    rn=load_rcfg()
    model=load_rtfb_model_reference(CKPT, refiner=False, device=DEV)
    print(f"loaded s20contz: {CKPT}", flush=True)
    for subj in SUBJS:
        p=f"scratch/data/MIITT/nifti/{subj}/gated/sax/4d_recon.nii.gz"
        if not os.path.exists(p): print("skip",subj); continue
        ad=MIITTGatedAdapter(p)
        bundle_np,bbox=ad.build_canonical_bundle(continuous_z=True)
        pb=torch.from_numpy(bundle_np).to(DEV)
        odir=os.path.join(OUT,"miitt",subj); os.makedirs(odir,exist_ok=True)
        print(f"\n=== s20contz MULTIFRAME {subj}  bbox={bbox.tolist()} ===", flush=True)
        for tag,breathing in [("clean",False),("normal",True)]:
            cap=capture(model, pb, bbox, breathing, rn, "multiframe", clean_ref=True,
                        seq_index=SEED, max_phases=MAXPH)
            RE,GT=cap["RE"],cap["GT"]; ref=cap["z_mid"]
            re_a,gt_a=alive(RE,GT),alive(GT,GT)
            nonref=[q for q in range(12) if q!=ref and gt_a[q]>1e-4]
            m={k:float(np.nanmean(v)) for k,v in cap["metr"].items()}
            base=os.path.join(odir,f"{tag}_7row")
            render_7row(cap, f"s20contz MULTIFRAME | miitt/{subj} | {tag}", base+".gif", dpi=130)
            np.savez_compressed(base+".npz", gt=GT, recon=RE, inp=cap["IN"], dvf=cap["DV"],
                cov=cap["CO"], has_slot=np.array(cap["has_slot"]), ref_zmid=ref,
                zbr=cap["rd"][:,0], sop=np.array(cap["sop"]), applied_disp=cap["rd"],
                per_phase_motion=np.array(cap["metr"]["motion"]),
                per_phase_full=np.array(cap["metr"]["full"]),
                per_phase_ssim=np.array(cap["metr"]["ssim"]))
            print(f"  [{tag:6s}] S_slots={cap['rd'].shape[0]} | motion={m['motion']:.2f} full={m['full']:.2f}dB ssim={m['ssim']:.3f}", flush=True)
            print(f"           ALIVENESS ref-plane z{ref}: RE {re_a[ref]:.3f} / GT {gt_a[ref]:.3f} ({re_a[ref]/max(gt_a[ref],1e-6)*100:.0f}%)", flush=True)
            print(f"           ALIVENESS non-ref mean:     RE {np.mean([re_a[q] for q in nonref]):.3f} / GT {np.mean([gt_a[q] for q in nonref]):.3f} ({np.mean([re_a[q] for q in nonref])/max(np.mean([gt_a[q] for q in nonref]),1e-6)*100:.0f}%)", flush=True)
    print("DONE", flush=True)

if __name__=="__main__":
    main()
