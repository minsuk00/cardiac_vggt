"""Snapshot the 6 one-frame-ablation checkpoints as weights-only, before the runs resume.

A resume overwrites <log>/ckpts/checkpoint_last.pt, so these are staged now. Weights-only
({"model": sd}) is ~3.8 GB vs ~8.9 GB full and is all the eval harness needs. The training
epoch is asserted against the expected value and encoded into the filename, so the epoch
confound travels with the artifact.
"""
import os
import sys
import torch

LOGS = "/home/minsukc/vggt/scratch/logs"
OUT = "/home/minsukc/vggt/scratch/checkpoints"

# variant -> (log dir prefix, wandb id, expected training epoch)
RUNS = {
    "gather05":   ("216005406_mri_volume_diffusion_oneframe_baseline_gather05_dynamic_axial_Cine_combined", "fhkgalju", 39),
    "no_gather":  ("215996123_mri_volume_diffusion_oneframe_ctrl_nogather_dynamic_axial_Cine_combined",     "lmboejhq", 38),
    "contz":      ("216000753_mri_volume_diffusion_oneframe_gather05_contz_dynamic_axial_Cine_combined",    "tfz1x7ft", 39),
    "dino_ft":    ("216002704_mri_volume_diffusion_oneframe_dino_ft_lr2e5_dynamic_axial_Cine_combined",     "hlh3emae", 34),
    "aug_moderate": ("216003592_mri_volume_diffusion_oneframe_aug_moderate_dynamic_axial_Cine_combined",    "lylgvajs", 39),
    "lowdiff100": ("215949615_mri_volume_diffusion_oneframe_lowdiff100_dynamic_axial_Cine_combined",        "2kwj0tkd", 25),
}


def main():
    # First arg may be a date prefix (YYYYMMDD) for the output filename; default keeps the
    # original 2026-07-15 ep25-39 snapshot set. Pass a new date to stage a fresh snapshot
    # (e.g. the resumed ep50-60 checkpoints) WITHOUT overwriting the older set.
    args = sys.argv[1:]
    date = "20260715"
    if args and args[0].isdigit() and len(args[0]) == 8:
        date = args.pop(0)
    only = args or list(RUNS)
    for variant in only:
        prefix, wid, exp_epoch = RUNS[variant]
        src = os.path.join(LOGS, prefix, "ckpts", "checkpoint_last.pt")
        print(f"\n=== {variant} ({wid})  <- {src}", flush=True)
        ck = torch.load(src, map_location="cpu")

        # The trainer writes "prev_epoch" (= the epoch just completed) and resumes at
        # prev_epoch+1 (trainer.py:307,419). There is no "epoch" key.
        #
        # The checkpoint is the authority, NOT log.txt: "Saving checkpoint at epoch N" is printed
        # BEFORE the write, so a run killed mid-save (no_gather, dino_ft -- see their leftover
        # .tmp files) has a log claiming N while the last COMPLETE checkpoint holds N-1.
        # So: name the file from prev_epoch, and only warn on divergence from the log.
        epoch = ck["prev_epoch"]
        print(f"  keys={sorted(k for k in ck if k != 'model')}  prev_epoch={epoch}", flush=True)
        if not isinstance(epoch, int) or not (0 <= epoch <= 100):
            raise SystemExit(f"  !! implausible prev_epoch for {variant}: {epoch}")

        sd = ck["model"]
        dst = os.path.join(OUT, f"{date}_1frame_{variant}_ep{epoch}.pt")
        tmp = dst + ".tmp"
        torch.save({"model": sd}, tmp)
        os.replace(tmp, dst)  # atomic: never leave a half-written .pt at the real path

        # Verify the staged file independently of the in-memory copy.
        back = torch.load(dst, map_location="cpu")["model"]
        assert set(back) == set(sd), f"{variant}: key set changed on roundtrip"
        probe = sorted(sd)[len(sd) // 2]
        assert torch.equal(back[probe].float(), sd[probe].float()), f"{variant}: tensor mismatch at {probe}"
        print(f"  -> {dst}  ({os.path.getsize(dst)/1e9:.2f} GB, {len(back)} tensors, verified on {probe})", flush=True)
        del ck, sd, back


if __name__ == "__main__":
    main()
