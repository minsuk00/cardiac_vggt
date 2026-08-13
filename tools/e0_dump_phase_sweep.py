"""E0 dump driver — decode one checkpoint into a full 12-phase val sweep (docs/66 campaign).

Runs the target codebase's OWN `mode=val` path 12 times (t_target_fixed=0..11) and archives
each run's ${log_dir}/val_volumes/ dumps. Every checkpoint is decoded by the tree that trained
it (--tree), so old-code ckpts never pass through current z conventions and vice versa.

Why this is also the reference-swap phase-transfer probe: val slot sampling is a pure function
of seq_index (private random.Random(seq_index)), and t_target_fixed consumes no RNG, so the
non-reference slots (and the deterministic val respiratory corruption) are IDENTICAL across the
12 runs — only slot 0 (the target-phase reference) and V_gt change with k. The per-subject
LV(t) response curve therefore directly measures whether slot 0 controls the reconstruction.

Usage (current tree, CMRx24 val):
  micromamba run -n svr python tools/e0_dump_phase_sweep.py \
      --tree /home/minsukc/vggt --config default \
      --ckpt scratch/logs/<run>/ckpts/checkpoint_last.pt \
      --out result/e0_dumps/<name> --limit-val-batches 29 \
      --override split_file=training/splits/cmrx24only.txt \
      --override dataset_name=cmrx24only \
      --override ef_val_sweep=false --override logging.ef_eval_enable=false

Old worktree: same, with --tree <worktree> --config mri_volume_diffusion and WITHOUT the
ef_val_sweep/ef_eval_enable overrides (keys don't exist there).

Score the resulting dump dir with tools/e0_score_volumes.py.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys


def stage_ckpt(ckpt):
    """Copy the ckpt to node-local /tmp once (GPFS torch.load is pathologically slow)."""
    ckpt = os.path.abspath(ckpt)
    tag = hashlib.sha1(ckpt.encode()).hexdigest()[:10]
    dst = os.path.join("/tmp", f"e0_ckpt_{os.environ.get('USER','u')}",
                       f"{tag}_{os.path.basename(ckpt)}")
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(ckpt)):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"[e0] staging ckpt -> {dst}", flush=True)
        shutil.copy2(ckpt, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="repo/worktree root whose code decodes the ckpt")
    ap.add_argument("--config", required=True, help="hydra config name in that tree")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="dump dir; gets t00/..t11/ subdirs")
    ap.add_argument("--phases", default=",".join(str(k) for k in range(12)),
                    help="comma list of t_target phases (default 0..11)")
    ap.add_argument("--limit-val-batches", type=int, required=True,
                    help="number of val batches = number of val subjects (one pass)")
    ap.add_argument("--master-port", type=int, default=29601)
    ap.add_argument("--override", action="append", default=[],
                    help="extra hydra overrides, repeatable (tree-specific keys go here)")
    args = ap.parse_args()

    tree = os.path.abspath(args.tree)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    staged = stage_ckpt(args.ckpt)
    phases = [int(p) for p in args.phases.split(",")]

    env = dict(os.environ)
    # `temp` is load-bearing: the fixed12 hub arms resolve
    # `_target_=fixed12_dataset.MRIDatasetFixed12` from temp/, exactly as the training
    # sbatch scripts do (`PYTHONPATH=temp:training:.`). Omitting it makes every fixed12
    # arm die in hydra with ModuleNotFoundError('fixed12_dataset').
    env["PYTHONPATH"] = f"{os.path.join(tree, 'temp')}:{os.path.join(tree, 'training')}:{tree}"
    env["WANDB_MODE"] = "offline"

    for k in phases:
        dst = os.path.join(out, f"t{k:02d}")
        if os.path.isdir(dst) and any(f.endswith("_pred.nii.gz") for f in os.listdir(dst)):
            print(f"[e0] t{k:02d}: exists, skipping", flush=True)
            continue
        exp = f"e0tmp_{os.path.basename(out)}_t{k:02d}"
        log_dir = os.path.join(tree, "scratch", "logs", exp)
        # A leftover ckpts/ dir would win resume priority over --ckpt; refuse to reuse it.
        if os.path.isdir(os.path.join(log_dir, "ckpts")):
            shutil.rmtree(os.path.join(log_dir, "ckpts"))
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node=1", f"--master_port={args.master_port}",
            os.path.join(tree, "training", "launch.py"),
            "--config", args.config,
            "mode=val",
            f"t_target_fixed={k}",
            f"exp_name={exp}",
            f"checkpoint.resume_checkpoint_path={staged}",
            f"limit_val_batches={args.limit_val_batches}",
            "logging.save_val_volumes=true",
        ] + list(args.override)
        print(f"[e0] t{k:02d}: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=tree, env=env)
        vols = os.path.join(log_dir, "val_volumes")
        if r.returncode != 0 or not os.path.isdir(vols):
            raise RuntimeError(f"val run for t={k} failed (rc={r.returncode}, vols at {vols})")
        shutil.move(vols, dst)
        print(f"[e0] t{k:02d}: {len(os.listdir(dst))} files -> {dst}", flush=True)

    print(f"[e0] done: {out}", flush=True)


if __name__ == "__main__":
    main()
