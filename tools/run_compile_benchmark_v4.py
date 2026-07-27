#!/usr/bin/env python3
"""
tools/run_compile_benchmark_v4.py

Benchmark: additional speed levers on top of the recommended config
(48 blocks default + fused AdamW = 1761 ms/step on L40S).

Profile findings (tools/profile_train_step.py, L40S):
  - FA2 already active (confirmed by pytorch_flash::flash_fwd_kernel)
  - GradScaler already disabled for BF16
  - DPT convolutions: 48ms fwd + 89ms bwd = 137ms/step (7.7%)
    → channels_last targets these
  - Splat+loss: NOT in top-20 GPU ops (< 43ms) — already fast.
    Compiling reduces CPU "Command Buffer Full" stall (873ms CPU/step) more than GPU time.
  - Attention blocks dominate: ~750ms/step — already compiled.

Experiments (all include the recommended baseline: 48 blocks + fused AdamW):
  E0_Train  Recommended baseline [48 blocks default + fused AdamW]
  E1        + compile loss function
  E2        + channels_last memory format
  E3        + compile loss + channels_last (combined)

DataLoader item (pin_memory=True, persistent_workers=True) cannot be measured
in a synthetic benchmark — apply as a config change (see recommendation below).

Usage:
  PYTHONPATH=training:. python tools/run_compile_benchmark_v4.py
  PYTHONPATH=training:. python tools/run_compile_benchmark_v4.py --exp E1
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("training"))

EXPERIMENT_ORDER = ["E0_Train", "E1", "E2", "E3"]

EXPERIMENT_META = {
    "E0_Train": {"name": "Recommended baseline [48 blocks + fused AdamW]"},
    "E1":       {"name": "Baseline + compile loss fn"},
    "E2":       {"name": "Baseline + channels_last"},
    "E3":       {"name": "Baseline + compile loss + channels_last"},
}

# L40S v3 E3 result for reference
BASELINE_REF_MS = 1760.80

NUM_WARMUP = 10
NUM_BENCH  = 25
S = 10


def build_batch(device, S=10, B=1, H=518, W=518, D=12):
    torch.manual_seed(42)
    images    = torch.rand(B, S, 3, H, W, device=device, dtype=torch.float32)
    gt        = torch.rand(B, D, 256, 256, device=device, dtype=torch.float32)
    z_raw     = torch.randint(0, D, (B, S), device=device, dtype=torch.float32)
    z_indices = (z_raw / (D - 1)) * 2.0 - 1.0
    t_indices = torch.randint(0, 12, (B, S), device=device, dtype=torch.int64)
    px = torch.linspace(-1.0, 1.0, W, device=device)
    py = torch.linspace(-1.0, 1.0, H, device=device)
    gy, gx = torch.meshgrid(py, px, indexing="ij")
    sc = torch.zeros(B, S, H, W, 3, device=device)
    for b in range(B):
        for s in range(S):
            sc[b, s, :, :, 0] = gx
            sc[b, s, :, :, 1] = gy
            sc[b, s, :, :, 2] = z_indices[b, s]
    bbox   = torch.tensor([[0, D, 0, 256, 0, 256]], device=device, dtype=torch.int64)
    phases = torch.rand(B, 12, D, 256, 256, device=device, dtype=torch.float32)
    return {
        "images": images, "gt_target_volume": gt, "z_indices": z_indices,
        "t_indices": t_indices, "scanner_coords": sc, "anatomy_bbox": bbox, "phases": phases,
    }


def make_model(device, channels_last=False):
    from vggt.models.vggt import VGGT
    from vggt.utils.checkpoint_stage import stage_checkpoint_to_local
    torch.manual_seed(42)
    m = VGGT(
        depth=24, embed_dim=1024, num_heads=16,
        use_z_pose_embedding=True, reference_slot=True,
        use_reference_token=True, train_on_residual_dvf=True,
    ).to(device).train()
    ckpt = "scratch/base_weights/vggt1b_base.pt"
    if os.path.exists(ckpt):
        staged = stage_checkpoint_to_local(ckpt)
        sd = torch.load(staged, map_location=device, weights_only=True)
        m.load_state_dict(sd.get("model", sd), strict=False)
    if channels_last:
        for module in m.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                module.to(memory_format=torch.channels_last)
    # Always compile the 48 blocks (recommended baseline)
    for i in range(len(m.aggregator.frame_blocks)):
        m.aggregator.frame_blocks[i]  = torch.compile(m.aggregator.frame_blocks[i],  mode="default", dynamic=True)
        m.aggregator.global_blocks[i] = torch.compile(m.aggregator.global_blocks[i], mode="default", dynamic=True)
    return m


def bench(run_fn, batch, num_warmup, num_bench):
    t0 = time.perf_counter()
    loss_val, _ = run_fn(batch)
    torch.cuda.synchronize()
    compile_sec = time.perf_counter() - t0

    for _ in range(num_warmup):
        loss_val, _ = run_fn(batch)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_bench):
        loss_val, _ = run_fn(batch)
    end.record()
    torch.cuda.synchronize()

    avg_ms  = start.elapsed_time(end) / num_bench
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return compile_sec, avg_ms, peak_gb, float(loss_val.detach().cpu())


def run_experiment(exp_id):
    torch.set_float32_matmul_precision("high")
    import torch._inductor.config as inductor_config
    if not hasattr(inductor_config, "cpp"):
        pass
    else:
        try:
            inductor_config.cpp.cxx_flags = "-std=c++2a"
        except Exception:
            pass
    device = "cuda"

    from training.loss import compute_volume_intensity_loss

    channels_last = exp_id in ("E2", "E3")
    compile_loss  = exp_id in ("E1", "E3")

    b   = build_batch(device, S=S)
    m   = make_model(device, channels_last=channels_last)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)

    if compile_loss:
        loss_fn = torch.compile(compute_volume_intensity_loss, mode="default", dynamic=True)
    else:
        loss_fn = compute_volume_intensity_loss

    def run(batch):
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = m(batch["images"], batch=batch)
            d     = loss_fn(preds, batch, (12, 256, 256))
            loss  = d["loss_volume"] + 0.1 * d["loss_pos_tv"]
        loss.backward()
        opt.step()
        return loss, preds

    compile_sec, avg_ms, peak_gb, loss_val = bench(run, b, NUM_WARMUP, NUM_BENCH)

    result = {
        "exp_id":               exp_id,
        "exp_name":             EXPERIMENT_META[exp_id]["name"],
        "status":               "SUCCESS",
        "gpu":                  torch.cuda.get_device_name(0),
        "channels_last":        channels_last,
        "compile_loss":         compile_loss,
        "compilation_time_sec": round(compile_sec, 3),
        "avg_step_ms":          round(avg_ms, 2),
        "iters_per_sec":        round(1000.0 / avg_ms, 2),
        "peak_vram_gb":         round(peak_gb, 2),
        "loss":                 round(loss_val, 6),
        "speedup_vs_baseline":  round(BASELINE_REF_MS / avg_ms, 3),
    }
    print(json.dumps(result))


def orchestrate():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 70)
    print(f"  VGGT-MRI Benchmark v4 — Additional speed levers")
    print(f"  GPU: {gpu} | PyTorch: {torch.__version__} | S={S}")
    print(f"  Baseline ref: {BASELINE_REF_MS} ms/step (L40S v3 E3)")
    print("=" * 70)
    print()

    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    results = []

    for exp_id in EXPERIMENT_ORDER:
        meta = EXPERIMENT_META[exp_id]
        print(f"[{exp_id}] {meta['name']}", flush=True)

        t_wall = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--exp", exp_id],
            capture_output=True, text=True, env=env,
        )
        wall_sec = time.perf_counter() - t_wall

        if proc.returncode == 0:
            lines = [l for l in proc.stdout.strip().split("\n") if l.strip()]
            try:
                res = json.loads(lines[-1])
            except Exception:
                res = {"exp_id": exp_id, "exp_name": meta["name"], "status": "PARSE_ERROR", "error": proc.stdout[-300:],
                       "avg_step_ms": 0.0, "speedup_vs_baseline": 0.0}
        else:
            err = [l for l in (proc.stderr or proc.stdout or "").split("\n") if l.strip()]
            res = {"exp_id": exp_id, "exp_name": meta["name"], "status": "FAILED",
                   "error": (err[-1][:200] if err else "unknown"),
                   "avg_step_ms": 0.0, "speedup_vs_baseline": 0.0}

        res["wall_time_sec"] = round(wall_sec, 1)

        if res.get("status") == "SUCCESS":
            print(f"  {res['avg_step_ms']} ms/step  "
                  f"speedup-vs-baseline={res['speedup_vs_baseline']}x  "
                  f"VRAM={res['peak_vram_gb']} GB  compile={res['compilation_time_sec']}s  "
                  f"wall={wall_sec:.0f}s")
        else:
            print(f"  FAILED: {res.get('error','')[:120]}")

        results.append(res)
        print()

    os.makedirs("scratch", exist_ok=True)
    json_path = "scratch/compile_benchmark_v4_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    md_path = "scratch/compile_benchmark_v4_report.md"
    with open(md_path, "w") as f:
        f.write(f"# VGGT-MRI Benchmark v4 — Additional Speed Levers\n\n")
        f.write(f"**GPU**: {gpu} | **PyTorch**: {torch.__version__} | S={S}\n\n")
        f.write(f"**Baseline**: {BASELINE_REF_MS} ms/step (L40S v3 E3: 48 blocks + fused AdamW)\n\n")
        f.write("| Exp | Name | ms/step | vs Baseline | VRAM | Compile (s) |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: |\n")
        for r in results:
            if r.get("status") == "SUCCESS":
                f.write(f"| {r['exp_id']} | {r['exp_name']} "
                        f"| {r['avg_step_ms']} ms | {r['speedup_vs_baseline']}x "
                        f"| {r['peak_vram_gb']} GB | {r['compilation_time_sec']}s |\n")
            else:
                f.write(f"| {r['exp_id']} | {r['exp_name']} | FAILED | — | — | — |\n")
        f.write("\n**Note**: DataLoader `pin_memory=True` + `persistent_workers=True` not "
                "benchmarkable synthetically. Enable in config: `training/config/mri_finetune.yaml`.\n")

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None)
    args = parser.parse_args()
    if args.exp:
        run_experiment(args.exp)
    else:
        orchestrate()
