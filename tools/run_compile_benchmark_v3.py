#!/usr/bin/env python3
"""
tools/run_compile_benchmark_v3.py

GPU comparison benchmark: L40S vs A40 reference.
Each experiment runs as a fresh subprocess — zero fragmentation.

Design choices vs v2:
  - S=12, dynamic=True throughout (production-correct: one_frame_per_slice=true, Z varies 6-14)
  - Drops redundant experiments; adds L40S-specific precision knob
  - Report embeds A40 S=10 reference numbers for comparison

Experiment set (8 experiments):
  E0_Fwd    Eager forward baseline                         [S=12, no compile]
  E0_Train  Eager full train step baseline                 [S=12, no compile]
  E1        48 blocks default, dynamic=True                [fundamental compile lever]
  E2        48 blocks max-autotune-no-cudagraphs           [L40S may find better tiles]
  E3        48 blocks default + fused AdamW                [production config]
  E4        48 blocks max-autotune + fused AdamW           [L40S max config]
  E5        matmul_precision='medium' only                 [L40S 4th-gen tensor core knob]
  E6        48 blocks default + fused AdamW + medium prec  [combined best candidate]

Usage:
  PYTHONPATH=training:. python tools/run_compile_benchmark_v3.py
  PYTHONPATH=training:. python tools/run_compile_benchmark_v3.py --exp E3
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

# ─────────────────────────────────────────────────────────────────────────────
# A40 reference numbers (from v2 run at S=10, dynamic varies per exp)
# Included for qualitative comparison only — S differs (10 vs 12).
# ─────────────────────────────────────────────────────────────────────────────
A40_REF = {
    "E0_Fwd":   {"ms": 1025.43, "vram": 27.88},
    "E0_Train": {"ms": 3940.18, "vram": 24.66},
    "E1":       {"ms": 798.71,  "vram": 26.75, "speedup": "1.28x"},  # dynamic=False
    "E2":       {"ms": 806.49,  "vram": 26.75, "speedup": "1.27x"},  # max-autotune
    "E3":       {"ms": 3092.34, "vram": 23.53, "speedup": "1.27x"},
    "E4":       {"ms": 3074.66, "vram": 23.53, "speedup": "1.28x"},
    "E5":       None,  # not tested on A40
    "E6":       None,  # not tested on A40
}

# ─────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_ORDER = ["E0_Fwd", "E0_Train", "E1", "E2", "E3", "E4", "E5", "E6"]

EXPERIMENT_META = {
    "E0_Fwd":   {"name": "Eager Forward Baseline (S=12)",                            "is_train": False},
    "E0_Train": {"name": "Eager Train Step Baseline (S=12)",                          "is_train": True},
    "E1":       {"name": "48 Blocks default, dynamic=True",                           "is_train": False},
    "E2":       {"name": "48 Blocks max-autotune-no-cudagraphs, dynamic=True",        "is_train": False},
    "E3":       {"name": "48 Blocks default + Fused AdamW [PRODUCTION]",             "is_train": True},
    "E4":       {"name": "48 Blocks max-autotune-no-cudagraphs + Fused AdamW",        "is_train": True},
    "E5":       {"name": "matmul_precision=medium only (L40S 4th-gen tensor core)",   "is_train": False},
    "E6":       {"name": "48 Blocks default + Fused AdamW + precision=medium",        "is_train": True},
}

NUM_WARMUP_DEFAULT  = 10
NUM_WARMUP_AUTOTUNE = 35
NUM_BENCH = 25
S = 10   # matches A40 v2 benchmark for apples-to-apples GPU comparison


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (subprocess mode)
# ─────────────────────────────────────────────────────────────────────────────
def build_batch(device, S=12, B=1, H=518, W=518, D=12):
    torch.manual_seed(42)
    images  = torch.rand(B, S, 3, H, W, device=device, dtype=torch.float32)
    gt      = torch.rand(B, D, 256, 256, device=device, dtype=torch.float32)
    z_raw   = torch.randint(0, D, (B, S), device=device, dtype=torch.float32)
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


def make_model(device):
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
    return m


def fwd_loss(m, b, grid=(12, 256, 256)):
    from training.loss import compute_volume_intensity_loss
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = m(b["images"], batch=b)
        d = compute_volume_intensity_loss(preds, b, grid)
        return d["loss_volume"] + 0.1 * d["loss_pos_tv"], preds


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


def compile_blocks(m, mode, dynamic=True):
    for i in range(len(m.aggregator.frame_blocks)):
        m.aggregator.frame_blocks[i]  = torch.compile(m.aggregator.frame_blocks[i],  mode=mode, dynamic=dynamic)
        m.aggregator.global_blocks[i] = torch.compile(m.aggregator.global_blocks[i], mode=mode, dynamic=dynamic)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment implementations (subprocess mode)
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(exp_id):
    device = "cuda"
    b = build_batch(device, S=S)
    meta = EXPERIMENT_META[exp_id]
    is_autotune = "autotune" in meta["name"].lower()
    num_warmup  = NUM_WARMUP_AUTOTUNE if is_autotune else NUM_WARMUP_DEFAULT

    # Default: high precision (TF32 for FP32 matmuls, BF16 via autocast for BF16)
    torch.set_float32_matmul_precision("high")

    m = make_model(device)

    if exp_id == "E0_Fwd":
        run = lambda batch: fwd_loss(m, batch)

    elif exp_id == "E0_Train":
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        def run(batch):
            opt.zero_grad()
            loss, preds = fwd_loss(m, batch)
            loss.backward()
            opt.step()
            return loss, preds

    elif exp_id == "E1":
        compile_blocks(m, mode="default", dynamic=True)
        run = lambda batch: fwd_loss(m, batch)

    elif exp_id == "E2":
        compile_blocks(m, mode="max-autotune-no-cudagraphs", dynamic=True)
        run = lambda batch: fwd_loss(m, batch)

    elif exp_id == "E3":
        compile_blocks(m, mode="default", dynamic=True)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
        def run(batch):
            opt.zero_grad()
            loss, preds = fwd_loss(m, batch)
            loss.backward()
            opt.step()
            return loss, preds

    elif exp_id == "E4":
        compile_blocks(m, mode="max-autotune-no-cudagraphs", dynamic=True)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
        def run(batch):
            opt.zero_grad()
            loss, preds = fwd_loss(m, batch)
            loss.backward()
            opt.step()
            return loss, preds

    elif exp_id == "E5":
        # L40S 4th-gen tensor core knob: allow BF16 reducers in FP32 matmuls
        # (DPT point_head runs FP32; this may speed it up slightly on Ada Lovelace)
        torch.set_float32_matmul_precision("medium")
        run = lambda batch: fwd_loss(m, batch)

    elif exp_id == "E6":
        # Combined best candidate: blocks + fused AdamW + medium precision
        torch.set_float32_matmul_precision("medium")
        compile_blocks(m, mode="default", dynamic=True)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
        def run(batch):
            opt.zero_grad()
            loss, preds = fwd_loss(m, batch)
            loss.backward()
            opt.step()
            return loss, preds

    else:
        raise ValueError(f"Unknown exp_id: {exp_id}")

    compile_sec, avg_ms, peak_gb, loss_val = bench(run, b, num_warmup, NUM_BENCH)

    result = {
        "exp_id":               exp_id,
        "exp_name":             meta["name"],
        "is_train":             meta["is_train"],
        "status":               "SUCCESS",
        "gpu":                  torch.cuda.get_device_name(0),
        "S":                    S,
        "compilation_time_sec": round(compile_sec, 3),
        "avg_step_ms":          round(avg_ms, 2),
        "iters_per_sec":        round(1000.0 / avg_ms, 2),
        "peak_vram_gb":         round(peak_gb, 2),
        "loss":                 round(loss_val, 6),
    }
    print(json.dumps(result))


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def orchestrate():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 70)
    print(f"  VGGT-MRI torch.compile Benchmark v3  —  GPU Comparison")
    print(f"  GPU: {gpu} | PyTorch: {torch.__version__}")
    print(f"  S={S} (production), dynamic=True | {len(EXPERIMENT_ORDER)} experiments")
    print("=" * 70)
    print()

    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    results = []
    eager_fwd_ms   = None
    eager_train_ms = None

    for exp_id in EXPERIMENT_ORDER:
        meta = EXPERIMENT_META[exp_id]
        ref  = A40_REF.get(exp_id)
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
                res = {"exp_id": exp_id, "exp_name": meta["name"], "status": "PARSE_ERROR",
                       "error": proc.stdout[-300:], "avg_step_ms": 0.0, "peak_vram_gb": 0.0,
                       "compilation_time_sec": 0.0, "loss": 0.0}
        else:
            err_lines = [l for l in (proc.stderr or proc.stdout or "").split("\n") if l.strip()]
            res = {"exp_id": exp_id, "exp_name": meta["name"], "status": "FAILED",
                   "error": (err_lines[-1][:200] if err_lines else "unknown"),
                   "avg_step_ms": 0.0, "peak_vram_gb": 0.0,
                   "compilation_time_sec": 0.0, "loss": 0.0}

        res["wall_time_sec"] = round(wall_sec, 1)

        if res.get("status") == "SUCCESS":
            if exp_id == "E0_Fwd":
                eager_fwd_ms = res["avg_step_ms"]
                res["speedup_vs_eager"] = "1.00x"
            elif exp_id == "E0_Train":
                eager_train_ms = res["avg_step_ms"]
                res["speedup_vs_eager"] = "1.00x"
            elif not meta["is_train"] and eager_fwd_ms:
                res["speedup_vs_eager"] = f"{eager_fwd_ms / res['avg_step_ms']:.2f}x"
            elif meta["is_train"] and eager_train_ms:
                res["speedup_vs_eager"] = f"{eager_train_ms / res['avg_step_ms']:.2f}x"
            else:
                res["speedup_vs_eager"] = "N/A"

            # GPU comparison vs A40
            if ref and ref.get("ms"):
                res["vs_a40_ms"]      = ref["ms"]
                res["l40s_vs_a40"]    = f"{ref['ms'] / res['avg_step_ms']:.2f}x"
            else:
                res["vs_a40_ms"]   = None
                res["l40s_vs_a40"] = "N/A (new)"

            a40_str = f"  A40={ref['ms']}ms" if ref and ref.get("ms") else ""
            print(f"  {res['avg_step_ms']} ms/step  compile-speedup={res['speedup_vs_eager']}  "
                  f"L40S-vs-A40={res['l40s_vs_a40']}  "
                  f"VRAM={res['peak_vram_gb']}GB  compile={res['compilation_time_sec']}s  "
                  f"wall={wall_sec:.0f}s{a40_str}")
        else:
            print(f"  FAILED: {res.get('error', '')[:120]}")

        results.append(res)
        print()

    # Save JSON
    os.makedirs("scratch", exist_ok=True)
    json_path = "scratch/compile_benchmark_v3_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Markdown report
    md_path = "scratch/compile_benchmark_v3_report.md"
    with open(md_path, "w") as f:
        f.write(f"# VGGT-MRI torch.compile Benchmark v3 — GPU Comparison\n\n")
        f.write(f"**New GPU**: {gpu} | **PyTorch**: {torch.__version__} | S={S}, dynamic=True\n\n")
        f.write(f"**A40 reference** (v2, S=10): included for comparison. S differs — speedup *ratios* comparable, absolute ms are not.\n\n")
        f.write("| Exp | Name | L40S ms/step | L40S speedup | L40S vs A40 | A40 ms (S=10) | L40S VRAM | Compile (s) |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            a40_ms  = r.get("vs_a40_ms") or "—"
            l40s_vs = r.get("l40s_vs_a40", "—")
            spd     = r.get("speedup_vs_eager", "—")
            if r.get("status") == "SUCCESS":
                f.write(f"| {r['exp_id']} | {r['exp_name']} "
                        f"| {r['avg_step_ms']} ms | {spd} | {l40s_vs} "
                        f"| {a40_ms} ms | {r['peak_vram_gb']} GB | {r['compilation_time_sec']}s |\n")
            else:
                f.write(f"| {r['exp_id']} | {r['exp_name']} | FAILED | — | — | {a40_ms} | — | — |\n")

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
