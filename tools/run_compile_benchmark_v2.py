#!/usr/bin/env python3
"""
tools/run_compile_benchmark_v2.py

Clean per-process benchmark runner for VGGT-MRI torch.compile experiments.
Each experiment = fresh subprocess = fresh CUDA context = zero fragmentation carryover.

Experiment set (12 cells, 1 knob each, all viable on 44 GB A40):
  E0_Fwd   Eager forward baseline (no compile)
  E0_Train Eager full train step baseline (fwd + bwd + AdamW)
  E1       48 blocks default, dynamic=False      [fundamental compile lever]
  E2       48 blocks default, dynamic=True       [realistic: S varies 8-20 in training]
  E3       DPT point_head only, default          [head-only compile]
  E4       DINO patch_embed only, default        [backbone embed compile]
  E5       48 blocks max-autotune-no-cudagraphs  [best mode that respects checkpointing]
  E6       48 blocks + DPT head, default         [combined submodules]
  E7       48 blocks + DPT head, max-autotune    [combined + best mode]
  E8       Fused AdamW only (no model compile)   [optimizer knob]
  E9       48 blocks default + fused AdamW        [compile + optimizer, train step]
  E10      48 blocks max-autotune + fused AdamW  [production target, train step]

Usage:
  # Run all (orchestrator mode):
  PYTHONPATH=training:. python tools/run_compile_benchmark_v2.py

  # Run a single experiment (subprocess mode, called by orchestrator automatically):
  PYTHONPATH=training:. python tools/run_compile_benchmark_v2.py --exp E1
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
# Experiment registry
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_ORDER = [
    "E0_Fwd", "E0_Train",
    "E1", "E2", "E3", "E4", "E5", "E6", "E7",
    "E8", "E9", "E10",
]

EXPERIMENT_META = {
    "E0_Fwd":   {"name": "Eager Forward Baseline",                           "is_train": False},
    "E0_Train": {"name": "Eager Train Step Baseline",                         "is_train": True},
    "E1":       {"name": "48 Blocks default (dynamic=False)",                 "is_train": False},
    "E2":       {"name": "48 Blocks default (dynamic=True)",                  "is_train": False},
    "E3":       {"name": "DPT Head default",                                  "is_train": False},
    "E4":       {"name": "DINO patch_embed default",                          "is_train": False},
    "E5":       {"name": "48 Blocks max-autotune-no-cudagraphs",              "is_train": False},
    "E6":       {"name": "48 Blocks + DPT Head default",                      "is_train": False},
    "E7":       {"name": "48 Blocks + DPT Head max-autotune-no-cudagraphs",   "is_train": False},
    "E8":       {"name": "Fused AdamW only",                                  "is_train": True},
    "E9":       {"name": "48 Blocks default + Fused AdamW",                   "is_train": True},
    "E10":      {"name": "48 Blocks max-autotune-no-cudagraphs + Fused AdamW","is_train": True},
}

NUM_WARMUP_DEFAULT  = 10
NUM_WARMUP_AUTOTUNE = 35
NUM_BENCH = 25
S = 10  # slots per sample


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (used in subprocess mode)
# ─────────────────────────────────────────────────────────────────────────────
def build_batch(device, S=10, B=1, H=518, W=518, D=12):
    torch.manual_seed(42)
    images = torch.rand(B, S, 3, H, W, device=device, dtype=torch.float32)
    gt = torch.rand(B, D, 256, 256, device=device, dtype=torch.float32)
    z_raw = torch.randint(0, D, (B, S), device=device, dtype=torch.float32)
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
    bbox = torch.tensor([[0, D, 0, 256, 0, 256]], device=device, dtype=torch.int64)
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
    """Measure: compile time (first call), then warmup, then CUDA-event timing."""
    # First call: JIT compilation happens here
    t0 = time.perf_counter()
    loss_val, _ = run_fn(batch)
    torch.cuda.synchronize()
    compile_sec = time.perf_counter() - t0

    # Warmup: let Triton autotuning and kernel caching settle
    for _ in range(num_warmup):
        loss_val, _ = run_fn(batch)
    torch.cuda.synchronize()

    # Measure steady-state with hardware CUDA Events
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
    loss_f  = float(loss_val.detach().cpu())
    return compile_sec, avg_ms, peak_gb, loss_f


# ─────────────────────────────────────────────────────────────────────────────
# Experiment implementations (subprocess mode)
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(exp_id):
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    b = build_batch(device, S=S)
    meta = EXPERIMENT_META[exp_id]
    is_autotune = "autotune" in meta["name"].lower()
    num_warmup  = NUM_WARMUP_AUTOTUNE if is_autotune else NUM_WARMUP_DEFAULT

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
        # 48 blocks individually compiled — preserves gradient checkpointing
        for i in range(len(m.aggregator.frame_blocks)):
            m.aggregator.frame_blocks[i] = torch.compile(
                m.aggregator.frame_blocks[i], mode="default", dynamic=False)
            m.aggregator.global_blocks[i] = torch.compile(
                m.aggregator.global_blocks[i], mode="default", dynamic=False)
        run = lambda batch: fwd_loss(m, batch)

    elif exp_id == "E2":
        # Same as E1 but dynamic=True — handles variable S in training
        for i in range(len(m.aggregator.frame_blocks)):
            m.aggregator.frame_blocks[i] = torch.compile(
                m.aggregator.frame_blocks[i], mode="default", dynamic=True)
            m.aggregator.global_blocks[i] = torch.compile(
                m.aggregator.global_blocks[i], mode="default", dynamic=True)
        run = lambda batch: fwd_loss(m, batch)

    elif exp_id == "E3":
        # DPT point_head only
        m.point_head = torch.compile(m.point_head, mode="default")
        run = lambda batch: fwd_loss(m, batch)

    elif exp_id == "E4":
        # DINO patch_embed only (frozen in aggft, but still worth timing)
        m.aggregator.patch_embed = torch.compile(m.aggregator.patch_embed, mode="default")
        run = lambda batch: fwd_loss(m, batch)

    elif exp_id == "E5":
        # 48 blocks max-autotune-no-cudagraphs — best mode without breaking backward
        for i in range(len(m.aggregator.frame_blocks)):
            m.aggregator.frame_blocks[i] = torch.compile(
                m.aggregator.frame_blocks[i], mode="max-autotune-no-cudagraphs")
            m.aggregator.global_blocks[i] = torch.compile(
                m.aggregator.global_blocks[i], mode="max-autotune-no-cudagraphs")
        run = lambda batch: fwd_loss(m, batch)

    elif exp_id == "E6":
        # 48 blocks + DPT head combined, default
        for i in range(len(m.aggregator.frame_blocks)):
            m.aggregator.frame_blocks[i] = torch.compile(
                m.aggregator.frame_blocks[i], mode="default", dynamic=False)
            m.aggregator.global_blocks[i] = torch.compile(
                m.aggregator.global_blocks[i], mode="default", dynamic=False)
        m.point_head = torch.compile(m.point_head, mode="default")
        run = lambda batch: fwd_loss(m, batch)

    elif exp_id == "E7":
        # 48 blocks + DPT head combined, max-autotune-no-cudagraphs
        for i in range(len(m.aggregator.frame_blocks)):
            m.aggregator.frame_blocks[i] = torch.compile(
                m.aggregator.frame_blocks[i], mode="max-autotune-no-cudagraphs")
            m.aggregator.global_blocks[i] = torch.compile(
                m.aggregator.global_blocks[i], mode="max-autotune-no-cudagraphs")
        m.point_head = torch.compile(m.point_head, mode="max-autotune-no-cudagraphs")
        run = lambda batch: fwd_loss(m, batch)

    elif exp_id == "E8":
        # Fused AdamW only — no model compile, pure optimizer knob
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
        def run(batch):
            opt.zero_grad()
            loss, preds = fwd_loss(m, batch)
            loss.backward()
            opt.step()
            return loss, preds

    elif exp_id == "E9":
        # 48 blocks default + fused AdamW (full train step)
        for i in range(len(m.aggregator.frame_blocks)):
            m.aggregator.frame_blocks[i] = torch.compile(
                m.aggregator.frame_blocks[i], mode="default", dynamic=False)
            m.aggregator.global_blocks[i] = torch.compile(
                m.aggregator.global_blocks[i], mode="default", dynamic=False)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
        def run(batch):
            opt.zero_grad()
            loss, preds = fwd_loss(m, batch)
            loss.backward()
            opt.step()
            return loss, preds

    elif exp_id == "E10":
        # 48 blocks max-autotune-no-cudagraphs + fused AdamW — production target
        for i in range(len(m.aggregator.frame_blocks)):
            m.aggregator.frame_blocks[i] = torch.compile(
                m.aggregator.frame_blocks[i], mode="max-autotune-no-cudagraphs")
            m.aggregator.global_blocks[i] = torch.compile(
                m.aggregator.global_blocks[i], mode="max-autotune-no-cudagraphs")
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
        "exp_id":              exp_id,
        "exp_name":            meta["name"],
        "is_train":            meta["is_train"],
        "status":              "SUCCESS",
        "compilation_time_sec": round(compile_sec, 3),
        "avg_step_ms":         round(avg_ms, 2),
        "iters_per_sec":       round(1000.0 / avg_ms, 2),
        "peak_vram_gb":        round(peak_gb, 2),
        "loss":                round(loss_val, 6),
    }
    # Print JSON as last line of stdout — orchestrator parses this
    print(json.dumps(result))


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator (runs all experiments sequentially, each in its own subprocess)
# ─────────────────────────────────────────────────────────────────────────────
def orchestrate():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 70)
    print(f"  VGGT-MRI torch.compile Benchmark v2")
    print(f"  GPU: {gpu} | PyTorch: {torch.__version__}")
    print(f"  {len(EXPERIMENT_ORDER)} experiments — 1 fresh subprocess each (zero fragmentation)")
    print("=" * 70)
    print()

    # Each subprocess inherits PYTHONPATH + expandable_segments to prevent fragmentation
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    results = []
    eager_fwd_ms   = None
    eager_train_ms = None

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
                res = {
                    "exp_id": exp_id, "exp_name": meta["name"],
                    "status": "PARSE_ERROR", "error": proc.stdout[-300:],
                    "avg_step_ms": 0.0, "peak_vram_gb": 0.0,
                    "compilation_time_sec": 0.0, "loss": 0.0,
                }
        else:
            # Capture the last meaningful error line from stderr
            err_lines = [l for l in (proc.stderr or proc.stdout or "").split("\n") if l.strip()]
            err_msg = err_lines[-1][:200] if err_lines else "unknown error"
            res = {
                "exp_id": exp_id, "exp_name": meta["name"],
                "status": "FAILED", "error": err_msg,
                "avg_step_ms": 0.0, "peak_vram_gb": 0.0,
                "compilation_time_sec": 0.0, "loss": 0.0,
            }

        res["wall_time_sec"] = round(wall_sec, 1)

        # Compute speedup vs appropriate baseline
        if res.get("status") == "SUCCESS":
            if exp_id == "E0_Fwd":
                eager_fwd_ms = res["avg_step_ms"]
                res["speedup"] = "1.00x"
            elif exp_id == "E0_Train":
                eager_train_ms = res["avg_step_ms"]
                res["speedup"] = "1.00x"
            elif meta["is_train"] and eager_train_ms:
                res["speedup"] = f"{eager_train_ms / res['avg_step_ms']:.2f}x"
            elif not meta["is_train"] and eager_fwd_ms:
                res["speedup"] = f"{eager_fwd_ms / res['avg_step_ms']:.2f}x"
            else:
                res["speedup"] = "N/A"

            print(f"  {res['avg_step_ms']} ms/step  {res['speedup']}  "
                  f"compile={res['compilation_time_sec']}s  "
                  f"VRAM={res['peak_vram_gb']}GB  wall={wall_sec:.0f}s")
        else:
            print(f"  FAILED: {res.get('error', '')[:120]}")

        results.append(res)
        print()

    # Save JSON
    os.makedirs("scratch", exist_ok=True)
    json_path = "scratch/compile_benchmark_v2_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Markdown report
    md_path = "scratch/compile_benchmark_v2_report.md"
    with open(md_path, "w") as f:
        f.write("# VGGT-MRI torch.compile Benchmark v2\n\n")
        f.write(f"**GPU**: {gpu} | **PyTorch**: {torch.__version__}\n\n")
        f.write("Each experiment ran in a **fresh subprocess** — zero cross-experiment fragmentation.\n\n")
        f.write("| Exp | Name | ms/step | Speedup | Compile (s) | VRAM (GB) | Loss | Wall (s) |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            if r.get("status") == "SUCCESS":
                f.write(
                    f"| {r['exp_id']} | {r['exp_name']} "
                    f"| {r['avg_step_ms']} ms | {r.get('speedup','N/A')} "
                    f"| {r['compilation_time_sec']}s | {r['peak_vram_gb']} GB "
                    f"| {r['loss']} | {r.get('wall_time_sec','?')}s |\n"
                )
            else:
                f.write(
                    f"| {r['exp_id']} | {r['exp_name']} "
                    f"| FAILED | — | — | — "
                    f"| {r.get('error','')[:60]} | — |\n"
                )

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None,
                        help="Single experiment ID (subprocess mode, called by orchestrator)")
    args = parser.parse_args()

    if args.exp:
        run_experiment(args.exp)
    else:
        orchestrate()
