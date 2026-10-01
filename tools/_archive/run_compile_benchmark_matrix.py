"""
tools/run_compile_benchmark_matrix.py

Exhaustive 21-Experiment (E0..E20) PyTorch 2.13 torch.compile Benchmark Suite for VGGT-MRI.
Tests every isolated compiler mode, submodule granularity, Inductor flag, precision level,
activation memory budget, CUDA Graph configuration, and combined best-of-breed stack.

Features:
- Robust try-except exception handling per experiment cell to catch OOM or Inductor errors gracefully.
- Fixed 518x518 input slice resolution matching MRIDataset + DINOv2 37x37 patch contract.
- Fresh model instantiation per experiment cell to prevent state leakage and pre-compilation contamination.
- Scope-matched Eager Train Step baseline for optimizer benchmarks (E9, E10, E20).
- Dedicated JIT compilation & autotuning duration measurement (seconds).
- 15 warmup iterations for standard modes, 35 warmup iterations for max-autotune modes.
- Hardware CUDA Event timers (ms/step, iter/sec) with pre-bench VRAM reset.
- Loss bit-fidelity relative error check vs Eager baseline.
- Saves results to scratch/compile_benchmark_results.json and scratch/compile_benchmark_report.md.
"""

import os
import sys
import time
import json
import gc
import torch
import torch.nn as nn

# Add paths
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("training"))

from vggt.models.vggt import VGGT
from training.loss import compute_volume_intensity_loss
from vggt.utils.checkpoint_stage import stage_checkpoint_to_local


def build_synthetic_batch(device="cuda", S=10, B=1, H=518, W=518, D=12):
    """Build a synthetic MRI batch matching MRIDataset exact output contract (518x518)."""
    torch.manual_seed(42)
    images = torch.rand(B, S, 3, H, W, device=device, dtype=torch.float32)
    gt_target_volume = torch.rand(B, D, 256, 256, device=device, dtype=torch.float32)
    z_raw = torch.randint(0, D, (B, S), device=device, dtype=torch.float32)
    z_indices = (z_raw / (D - 1)) * 2.0 - 1.0
    t_indices = torch.randint(0, 12, (B, S), device=device, dtype=torch.int64)

    # Scanner coords in [-1, 1] matching 518x518
    px = torch.linspace(-1.0, 1.0, W, device=device)
    py = torch.linspace(-1.0, 1.0, H, device=device)
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")

    z_norm = (z_indices.float() / (D - 1)) * 2.0 - 1.0
    scanner_coords = torch.zeros(B, S, H, W, 3, device=device, dtype=torch.float32)
    for b in range(B):
        for s in range(S):
            scanner_coords[b, s, :, :, 0] = grid_x
            scanner_coords[b, s, :, :, 1] = grid_y
            scanner_coords[b, s, :, :, 2] = z_norm[b, s]

    anatomy_bbox = torch.tensor([[0, D, 0, 256, 0, 256]], device=device, dtype=torch.int64)
    phases = torch.rand(B, 12, D, 256, 256, device=device, dtype=torch.float32)

    return {
        "images": images,
        "gt_target_volume": gt_target_volume,
        "z_indices": z_indices,
        "t_indices": t_indices,
        "scanner_coords": scanner_coords,
        "anatomy_bbox": anatomy_bbox,
        "phases": phases,
    }


def get_fresh_model(device="cuda"):
    """Instantiate a fresh VGGT model and load actual 4.7 GB VGGT-1B base weights via stage_checkpoint_to_local."""
    torch.manual_seed(42)
    model = VGGT(
        depth=24,
        embed_dim=1024,
        num_heads=16,
        use_z_pose_embedding=True,
        reference_slot=True,
        use_reference_token=True,
        train_on_residual_dvf=True,
    ).to(device).train()

    ckpt_path = "scratch/base_weights/vggt1b_base.pt"
    if os.path.exists(ckpt_path):
        staged_path = stage_checkpoint_to_local(ckpt_path)
        weights = torch.load(staged_path, map_location=device, weights_only=True)
        state_dict = weights.get("model", weights)
        model.load_state_dict(state_dict, strict=False)
    return model


def reset_gpu_memory():
    """Reset GPU memory and trigger garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
        torch._dynamo.reset()


def benchmark_experiment(exp_id, exp_name, run_fn, batch, num_warmup=15, num_bench=25):
    """Benchmark a single compilation setup with hardware CUDA timers."""
    reset_gpu_memory()

    # Step 1: Initial compilation step (measures JIT compile & autotune latency)
    t_start_compile = time.perf_counter()
    loss_init, _ = run_fn(batch)
    torch.cuda.synchronize()
    compilation_time_sec = time.perf_counter() - t_start_compile

    # Step 2: Warmup iterations to let Triton kernel search settle
    for _ in range(num_warmup):
        loss_val, _ = run_fn(batch)
    torch.cuda.synchronize()

    # Reset peak memory stats after warmup to measure steady-state VRAM
    torch.cuda.reset_peak_memory_stats()

    # Step 3: Measured benchmark iterations using CUDA Events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_bench):
        loss_val, _ = run_fn(batch)
    end_event.record()
    torch.cuda.synchronize()

    total_cuda_ms = start_event.elapsed_time(end_event)
    avg_step_ms = total_cuda_ms / num_bench
    iters_per_sec = 1000.0 / avg_step_ms
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "compilation_time_sec": round(compilation_time_sec, 3),
        "avg_step_ms": round(avg_step_ms, 2),
        "iters_per_sec": round(iters_per_sec, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "loss": float(loss_val.detach().cpu()),
    }


def main():
    print("=" * 80)
    print("      VGGT-MRI PYTORCH 2.13 COMPILATION BENCHMARK SUITE (E0..E20)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Error: CUDA GPU is required to run torch.compile benchmark suite.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device: {gpu_name} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB VRAM)")
    print(f"PyTorch Version: {torch.__version__}\n")

    torch.set_float32_matmul_precision("high")
    grid_shape = (12, 256, 256)
    batch_standard = build_synthetic_batch(device=device, S=10)

    # Base eager run function constructor (Forward + Loss)
    def build_eager_fwd_run(m):
        def run(b):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = m(b["images"], batch=b)
                loss_dict = compute_volume_intensity_loss(preds, b, grid_shape)
                loss = loss_dict["loss_volume"] + 0.1 * loss_dict["loss_pos_tv"]
            return loss, preds
        return run

    # Base eager run function constructor (Full Train Step: Forward + Backward + Step)
    def build_eager_train_run(m, opt):
        def run(b):
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = m(b["images"], batch=b)
                loss_dict = compute_volume_intensity_loss(preds, b, grid_shape)
                loss = loss_dict["loss_volume"] + 0.1 * loss_dict["loss_pos_tv"]
            loss.backward()
            opt.step()
            return loss, preds
        return run

    results = []

    # E0: Eager Baseline (Forward + Loss)
    print("Executing E0: Eager Baseline (Forward + Loss)...")
    m_e0 = get_fresh_model(device)
    res_e0 = benchmark_experiment("E0", "Eager Baseline (Fwd+Loss)", build_eager_fwd_run(m_e0), batch_standard)
    res_e0["rel_loss_err"] = 0.0
    res_e0["speedup"] = "1.00x"
    results.append(res_e0)
    eager_loss = res_e0["loss"]
    eager_fwd_ms = res_e0["avg_step_ms"]

    # Eager Baseline (Full Train Step)
    m_e0_train = get_fresh_model(device)
    opt_e0 = torch.optim.AdamW(m_e0_train.parameters(), lr=1e-4)
    res_e0_train = benchmark_experiment("E0_Train", "Eager Train Step Baseline", build_eager_train_run(m_e0_train, opt_e0), batch_standard)
    eager_train_ms = res_e0_train["avg_step_ms"]
    eager_train_loss = res_e0_train["loss"]

    def record_exp(res, baseline_ms=eager_fwd_ms, ref_loss=eager_loss):
        if res.get("status") == "FAILED":
            results.append(res)
            print(f"[{res['exp_id']}] {res['exp_name']}: FAILED / OOM | Reason: {res.get('error', 'Unknown Error')}")
            return
        rel_err = abs(res["loss"] - ref_loss) / (abs(ref_loss) + 1e-10)
        res["rel_loss_err"] = round(rel_err, 6)
        speedup = baseline_ms / res["avg_step_ms"]
        res["speedup"] = f"{speedup:.2f}x"
        res["status"] = "SUCCESS"
        results.append(res)
        print(f"[{res['exp_id']}] {res['exp_name']}: {res['avg_step_ms']} ms/step ({res['speedup']}) | Compile: {res['compilation_time_sec']}s | VRAM: {res['peak_vram_gb']} GB")

    record_exp(res_e0_train, baseline_ms=eager_train_ms, ref_loss=eager_train_loss)

    print(f"[E0] Eager Forward Latency: {eager_fwd_ms:.2f} ms/step | VRAM: {res_e0['peak_vram_gb']:.2f} GB")
    print(f"[E0_Train] Eager Full Train Step Latency: {eager_train_ms:.2f} ms/step\n")

    def safe_run_cell(exp_id, exp_name, setup_and_run_fn, num_warmup=15, baseline_ms=eager_fwd_ms, ref_loss=eager_loss):
        print(f"Executing {exp_id}: {exp_name}...")
        try:
            res = setup_and_run_fn(num_warmup)
            record_exp(res, baseline_ms=baseline_ms, ref_loss=ref_loss)
        except Exception as e:
            reset_gpu_memory()
            err_msg = str(e).split('\n')[0][:100]
            failed_res = {
                "exp_id": exp_id,
                "exp_name": exp_name,
                "status": "FAILED",
                "avg_step_ms": 0.0,
                "speedup": "N/A",
                "compilation_time_sec": 0.0,
                "peak_vram_gb": 0.0,
                "rel_loss_err": 0.0,
                "error": err_msg,
            }
            record_exp(failed_res, baseline_ms=baseline_ms, ref_loss=ref_loss)

    # E1: Whole Model Compile (default, dynamic=False for memory stability)
    def cell_e1(num_warmup):
        m = get_fresh_model(device)
        comp = torch.compile(m, mode="default", dynamic=False)
        return benchmark_experiment("E1", "Whole Model Default", build_eager_fwd_run(comp), batch_standard, num_warmup=num_warmup)
    safe_run_cell("E1", "Whole Model Default", cell_e1)

    # E2: Full Step Compile (default, dynamic=False)
    class FullStepE2(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, images, z_indices, t_indices, scanner_coords, gt_target_volume, anatomy_bbox, phases):
            b = {
                "images": images, "z_indices": z_indices, "t_indices": t_indices,
                "scanner_coords": scanner_coords, "gt_target_volume": gt_target_volume,
                "anatomy_bbox": anatomy_bbox, "phases": phases
            }
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = self.m(images, batch=b)
                loss_dict = compute_volume_intensity_loss(preds, b, (12, 256, 256))
                loss = loss_dict["loss_volume"] + 0.1 * loss_dict["loss_pos_tv"]
            return loss, preds["world_points"]

    def cell_e2(num_warmup):
        m = get_fresh_model(device)
        comp_full = torch.compile(FullStepE2(m), mode="default", dynamic=False)
        def run(b):
            loss, pos_pred = comp_full(b["images"], b["z_indices"], b["t_indices"], b["scanner_coords"], b["gt_target_volume"], b["anatomy_bbox"], b["phases"])
            return loss, {"world_points": pos_pred}
        return benchmark_experiment("E2", "Full Step Default", run, batch_standard, num_warmup=num_warmup)
    safe_run_cell("E2", "Full Step Default", cell_e2)

    # E3: 48 Transformer Blocks Compile
    def cell_e3(num_warmup):
        m = get_fresh_model(device)
        for i in range(len(m.aggregator.frame_blocks)):
            m.aggregator.frame_blocks[i] = torch.compile(m.aggregator.frame_blocks[i], mode="default")
            m.aggregator.global_blocks[i] = torch.compile(m.aggregator.global_blocks[i], mode="default")
        return benchmark_experiment("E3", "48 Transformer Blocks", build_eager_fwd_run(m), batch_standard, num_warmup=num_warmup)
    safe_run_cell("E3", "48 Transformer Blocks Default", cell_e3)

    # E4: DPT Point Head Compile
    def cell_e4(num_warmup):
        m = get_fresh_model(device)
        comp_head = torch.compile(m.point_head, mode="default")
        def run(b):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                tokens, p_idx = m.aggregator(b["images"], z_indices=b["z_indices"], t_indices=b["t_indices"])
                head_out, _ = comp_head(tokens, images=b["images"], patch_start_idx=p_idx)
                world_points = b["scanner_coords"] + head_out
                preds = {"world_points": world_points}
                loss_dict = compute_volume_intensity_loss(preds, b, grid_shape)
                loss = loss_dict["loss_volume"] + 0.1 * loss_dict["loss_pos_tv"]
            return loss, preds
        return benchmark_experiment("E4", "DPT Point Head Default", run, batch_standard, num_warmup=num_warmup)
    safe_run_cell("E4", "DPT Point Head Default", cell_e4)

    # E5: DINO Backbone Compile
    def cell_e5(num_warmup):
        m = get_fresh_model(device)
        m.aggregator.patch_embed = torch.compile(m.aggregator.patch_embed, mode="default")
        return benchmark_experiment("E5", "DINO Backbone Default", build_eager_fwd_run(m), batch_standard, num_warmup=num_warmup)
    safe_run_cell("E5", "DINO Backbone Default", cell_e5)

    # E6: Blocks + DPT Head Combined
    def cell_e6(num_warmup):
        m = get_fresh_model(device)
        for i in range(len(m.aggregator.frame_blocks)):
            m.aggregator.frame_blocks[i] = torch.compile(m.aggregator.frame_blocks[i], mode="default")
            m.aggregator.global_blocks[i] = torch.compile(m.aggregator.global_blocks[i], mode="default")
        m.point_head = torch.compile(m.point_head, mode="default")
        return benchmark_experiment("E6", "Blocks + Head Combined", build_eager_fwd_run(m), batch_standard, num_warmup=num_warmup)
    safe_run_cell("E6", "Blocks + Head Combined", cell_e6)

    # E7: Static S=10 CUDA Graphs
    def cell_e7(num_warmup):
        m = get_fresh_model(device)
        comp = torch.compile(m, mode="reduce-overhead", dynamic=False)
        return benchmark_experiment("E7", "Static S=10 CUDA Graphs", build_eager_fwd_run(comp), batch_standard, num_warmup=num_warmup)
    safe_run_cell("E7", "Static S=10 CUDA Graphs", cell_e7)

    # E8: Inductor Advanced Flags
    def cell_e8(num_warmup):
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = True
        m = get_fresh_model(device)
        comp = torch.compile(m, mode="default", dynamic=False)
        res = benchmark_experiment("E8", "Inductor Advanced Flags", build_eager_fwd_run(comp), batch_standard, num_warmup=num_warmup)
        torch._inductor.config.coordinate_descent_tuning = False
        torch._inductor.config.epilogue_fusion = False
        return res
    safe_run_cell("E8", "Inductor Advanced Flags", cell_e8)

    # E9: Optimizer Step Compile
    def cell_e9(num_warmup):
        m = get_fresh_model(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        opt_step_compiled = torch.compile(opt.step)
        def run(b):
            opt.zero_grad()
            loss, preds = build_eager_fwd_run(m)(b)
            loss.backward()
            opt_step_compiled()
            return loss, preds
        return benchmark_experiment("E9", "Optimizer Step Compile", run, batch_standard, num_warmup=num_warmup)
    safe_run_cell("E9", "Optimizer Step Compile", cell_e9, baseline_ms=eager_train_ms, ref_loss=eager_train_loss)

    # E10: Native Fused AdamW
    def cell_e10(num_warmup):
        m = get_fresh_model(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
        def run(b):
            opt.zero_grad()
            loss, preds = build_eager_fwd_run(m)(b)
            loss.backward()
            opt.step()
            return loss, preds
        return benchmark_experiment("E10", "Native Fused AdamW", run, batch_standard, num_warmup=num_warmup)
    safe_run_cell("E10", "Native Fused AdamW", cell_e10, baseline_ms=eager_train_ms, ref_loss=eager_train_loss)

    # E11: Whole Model Max-Autotune
    def cell_e11(num_warmup):
        m = get_fresh_model(device)
        comp = torch.compile(m, mode="max-autotune", dynamic=False)
        return benchmark_experiment("E11", "Whole Model Max-Autotune", build_eager_fwd_run(comp), batch_standard, num_warmup=num_warmup)
    safe_run_cell("E11", "Whole Model Max-Autotune", cell_e11, num_warmup=35)

    # E12: 48 Blocks Max-Autotune
    def cell_e12(num_warmup):
        m = get_fresh_model(device)
        for i in range(len(m.aggregator.frame_blocks)):
            m.aggregator.frame_blocks[i] = torch.compile(m.aggregator.frame_blocks[i], mode="max-autotune")
            m.aggregator.global_blocks[i] = torch.compile(m.aggregator.global_blocks[i], mode="max-autotune")
        return benchmark_experiment("E12", "48 Blocks Max-Autotune", build_eager_fwd_run(m), batch_standard, num_warmup=num_warmup)
    safe_run_cell("E12", "48 Blocks Max-Autotune", cell_e12, num_warmup=35)

    # E13: Full Step Max-Autotune
    def cell_e13(num_warmup):
        m = get_fresh_model(device)
        comp_full = torch.compile(FullStepE2(m), mode="max-autotune", dynamic=False)
        def run(b):
            loss, pos_pred = comp_full(b["images"], b["z_indices"], b["t_indices"], b["scanner_coords"], b["gt_target_volume"], b["anatomy_bbox"], b["phases"])
            return loss, {"world_points": pos_pred}
        return benchmark_experiment("E13", "Full Step Max-Autotune", run, batch_standard, num_warmup=num_warmup)
    safe_run_cell("E13", "Full Step Max-Autotune", cell_e13, num_warmup=35)

    # E14: Max-Autotune No CUDA Graphs
    def cell_e14(num_warmup):
        m = get_fresh_model(device)
        comp = torch.compile(m, mode="max-autotune-no-cudagraphs", dynamic=False)
        return benchmark_experiment("E14", "Max-Autotune No CUDA Graphs", build_eager_fwd_run(comp), batch_standard, num_warmup=num_warmup)
    safe_run_cell("E14", "Max-Autotune No CUDA Graphs", cell_e14, num_warmup=35)

    # E15: AOTAutograd Min-Cut Rematerialization
    def cell_e15(num_warmup):
        torch._functorch.config.activation_memory_budget = 0.5
        m = get_fresh_model(device)
        comp = torch.compile(m, mode="default", dynamic=False)
        res = benchmark_experiment("E15", "AOTAutograd Min-Cut Rematerialization", build_eager_fwd_run(comp), batch_standard, num_warmup=num_warmup)
        torch._functorch.config.activation_memory_budget = None
        return res
    safe_run_cell("E15", "AOTAutograd Min-Cut Rematerialization", cell_e15)

    # E16: CUTLASS + Triton GEMM Autotuning
    def cell_e16(num_warmup):
        torch._inductor.config.max_autotune_gemm_backends = "TRITON,CUTLASS,ATEN"
        m = get_fresh_model(device)
        comp = torch.compile(m, mode="max-autotune-no-cudagraphs", dynamic=False)
        res = benchmark_experiment("E16", "CUTLASS + Triton GEMM Autotuning", build_eager_fwd_run(comp), batch_standard, num_warmup=num_warmup)
        torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
        return res
    safe_run_cell("E16", "CUTLASS + Triton GEMM Autotuning", cell_e16, num_warmup=35)

    # E17: FX GEMM Fuser & Layout Opt
    def cell_e17(num_warmup):
        torch._inductor.config.fx_graph_linear_fuser = True
        torch._inductor.config.layout_optimization = True
        m = get_fresh_model(device)
        comp = torch.compile(m, mode="default", dynamic=False)
        res = benchmark_experiment("E17", "FX GEMM Fuser & Layout Opt", build_eager_fwd_run(comp), batch_standard, num_warmup=num_warmup)
        torch._inductor.config.fx_graph_linear_fuser = False
        torch._inductor.config.layout_optimization = False
        return res
    safe_run_cell("E17", "FX GEMM Fuser & Layout Opt", cell_e17)

    # E18: Static vs Dynamic Specialization
    def cell_e18(num_warmup):
        m = get_fresh_model(device)
        comp = torch.compile(m, mode="default", dynamic=False)
        return benchmark_experiment("E18", "Static Specialization (dynamic=False)", build_eager_fwd_run(comp), batch_standard, num_warmup=num_warmup)
    safe_run_cell("E18", "Static Specialization (dynamic=False)", cell_e18)

    # E19: TF32 vs IEEE FP32 Precision
    def cell_e19(num_warmup):
        torch.set_float32_matmul_precision("highest")
        m = get_fresh_model(device)
        comp = torch.compile(m, mode="default", dynamic=False)
        res = benchmark_experiment("E19", "IEEE FP32 Precision (highest)", build_eager_fwd_run(comp), batch_standard, num_warmup=num_warmup)
        torch.set_float32_matmul_precision("high")
        return res
    safe_run_cell("E19", "IEEE FP32 Precision (highest)", cell_e19)

    # E20: Combined Best-of-Breed Stack
    def cell_e20(num_warmup):
        m = get_fresh_model(device)
        for i in range(len(m.aggregator.frame_blocks)):
            m.aggregator.frame_blocks[i] = torch.compile(m.aggregator.frame_blocks[i], mode="max-autotune-no-cudagraphs")
            m.aggregator.global_blocks[i] = torch.compile(m.aggregator.global_blocks[i], mode="max-autotune-no-cudagraphs")
        m.point_head = torch.compile(m.point_head, mode="max-autotune-no-cudagraphs")
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
        def run(b):
            opt.zero_grad()
            loss, preds = build_eager_fwd_run(m)(b)
            loss.backward()
            opt.step()
            return loss, preds
        return benchmark_experiment("E20", "Combined Best-of-Breed Stack", run, batch_standard, num_warmup=num_warmup)
    safe_run_cell("E20", "Combined Best-of-Breed Stack", cell_e20, num_warmup=35, baseline_ms=eager_train_ms, ref_loss=eager_train_loss)

    # Save summary json
    os.makedirs("scratch", exist_ok=True)
    json_path = "scratch/compile_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw benchmark JSON results to {json_path}")

    # Generate Markdown Summary
    md_path = "scratch/compile_benchmark_report.md"
    with open(md_path, "w") as f:
        f.write("# PyTorch 2.13 torch.compile Benchmark Results\n\n")
        f.write(f"**GPU**: {gpu_name} | **PyTorch**: {torch.__version__}\n\n")
        f.write("| Exp ID | Experiment Name | Step Latency (ms) | Speedup | Compile Time (s) | Peak VRAM (GB) | Rel Loss Err |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            if r.get("status") == "FAILED":
                f.write(f"| {r['exp_id']} | {r['exp_name']} | FAILED | N/A | N/A | N/A | {r.get('error', 'OOM/Error')} |\n")
            else:
                f.write(f"| {r['exp_id']} | {r['exp_name']} | {r['avg_step_ms']} ms | {r['speedup']} | {r['compilation_time_sec']}s | {r['peak_vram_gb']} GB | {r['rel_loss_err']} |\n")
    print(f"Saved formatted markdown report to {md_path}")


if __name__ == "__main__":
    main()
