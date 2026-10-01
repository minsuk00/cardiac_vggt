#!/usr/bin/env python3
"""
tools/profile_train_step.py

Profiles one full VGGT-MRI train step (fwd + bwd + optimizer) using torch.profiler.
Reports:
  1. Top-20 GPU kernels by self-CUDA time
  2. Time fraction: attention blocks vs splat vs loss vs backward vs optimizer
  3. Which SDPA backend is selected (Flash vs MemEfficient vs Math)
  4. Saves Chrome trace to scratch/profile_train_step.json (open in chrome://tracing)

Usage:
  PYTHONPATH=training:. python tools/profile_train_step.py
"""

import os
import sys
import json

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity, schedule

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("training"))

torch.set_float32_matmul_precision("high")
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
    # Compile the 48 blocks (production setting)
    for i in range(len(m.aggregator.frame_blocks)):
        m.aggregator.frame_blocks[i]  = torch.compile(m.aggregator.frame_blocks[i],  mode="default", dynamic=True)
        m.aggregator.global_blocks[i] = torch.compile(m.aggregator.global_blocks[i], mode="default", dynamic=True)
    return m


def check_sdp_backends():
    """Report which SDPA backends are enabled."""
    print("\n--- SDPA Backend Status ---")
    print(f"  Flash SDP enabled:          {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"  Mem-efficient SDP enabled:  {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  Math SDP enabled:           {torch.backends.cuda.math_sdp_enabled()}")
    # Run a small test to see which backend is actually selected
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        try:
            q = torch.randn(1, 16, 100, 64, device="cuda", dtype=torch.bfloat16)
            _ = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            print("  Flash Attention (FA2):      AVAILABLE and selected for BF16")
        except Exception as e:
            print(f"  Flash Attention (FA2):      NOT available — {e}")
    print()


def main():
    device = "cuda"
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu} | PyTorch: {torch.__version__}")

    check_sdp_backends()

    b   = build_batch(device, S=S)
    m   = make_model(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)

    from training.loss import compute_volume_intensity_loss

    def train_step(batch):
        opt.zero_grad()
        with record_function("forward"):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with record_function("aggregator"):
                    preds = m(batch["images"], batch=batch)
                with record_function("splat+loss"):
                    d = compute_volume_intensity_loss(preds, batch, (12, 256, 256))
                    loss = d["loss_volume"] + 0.1 * d["loss_pos_tv"]
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            opt.step()
        return loss

    # Warmup (including compile)
    print("Warming up (3 steps, includes compile)...")
    for _ in range(3):
        train_step(b)
    torch.cuda.synchronize()
    print("Warmup done.\n")

    # Profile
    os.makedirs("scratch", exist_ok=True)
    trace_path = "scratch/profile_train_step.json"

    print("Profiling 3 steps...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    ) as prof:
        for _ in range(5):  # wait=1, warmup=1, active=3
            train_step(b)
            prof.step()

    torch.cuda.synchronize()

    # Save Chrome trace
    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace saved: {trace_path}")
    print("  Open in chrome://tracing or https://ui.perfetto.dev\n")

    # Print top-20 GPU kernels by self-CUDA time
    print("=== Top 20 GPU ops by self-CUDA time ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=20
    ))

    # Print top-10 CPU ops
    print("\n=== Top 10 CPU ops by self-CPU time ===")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=10
    ))

    # Summary: custom sections
    print("\n=== Custom section times (avg over 3 steps) ===")
    for key in ["forward", "aggregator", "splat+loss", "backward", "optimizer"]:
        events = [e for e in prof.key_averages() if e.key == key]
        if events:
            e = events[0]
            print(f"  {key:15s}: CPU={e.self_cpu_time_total/3/1000:.1f} ms  "
                  f"CUDA={e.self_cuda_time_total/3/1000:.1f} ms  "
                  f"CUDA-total={e.cuda_time_total/3/1000:.1f} ms")


if __name__ == "__main__":
    main()
