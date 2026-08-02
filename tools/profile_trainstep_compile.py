"""
Faithful one-train-step profiling harness for VGGT-MRI (mri_volume_diffusion, 1-frame).

Reproduces the real trainer's forward/loss/backward/optim sequence WITHOUT editing repo
code: it Hydra-composes the same config, `instantiate()`s the same model/loss/data, applies
the same aggft freeze, uses the same bf16 autocast, and (optionally) the same DDP wrap.
Only difference vs the real run: RANDOM init (skip the 8GB base-weights load — shapes are
checkpoint-independent, confirmed by extraction).

Modes (argv):
  eager        -> phase timing + component split + torch.profiler kernel breakdown
  compile      -> compile experiments (model-only & full-step; default & max-autotune)
Flags:
  --ddp        -> wrap model in DDP(find_unused_parameters=True) like the real run
  --s N        -> not used (S is set by the data/one_frame); informational
Run from repo root with:  PYTHONPATH=training:. python profile_harness.py <mode> [flags]
"""
import os, sys, time, json, contextlib, statistics
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True   # matches default.yaml allow_tf32:true
torch.backends.cudnn.allow_tf32 = True

REPO = "/home/minsukc/vggt"
SCRATCH = "/tmp/claude-114459240/-home-minsukc-vggt/38f5000c-c83b-4a11-8157-e666a5fcf314/scratchpad"
CONFIG = "default"      # was "mri_volume_diffusion", deleted in the 2026-08-01 config
                        # flattening; `default.yaml` IS that config (docs/62 §5.5)
OVERRIDES = [
    # `max_img_per_gpu=12` dropped: the key was deleted (docs/59 F9)
    "one_frame_per_slice=true",
    "loss.volume.gather_weight=0.5",
]

# ----------------------------------------------------------------------------- setup
def init_dist():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29610")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

def build(cfg_overrides=OVERRIDES):
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("rev_ts", lambda: "0", replace=True)
    OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p), replace=True)
    OmegaConf.register_new_resolver(
        "phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}", replace=True)
    with initialize_config_dir(version_base=None, config_dir=f"{REPO}/training/config"):
        cfg = compose(config_name=CONFIG, overrides=cfg_overrides)
    # Model (random init; skip checkpoint load) — instantiate exactly like trainer.py:345
    model = instantiate(cfg.model, _recursive_=False).cuda()
    # aggft freeze (trainer.py:355) — freeze_modules(["*patch_embed*"])
    from train_utils.freeze import freeze_modules
    if getattr(cfg.optim, "frozen_module_names", None):
        model = freeze_modules(model, patterns=cfg.optim.frozen_module_names)
    model.train()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[build] trainable {n_tr/1e6:.1f}M / {n_all/1e6:.1f}M params")
    # Loss (trainer.py:346)
    loss_fn = instantiate(cfg.loss, _recursive_=False)
    # Optimizer (trainer.py:158)
    from train_utils.optimizer import construct_optimizers
    optims = construct_optimizers(model, cfg.optim)
    # Data (trainer.py:377) -> real DynamicTorchDataset -> real DataLoader
    train_ds = instantiate(cfg.data.train, _recursive_=False)
    train_ds.seed = 42
    return cfg, model, loss_fn, optims, train_ds

def to_cuda(batch):
    out = {}
    for k, v in batch.items():
        out[k] = v.cuda(non_blocking=True) if torch.is_tensor(v) else v
    return out

def capture_batches(train_ds, n=3):
    """Iterate the REAL dataloader; time it; capture n batches to GPU. Lazy monai cache
    means only n subjects get resampled."""
    loader = train_ds.get_loader(epoch=0)
    batches, dl_times = [], []
    it = iter(loader)
    for i in range(n):
        t0 = time.perf_counter()
        b = next(it)
        dl_times.append(time.perf_counter() - t0)
        b = to_cuda(b)
        S = b["images"].shape[1] if b["images"].dim() == 5 else b["images"].shape[0]
        print(f"[data] batch {i}: images {tuple(b['images'].shape)}  S={S}  "
              f"gt_vol {tuple(b['gt_target_volume'].shape)}  dl={dl_times[-1]*1e3:.0f}ms")
        batches.append(b)
    return batches, dl_times

# ----------------------------------------------------------------------------- step
AMP = dict(device_type="cuda", dtype=torch.bfloat16, enabled=True)

def forward_loss(model, loss_fn, batch):
    with torch.autocast(**AMP):
        y_hat = model(images=batch["images"], batch=batch)
        loss_dict = loss_fn(y_hat, batch)
    return loss_dict["objective"], y_hat

def full_step(model, loss_fn, optims, batch):
    for o in optims:
        o.zero_grad(set_to_none=True)
    loss, _ = forward_loss(model, loss_fn, batch)
    loss.backward()
    for o in optims:
        o.optimizer.step()
    return loss

# --------------------------------------------------------------- CUDA-event timers
def cuda_time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts), max(ts)

def phase_timing(model, loss_fn, optims, batch, iters=20, warmup=6):
    """Fine-grained per-phase timing (eager only). Events around each phase in ONE step."""
    ev = {k: (torch.cuda.Event(True), torch.cuda.Event(True))
          for k in ["fwd", "loss", "bwd", "opt"]}
    def one():
        for o in optims: o.zero_grad(set_to_none=True)
        ev["fwd"][0].record()
        with torch.autocast(**AMP):
            y = model(images=batch["images"], batch=batch)
        ev["fwd"][1].record()
        ev["loss"][0].record()
        with torch.autocast(**AMP):
            ld = loss_fn(y, batch); loss = ld["objective"]
        ev["loss"][1].record()
        ev["bwd"][0].record(); loss.backward(); ev["bwd"][1].record()
        ev["opt"][0].record()
        for o in optims: o.optimizer.step()
        ev["opt"][1].record()
        return loss
    for _ in range(warmup): one()
    torch.cuda.synchronize()
    acc = {k: [] for k in ev}
    for _ in range(iters):
        one(); torch.cuda.synchronize()
        for k in ev: acc[k].append(ev[k][0].elapsed_time(ev[k][1]))
    return {k: statistics.median(v) for k, v in acc.items()}

def component_timing(model, loss_fn, batch, iters=20, warmup=6):
    """Split forward into aggregator vs head, and time the splat, faithfully."""
    from vggt.utils.splat import splat_predictions
    imgs = batch["images"]
    zi, ti, tti = batch.get("z_indices"), batch.get("t_indices"), batch.get("target_t_indices")
    grid = tuple(model.grid_shape) if hasattr(model, "grid_shape") else (12, 256, 256)
    res = {}
    def agg():
        with torch.autocast(**AMP):
            return model.aggregator(imgs, z_indices=zi, t_indices=ti, target_t_indices=tti)
    res["aggregator"] = cuda_time(lambda: agg(), iters, warmup)
    tokens, psi = agg()
    # Real VGGT.forward runs the head under autocast(enabled=False) (fp32) — vggt.py:98.
    def head():
        with torch.amp.autocast("cuda", enabled=False):
            return model.point_head(tokens, images=imgs, patch_start_idx=psi)
    res["point_head"] = cuda_time(lambda: head(), iters, warmup)
    # splat: build predictions like VGGT.forward (head fp32)
    with torch.amp.autocast("cuda", enabled=False):
        ho, hc = model.point_head(tokens, images=imgs, patch_start_idx=psi)
    dvf = ho
    sc = batch["scanner_coords"]
    wp = sc + dvf if getattr(model, "train_on_residual_dvf", True) else ho
    preds = {"world_points": wp, "world_points_conf": hc}
    res["splat"] = cuda_time(lambda: splat_predictions(preds, batch, grid), iters, warmup)
    return res

# ----------------------------------------------------------------------------- profiler
def profiler_trace(model, loss_fn, optims, batch, out_prefix):
    from torch.profiler import profile, ProfilerActivity, schedule
    for _ in range(6):
        full_step(model, loss_fn, optims, batch)
    torch.cuda.synchronize()
    sch = schedule(wait=1, warmup=2, active=4, repeat=1)
    rows = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=sch, record_shapes=False, with_stack=False) as prof:
        for _ in range(7):
            full_step(model, loss_fn, optims, batch)
            prof.step()
    torch.cuda.synchronize()
    ka = prof.key_averages()
    tbl = ka.table(sort_by="self_cuda_time_total", row_limit=35)
    with open(f"{out_prefix}_kernels.txt", "w") as f:
        f.write(tbl)
    print(tbl)
    return tbl

# ----------------------------------------------------------------------------- compile
def _explain_breaks(fn, *args):
    try:
        import torch._dynamo as dyn
        exp = dyn.explain(fn)(*args)
        # torch 2.3 returns an ExplainOutput with .graph_break_count / .break_reasons
        gb = getattr(exp, "graph_break_count", None)
        reasons = getattr(exp, "break_reasons", None)
        return gb, reasons
    except Exception as e:
        return f"explain-failed: {type(e).__name__}: {e}", None

def _clean_loss(fn):
    """Compute loss WITHOUT an optimizer step (same weights), no grad, for numeric match."""
    with torch.no_grad():
        return float(fn().detach())

def compile_experiments(cfg, model, loss_fn, optims, batch, use_ddp=False,
                        modes=("default", "max-autotune")):
    import torch._dynamo as dyn
    results = {}
    # Snapshot initial weights so every numeric-match is on IDENTICAL weights (the timing
    # loops call optimizer.step() and drift the weights; without restore, rel_err would
    # compare compiled-loss-on-drifted vs eager-loss-on-initial — invalid).
    m = model.module if hasattr(model, "module") else model
    snapshot = {k: v.detach().clone() for k, v in m.state_dict().items()}
    def restore():
        m.load_state_dict(snapshot)
    # ---- eager reference: numeric (no step, INITIAL weights) + timing ----
    ref_loss = _clean_loss(lambda: forward_loss(model, loss_fn, batch)[0])
    t_eager = cuda_time(lambda: full_step(model, loss_fn, optims, batch), 20, 6)
    results["eager"] = {"step_ms": t_eager, "loss": ref_loss}
    print(f"[eager] step {t_eager[0]:.1f}ms  loss(frozen) {ref_loss:.5f}")

    def rel(a, b):
        return abs(a - b) / (abs(b) + 1e-9)

    for mode in modes:
        # ===== model-only compile =====
        tag = f"model_{mode}"
        try:
            dyn.reset()
            cmodel = torch.compile(model, mode=mode, fullgraph=False)
            def fl_cm():
                with torch.autocast(**AMP):
                    y = cmodel(images=batch["images"], batch=batch)
                    return loss_fn(y, batch)["objective"]
            restore()                               # identical weights for a valid numeric match
            l = _clean_loss(fl_cm)                  # frozen-weight numeric match (also triggers compile)
            def step_cm():
                for o in optims: o.zero_grad(set_to_none=True)
                with torch.autocast(**AMP):
                    y = cmodel(images=batch["images"], batch=batch)
                    loss = loss_fn(y, batch)["objective"]
                loss.backward()
                for o in optims: o.optimizer.step()
                return loss
            t = cuda_time(step_cm, 20, 8)
            results[tag] = {"step_ms": t, "loss": l, "rel_err": rel(l, ref_loss),
                            "speedup": t_eager[0]/t[0]}
            print(f"[{tag}] step {t[0]:.1f}ms  x{t_eager[0]/t[0]:.2f}  loss {l:.5f} (rel {rel(l,ref_loss):.2e})")
        except Exception as e:
            results[tag] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
            print(f"[{tag}] ERROR {type(e).__name__}: {str(e)[:200]}")

        # ===== full-step compile (forward+loss, includes splat) =====
        tag = f"full_{mode}"
        try:
            dyn.reset()
            def fwd_loss(b):
                with torch.autocast(**AMP):
                    y = model(images=b["images"], batch=b)
                    ld = loss_fn(y, b)
                return ld["objective"]
            cfl = torch.compile(fwd_loss, mode=mode, fullgraph=False)
            restore()                               # identical weights for a valid numeric match
            l = _clean_loss(lambda: cfl(batch))     # frozen-weight numeric match
            def step_full():
                for o in optims: o.zero_grad(set_to_none=True)
                loss = cfl(batch)
                loss.backward()
                for o in optims: o.optimizer.step()
                return loss
            t = cuda_time(step_full, 20, 8)
            results[tag] = {"step_ms": t, "loss": l, "rel_err": rel(l, ref_loss),
                            "speedup": t_eager[0]/t[0]}
            print(f"[{tag}] step {t[0]:.1f}ms  x{t_eager[0]/t[0]:.2f}  loss {l:.5f} (rel {rel(l,ref_loss):.2e})")
        except Exception as e:
            results[tag] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
            print(f"[{tag}] ERROR {type(e).__name__}: {str(e)[:200]}")

    # ---- graph-break census on the forward (default mode) ----
    dyn.reset()
    gb, reasons = _explain_breaks(lambda: model(images=batch["images"], batch=batch))
    results["model_graph_breaks"] = {"count": gb, "reasons": str(reasons)[:2000]}
    print(f"[graph-breaks model.forward] count={gb}")
    return results

# ----------------------------------------------------------------------------- main
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "eager"
    use_ddp = "--ddp" in sys.argv
    os.chdir(REPO)
    init_dist()
    cfg, model, loss_fn, optims, train_ds = build()
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[0], find_unused_parameters=True,
                    gradient_as_bucket_view=True, bucket_cap_mb=25, broadcast_buffers=True)
        print("[ddp] wrapped")
    batches, dl_times = capture_batches(train_ds, n=3)
    b0 = batches[0]
    print(f"\n[data] dl median {statistics.median(dl_times)*1e3:.0f}ms "
          f"(first {dl_times[0]*1e3:.0f}ms incl. cache build)")

    tag = "ddp" if use_ddp else "nod"
    if mode == "eager":
        ph = phase_timing(model, loss_fn, optims, b0)
        tot = sum(ph.values())
        print("\n=== PHASE TIMING (median ms, batch0 S="
              f"{b0['images'].shape[1]}) ===")
        for k, v in ph.items():
            print(f"  {k:5s} {v:7.1f} ms  ({100*v/tot:4.1f}%)")
        print(f"  TOTAL {tot:7.1f} ms/step  ({1000/tot:.2f} it/s)")
        if not use_ddp:
            comp = component_timing(model, loss_fn, b0)
            print("\n=== COMPONENT SPLIT (median ms) ===")
            for k, v in comp.items():
                print(f"  {k:12s} {v[0]:7.1f} ms")
        print("\n=== PROFILER (batch0) ===")
        profiler_trace(model, loss_fn, optims, b0, f"{SCRATCH}/prof_{tag}")
        # per-batch step time across the 3 captured S values
        print("\n=== STEP TIME per captured batch (S varies with one_frame) ===")
        for i, b in enumerate(batches):
            t = cuda_time(lambda: full_step(model, loss_fn, optims, b), 15, 5)
            print(f"  batch{i} S={b['images'].shape[1]:2d}: {t[0]:7.1f} ms/step")
        json.dump({"phase": ph, "dl_ms": [x*1e3 for x in dl_times]},
                  open(f"{SCRATCH}/eager_{tag}.json", "w"), indent=2)
    elif mode == "proper":
        # Test a PROPERLY set-up compile: remove manual gradient checkpointing and let
        # AOTAutograd's min-cut partitioner manage recompute. Measure peak mem + step time.
        import vggt.models.aggregator as agg
        import torch._dynamo as dyn
        orig_ckpt = agg.checkpoint
        S = b0["images"].shape[1]
        print(f"\n=== PROPER-COMPILE experiment (S={S}) ===")
        def measure(fn, name, warmup=6, iters=15):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                t = cuda_time(fn, iters, warmup)
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  [{name}] step {t[0]:7.1f} ms   peak {peak:5.1f} GB")
                return {"step_ms": t[0], "peak_gb": peak}
            except RuntimeError as e:
                oom = "out of memory" in str(e).lower()
                print(f"  [{name}] {'OOM' if oom else 'ERR'}: {str(e)[:160]}")
                torch.cuda.empty_cache()
                return {"error": "OOM" if oom else str(e)[:300]}
        res = {}
        # 1) eager WITH manual checkpointing (the real baseline)
        agg.checkpoint = orig_ckpt
        res["eager_ckpt"] = measure(lambda: full_step(model, loss_fn, optims, b0), "eager +manual-ckpt")
        # 2) COMPILED without manual ckpt -> AOTAutograd min-cut manages recompute+memory.
        #    Run FIRST (on clean memory) to rule out fragmentation from a prior OOM.
        def _passthrough(fn, *a, **k):
            k.pop('use_reentrant', None); k.pop('preserve_rng_state', None)
            return fn(*a, **k)
        agg.checkpoint = _passthrough
        dyn.reset()
        torch.cuda.empty_cache()
        cmodel = torch.compile(model, mode="default", fullgraph=False)
        def step_c():
            for o in optims: o.zero_grad(set_to_none=True)
            with torch.autocast(**AMP):
                y = cmodel(images=b0["images"], batch=b0)
                loss = loss_fn(y, b0)["objective"]
            loss.backward()
            for o in optims: o.optimizer.step()
            return loss
        res["compiled_nockpt_mincut"] = measure(step_c, "compiled no-ckpt (min-cut)", warmup=12)
        # 3) eager WITHOUT checkpointing (store-everything; likely OOM at S>=10)
        agg.checkpoint = _passthrough
        torch.cuda.empty_cache()
        res["eager_nockpt"] = measure(lambda: full_step(model, loss_fn, optims, b0), "eager  no-ckpt")
        # 4) COMPILED with manual ckpt kept (what my earlier run tested) for contrast
        agg.checkpoint = orig_ckpt
        dyn.reset()
        cmodel2 = torch.compile(model, mode="default", fullgraph=False)
        def step_c2():
            for o in optims: o.zero_grad(set_to_none=True)
            with torch.autocast(**AMP):
                y = cmodel2(images=b0["images"], batch=b0)
                loss = loss_fn(y, b0)["objective"]
            loss.backward()
            for o in optims: o.optimizer.step()
            return loss
        res["compiled_with_ckpt"] = measure(step_c2, "compiled +manual-ckpt", warmup=12)
        agg.checkpoint = orig_ckpt
        json.dump(res, open(f"{SCRATCH}/proper_{tag}.json", "w"), indent=2, default=str)
        print(f"\nsaved -> {SCRATCH}/proper_{tag}.json")
    elif mode == "parts":
        # Test PARTIAL compilation: compile the frozen DINO backbone (forward-only, no ckpt)
        # and/or compile each attention Block while keeping manual checkpoint AROUND it.
        import torch._dynamo as dyn
        from vggt.utils.splat import splat_predictions
        agg_mod = model.aggregator
        S = b0["images"].shape[1]
        print(f"\n=== PARTIAL-COMPILE experiment (S={S}) ===")
        def measure(fn, name, warmup=8, iters=15):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                t = cuda_time(fn, iters, warmup)
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  [{name:34s}] step {t[0]:7.1f} ms   peak {peak:5.1f} GB")
                return {"step_ms": t[0], "peak_gb": peak}
            except RuntimeError as e:
                oom = "out of memory" in str(e).lower()
                print(f"  [{name:34s}] {'OOM' if oom else 'ERR'}: {str(e)[:150]}")
                torch.cuda.empty_cache(); return {"error": "OOM" if oom else str(e)[:300]}
        # sub-component forward timing: how big is DINO vs the 48 blocks in the forward?
        imgs = b0["images"].view(1 * S, 3, 518, 518) if b0["images"].dim() == 5 else b0["images"]
        def pe_fwd():
            with torch.autocast(**AMP):
                return agg_mod.patch_embed(imgs)
        print("  --- forward-only sub-times ---")
        tpe = cuda_time(pe_fwd, 15, 6)
        print(f"  [DINO patch_embed fwd] {tpe[0]:.1f} ms")
        res = {"dino_patchembed_fwd_ms": tpe[0]}
        # baseline
        res["baseline"] = measure(lambda: full_step(model, loss_fn, optims, b0), "eager baseline")
        # A) compile ONLY the frozen DINO backbone
        dyn.reset()
        orig_pe = agg_mod.patch_embed
        agg_mod.patch_embed = torch.compile(orig_pe, mode="default")
        res["compile_dino"] = measure(lambda: full_step(model, loss_fn, optims, b0), "compile DINO only")
        agg_mod.patch_embed = orig_pe
        # B) compile each attention Block (checkpoint kept AROUND them -> memory preserved)
        dyn.reset()
        orig_fb = list(agg_mod.frame_blocks); orig_gb = list(agg_mod.global_blocks)
        for i in range(len(agg_mod.frame_blocks)):
            agg_mod.frame_blocks[i] = torch.compile(orig_fb[i], mode="default")
        for i in range(len(agg_mod.global_blocks)):
            agg_mod.global_blocks[i] = torch.compile(orig_gb[i], mode="default")
        res["compile_blocks"] = measure(lambda: full_step(model, loss_fn, optims, b0),
                                        "compile 48 blocks (+ckpt)", warmup=12)
        # C) both DINO + blocks
        dyn.reset()
        agg_mod.patch_embed = torch.compile(orig_pe, mode="default")
        res["compile_dino_and_blocks"] = measure(lambda: full_step(model, loss_fn, optims, b0),
                                                 "compile DINO + 48 blocks", warmup=12)
        # restore
        agg_mod.patch_embed = orig_pe
        for i in range(len(agg_mod.frame_blocks)): agg_mod.frame_blocks[i] = orig_fb[i]
        for i in range(len(agg_mod.global_blocks)): agg_mod.global_blocks[i] = orig_gb[i]
        json.dump(res, open(f"{SCRATCH}/parts_{tag}.json", "w"), indent=2, default=str)
        print(f"\nsaved -> {SCRATCH}/parts_{tag}.json")
    elif mode == "compile":
        # second positional arg selects modes: "default" | "maxautotune" | "both"
        which = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "default"
        modes = {"default": ["default"], "maxautotune": ["max-autotune"],
                 "both": ["default", "max-autotune"]}[which]
        res = compile_experiments(cfg, model, loss_fn, optims, b0, use_ddp, modes=modes)
        json.dump(res, open(f"{SCRATCH}/compile_{tag}_{which}.json", "w"),
                  indent=2, default=str)
        print(f"\nsaved -> {SCRATCH}/compile_{tag}_{which}.json")
    print("\n[done]")

if __name__ == "__main__":
    main()
