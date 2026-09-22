"""Fit one CiNeVol model to one subject's observed pixels."""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from .config import configuration, write_json
from .data import Observations, chunks
from .model import CiNeVol
from .psf import training_samples
from .losses import loss_terms, weighted_loss


def load_checkpoint(path, device="cpu"):
    # Checkpoints include configuration and RNG state. Load only your own files.
    return torch.load(path, map_location=device, weights_only=False)


def model_from_checkpoint(path, device):
    state = load_checkpoint(path)
    if state["config"]["backend"] == "grid4d" and str(device) == "cpu":
        raise ValueError("This checkpoint uses the Grid4D CUDA encoder. Reconstruct on a GPU; exported NIfTI files are viewable on CPU.")
    model = CiNeVol(state["config"], state["bounds"], state["n_slices"], state["n_frames"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), state


def batch_gradient(model, batch, noise, microbatch):
    count = len(batch["value"])
    global_abs_bias = torch.zeros((), device=batch["xyz"].device)
    # Same sampled pixels and PSF noise in both passes; parameters are unchanged.
    with torch.no_grad():
        for start, part in chunks(batch, microbatch):
            n = len(part["value"])
            xyz = training_samples(part, noise[start:start + n])
            expand = lambda x: x[:, None].expand(n, noise.shape[1])
            out = model(xyz, expand(part["cardiac"]), expand(part["respiratory"]),
                        expand(part["slice_index"]), expand(part["frame_index"]))
            global_abs_bias += out["bias"].abs().mean() * (n / count)
    logged = {}
    for start, part in chunks(batch, microbatch):
        n = len(part["value"])
        xyz = training_samples(part, noise[start:start + n])
        terms = loss_terms(model, part, xyz, global_abs_bias)
        objective = weighted_loss(terms, model.config) * (n / count)
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite objective; stopping instead of discarding invalid values")
        objective.backward()
        for k, value in terms.items():
            logged[k] = logged.get(k, 0.) + float(value.detach()) * (n / count)
    logged["total"] = sum(model.config["loss_weights"][k] * v for k, v in logged.items())
    return logged


def fit(manifest, output, profile="invivo", backend="grid4d", device="cuda", microbatch=1024,
        seed=7, smoke=False, resume=False, checkpoint_every=25):
    if microbatch < 1 or checkpoint_every < 1:
        raise ValueError("Microbatch and checkpoint interval must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run inside srun on a GPU allocation, or use --device cpu --backend torch --smoke.")
    if backend == "grid4d" and not device.startswith("cuda"):
        raise ValueError("Grid4D requires CUDA")
    t_start = time.perf_counter()
    dataset = Observations(manifest)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    last = output / "last.pt"
    # Avoid concurrent notebooks/jobs writing the same run.
    import fcntl
    lock = open(output / ".fit.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"Another process is fitting {output}")
    try:
        if last.exists() and not resume:
            raise FileExistsError(f"{last} already exists. Use --resume or a new --output directory.")
        if resume and not last.exists():
            raise FileNotFoundError(f"Cannot resume: {last} does not exist")
        if not resume and (output / "losses.jsonl").exists():
            raise FileExistsError("Run has logs but no checkpoint. Use a fresh output directory.")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        config = configuration(profile, smoke)
        config["backend"] = backend
        bounds = dataset.bounds(config["implementation_choices"]["coordinate_padding_mm"])
        model = CiNeVol(config, bounds, dataset.n_slices, dataset.n_frames).to(device)
        opt_config = config["optimization"]
        choices = config["implementation_choices"]
        optimizer = torch.optim.AdamW(model.optimizer_groups(), lr=opt_config["learning_rate"],
                                      betas=tuple(choices["adam_betas"]), eps=choices["adam_eps"])
        start, previous_seconds = 0, 0.
        if resume:
            state = load_checkpoint(last)
            if state["fingerprint"] != dataset.fingerprint or state["config"] != config:
                raise ValueError("Resume manifest/data/configuration differs from checkpoint")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            rng.set_state(state["sampling_rng"])
            torch.set_rng_state(state["torch_rng"])
            if device.startswith("cuda") and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            start, previous_seconds = state["step"], state["optimization_seconds"]
            if start >= opt_config["steps_per_subject"]:
                print(f"Run already complete at step {start}: {last}", flush=True)
                return last
            # Discard only records newer than the last atomically saved checkpoint.
            valid = []
            for line in (output / "losses.jsonl").read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["step"] <= start:
                    valid.append(line)
            (output / "losses.jsonl").write_text("\n".join(valid) + ("\n" if valid else ""))
        write_json(output / "config.json", config)
        write_json(output / "subject.json", {**dataset.meta, "source_manifest": str(dataset.path)})
        write_json(output / "environment.json", {
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
            "seed": seed, "microbatch": microbatch, "fingerprint": dataset.fingerprint,
            "NeSVoR_reviewed_commit": "730ddaa3711a2304386de34193ea4b957892fe7b",
            "Grid4D_commit": "a8992a9bd18b1828d2890a190421bca8bf0d2e80",
        })
        setup_seconds = time.perf_counter() - t_start
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t_fit = time.perf_counter()
        print(f"{'SMOKE ONLY' if smoke else 'PAPER CONFIGURATION'}: {opt_config['steps_per_subject']} steps; "
              f"{opt_config['batch_observed_pixels']} pixels/step; {config['psf_samples']['fitting']} PSF samples; {dataset.n:,} observations", flush=True)
        model.train()
        with open(output / "losses.jsonl", "a", buffering=1) as log:
            for step in range(start + 1, opt_config["steps_per_subject"] + 1):
                batch = dataset.sample(opt_config["batch_observed_pixels"], rng, device)
                noise = torch.randn(len(batch["value"]), config["psf_samples"]["fitting"], 3,
                                    generator=rng).to(device)
                optimizer.zero_grad(set_to_none=True)
                terms = batch_gradient(model, batch, noise, microbatch)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Non-finite model gradient")
                optimizer.step()
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = previous_seconds + time.perf_counter() - t_fit
                row = dict(step=step, optimization_seconds=elapsed, **terms)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if step == 1 or step % 10 == 0 or step == opt_config["steps_per_subject"]:
                    print(json.dumps(row), flush=True)
                if step % checkpoint_every == 0 or step == opt_config["steps_per_subject"]:
                    state = {"schema_version": 1, "step": step, "config": config,
                             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "bounds": bounds.tolist(), "n_slices": dataset.n_slices, "n_frames": dataset.n_frames,
                             "manifest": str(dataset.path), "subject_metadata": dataset.meta,
                             "fingerprint": dataset.fingerprint, "sampling_rng": rng.get_state(),
                             "torch_rng": torch.get_rng_state(),
                             "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
                             "optimization_seconds": elapsed}
                    torch.save(state, output / "last.pt.tmp")
                    os.replace(output / "last.pt.tmp", last)
        write_json(output / "timing_fit.json", {"setup_seconds_this_invocation": setup_seconds,
                   "optimization_seconds": previous_seconds + time.perf_counter() - t_fit,
                   "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
                   "complete": True, "steps": opt_config["steps_per_subject"], "smoke": smoke})
        return last
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--profile", choices=["phantom", "invivo"], default="invivo")
    p.add_argument("--backend", choices=["torch", "grid4d"], default="grid4d")
    p.add_argument("--device", default="cuda")
    p.add_argument("--microbatch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--smoke", action="store_true", help="Reduced architecture and budget for software verification ONLY")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    fit(**vars(a))


if __name__ == "__main__":
    main()
