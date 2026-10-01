#!/usr/bin/env python3
"""Benchmark VGGT-Ω selective aggregator-output caching on one GPU.

Uses the full production architecture with identical random weights while toggling the
aggregator cache between all layers and ``(4, 11, 17, 23)``. Reports exact output equality,
CUDA peak allocated memory, and median CUDA-event time. The optional training measurement
runs model forward + backward through a simple squared-world-point loss; it intentionally
does not include the splat, volume loss, optimizer state, or an optimizer step.

Run from the repository root:

    PYTHONPATH=training:. python tools/benchmark_aggregator_cache.py --s 10 --train
"""

import argparse
import gc
import json
import statistics

import torch

from vggt.models.vggt import VGGT


SELECTIVE = (4, 11, 17, 23)


def clean_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def measure_memory_once(fn):
    clean_cuda()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    output = fn()
    torch.cuda.synchronize()
    return output, {
        "baseline_gib": baseline / 2**30,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "incremental_peak_gib": (torch.cuda.max_memory_allocated() - baseline) / 2**30,
        "live_gib": torch.cuda.memory_allocated() / 2**30,
    }


def time_once(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = fn()
    end.record()
    torch.cuda.synchronize()
    return output, start.elapsed_time(end)


def summarize_times(times):
    return {
        "ms": statistics.median(times),
        "ms_min": min(times),
        "ms_max": max(times),
    }


def measure_scope(fn, cache_target, caches, repeats, before_each=None):
    memory = {}
    for label, cache in caches:
        if before_each is not None:
            before_each()
        cache_target.cached_layer_indices = set(cache)
        output, memory[label] = measure_memory_once(fn)
        del output

    for _, cache in caches:
        if before_each is not None:
            before_each()
        cache_target.cached_layer_indices = set(cache)
        warmup = fn()
        del warmup
    torch.cuda.synchronize()

    times = {label: [] for label, _ in caches}
    for repeat in range(repeats):
        order = caches if repeat % 2 == 0 else tuple(reversed(caches))
        for label, cache in order:
            if before_each is not None:
                before_each()
            cache_target.cached_layer_indices = set(cache)
            output, elapsed = time_once(fn)
            times[label].append(elapsed)
            del output

    return {
        label: {**memory[label], **summarize_times(times[label])}
        for label, _ in caches
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.repeats < 4 or args.repeats % 2:
        parser.error("--repeats must be an even integer >= 4 for balanced timing order")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    model = VGGT(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_point=True,
        use_z_pose_embedding=True,
        use_reference_token=True,
        train_on_residual_dvf=True,
        gradient_checkpointing=True,
        warp_head_type="dpt",
    ).to(device)
    for parameter in model.aggregator.patch_embed.parameters():
        parameter.requires_grad_(False)

    images = torch.rand(1, args.s, 3, 518, 518, device=device)
    z_indices = torch.linspace(-0.8, 0.8, args.s, device=device).view(1, args.s, 1)
    scanner_coords = torch.zeros(1, args.s, 518, 518, 3, device=device)
    batch = {"z_indices": z_indices, "scanner_coords": scanner_coords}
    all_layers = tuple(range(model.aggregator.depth))
    caches = (("all", all_layers), ("selective", SELECTIVE))

    def aggregator_forward():
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return model.aggregator(images, z_indices=z_indices)

    def model_forward():
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return model(images, batch=batch)

    model.eval()
    model.aggregator.cached_layer_indices = set(SELECTIVE)
    warmup = aggregator_forward()
    del warmup

    model.aggregator.cached_layer_indices = set(all_layers)
    all_outputs, _ = aggregator_forward()
    model.aggregator.cached_layer_indices = set(SELECTIVE)
    selective_outputs, _ = aggregator_forward()
    retained_dtype = str(selective_outputs[SELECTIVE[0]].dtype)
    retained_element_size = selective_outputs[SELECTIVE[0]].element_size()
    aggregator_equal = {
        str(index): torch.equal(all_outputs[index], selective_outputs[index]) for index in SELECTIVE
    }
    del all_outputs, selective_outputs
    clean_cuda()

    aggregate_results = measure_scope(
        aggregator_forward, model.aggregator, caches, args.repeats
    )

    model.aggregator.cached_layer_indices = set(all_layers)
    pred_all = model_forward()
    model.aggregator.cached_layer_indices = set(SELECTIVE)
    pred_selective = model_forward()
    prediction_equal = {
        key: torch.equal(pred_all[key], pred_selective[key])
        for key in ("world_points", "world_points_conf", "dvfs")
    }
    prediction_max_abs_diff = {
        key: (pred_all[key] - pred_selective[key]).abs().max().item() for key in prediction_equal
    }
    del pred_all, pred_selective
    clean_cuda()

    model_results = measure_scope(model_forward, model.aggregator, caches, args.repeats)

    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "S": args.s,
        "dtype": "bf16 aggregator autocast; fp32 DPT as implemented",
        "retained_output_dtype": retained_dtype,
        "retained_output_element_size_bytes": retained_element_size,
        "weights": "random initialization; identical model toggled in-place",
        "aggregator_retained_equal": aggregator_equal,
        "model_prediction_equal": prediction_equal,
        "model_prediction_max_abs_diff": prediction_max_abs_diff,
        "aggregator": aggregate_results,
        "full_model_inference": model_results,
    }

    if args.train:
        model.train()

        def train_forward_backward():
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(images, batch=batch)
                loss = prediction["world_points"].square().mean()
            loss.backward()
            return loss

        train_results = measure_scope(
            train_forward_backward,
            model.aggregator,
            caches,
            args.repeats,
            before_each=lambda: model.zero_grad(set_to_none=True),
        )
        for label in train_results:
            model.zero_grad(set_to_none=True)
            model.aggregator.cached_layer_indices = set(dict(caches)[label])
            loss = train_forward_backward()
            train_results[label]["loss"] = loss.detach().item()
            del loss
        result["train_forward_backward"] = train_results

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
