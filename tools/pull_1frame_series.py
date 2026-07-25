"""Pull the full per-epoch wandb history for the 6 one-frame-ablation runs -> one cached JSON.

Cached so every downstream figure/report build is offline and reproducible.

Two gotchas this encodes (both verified, both silently corrupt the analysis otherwise):
  * run.history(keys=[...]) inner-joins: a row survives only if EVERY requested key is non-null.
    EF is logged every 5 val epochs, val metrics every epoch -> mixing them collapses 40 rows to 8.
    So: scan_history() with no keys=, and filter per-metric afterwards.
  * The identity baseline is baked into the PSNR key name (`.../mean_n60_base17.1`), and contz
    logs base17.3 because continuous_z changes which planes are drawn. Matching the full string
    silently DROPS contz. So: resolve those keys by regex.
"""
import json
import os
import re
import wandb

ENTITY = "minsuk-choi/vggt-mri"
OUT = "/home/minsukc/vggt/result/1frame_series"

RUNS = {  # variant -> (wandb id, ckpt epoch)
    "gather05":     ("fhkgalju", 39),
    "no_gather":    ("lmboejhq", 38),
    "contz":        ("tfz1x7ft", 39),
    "dino_ft":      ("hlh3emae", 34),
    "aug_moderate": ("lylgvajs", 39),
    "lowdiff100":   ("2kwj0tkd", 25),
}

# Exact keys, identical across all 6 runs.
EXACT = [
    "val/resp/slope_dz", "val/resp/corr_dz", "val/resp/epe_dz_mm",
    "val/resp/frac_deep_ignored", "val/resp/disp_mm_mean", "val/resp/disp_mm_max",
    "val/metric/recov_frac_heart", "val/metric/hole_frac_heart", "val/metric/coverage_frac",
    "val/metric/mse_heart_identity", "val/metric/mse_heart_model", "val/metric/mse_heart_oracle",
    "val/metric/ssim_3d_full",
    "val/psnr/bbox_mean", "val/psnr/static", "val/psnr/heartseg",
    "val/ef/slope", "val/ef/spearman", "val/ef/mae_pct", "val/ef/n",
    "train/optim/lr", "train/optim/epoch",
]
# Regex keys: baseline is baked into the name and DIFFERS across runs (contz!).
REGEX = {
    "val/psnr/motion/mean": re.compile(r"^val/psnr/motion/mean_n\d+_base[\d.]+$"),
    "val/psnr/bbox/mean":   re.compile(r"^val/psnr/bbox/mean_n\d+_base[\d.]+$"),
    "val/psnr/full/mean":   re.compile(r"^val/psnr/full/mean_n\d+_base[\d.]+$"),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    api = wandb.Api(timeout=60)
    out = {}
    for variant, (rid, ckpt_epoch) in RUNS.items():
        run = api.run(f"{ENTITY}/{rid}")
        # Resolve the regex keys (and record the baseline each run actually used).
        resolved, bases = {}, {}
        for canon, pat in REGEX.items():
            hits = [k for k in run.summary.keys() if pat.match(k)]
            if hits:
                resolved[canon] = hits[0]
                bases[canon] = float(hits[0].rsplit("base", 1)[1])
        wanted = set(EXACT) | set(resolved.values())

        # scan_history flakes with HTTP 500 on large/actively-logging runs; retry a few times.
        series = {}
        for attempt in range(4):
            try:
                series = {}
                for row in run.scan_history():  # no keys= -> no inner join
                    step = row.get("_step")
                    if step is None:
                        continue
                    for k in wanted:
                        v = row.get(k)
                        if v is None:
                            continue
                        canon = next((c for c, r in resolved.items() if r == k), k)
                        series.setdefault(canon, []).append([step, v])
                break
            except Exception as e:
                print(f"  {variant}: scan_history attempt {attempt+1} failed ({e}); retrying", flush=True)
                if attempt == 3:
                    raise
        for k in series:  # dedupe steps, keep last, sort
            series[k] = sorted({s: v for s, v in series[k]}.items())

        out[variant] = {"wandb_id": rid, "ckpt_epoch": ckpt_epoch, "state": run.state,
                        "tags": run.tags, "psnr_baselines": bases,
                        "resolved_keys": resolved, "series": series}
        print(f"{variant:13s} {rid}  metrics={len(series):2d}  "
              f"motion_base={bases.get('val/psnr/motion/mean')}  ep_max={max(s for s, _ in series['val/psnr/bbox_mean'])//1000}",
              flush=True)

    path = os.path.join(OUT, "history.json")
    json.dump(out, open(path, "w"))
    print(f"\n-> {path} ({os.path.getsize(path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
