"""Epoch-matched comparison table for the 6 one-frame-ablation runs, from the cached wandb history.

Reads result/1frame_series/history.json (tools/pull_1frame_series.py). No network.

Why epoch-matched: the runs died at different epochs (25..39) and the hub's breathing metrics are
still improving at ep40, so a "final epoch" table structurally favours the longer runs. ep26 is the
last epoch where all six are alive.
"""
import json
import sys

H = json.load(open("/home/minsukc/vggt/result/1frame_series/history.json"))
ORDER = ["gather05", "no_gather", "contz", "dino_ft", "aug_moderate", "lowdiff100"]

METRICS = [
    ("val/resp/epe_dz_mm", "resp EPE mm", 3),
    ("val/resp/slope_dz", "resp slope", 3),
    ("val/resp/corr_dz", "resp corr", 3),
    ("val/resp/frac_deep_ignored", "deep ignored", 3),
    ("val/metric/recov_frac_heart", "recov", 3),
    ("val/metric/hole_frac_heart", "hole", 3),
    ("val/psnr/motion/mean", "psnr motion", 3),
    ("val/psnr/bbox_mean", "psnr bbox", 3),
    ("val/psnr/static", "psnr static", 3),
]


def at_epoch(variant, key, ep):
    """Value at the val point whose epoch == ep (step == ep*1000). None if absent."""
    for step, v in H[variant]["series"].get(key, []):
        if step // 1000 == ep:
            return v
    return None


def last(variant, key):
    s = H[variant]["series"].get(key, [])
    return (s[-1][1], s[-1][0] // 1000) if s else (None, None)


def fmt(v, nd=3):
    return "  -  " if v is None else f"{v:.{nd}f}"


def main():
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    print(f"\n=== EPOCH-MATCHED @ ep{ep} (last epoch where all 6 are alive) ===\n")
    print(f"{'metric':14s}" + "".join(f"{v[:11]:>12s}" for v in ORDER))
    for key, label, nd in METRICS:
        row = [at_epoch(v, key, ep) for v in ORDER]
        print(f"{label:14s}" + "".join(f"{fmt(x, nd):>12s}" for x in row))

    print(f"\n=== AS-IS (each run's final epoch — NOT comparable across columns) ===\n")
    print(f"{'metric':14s}" + "".join(f"{v[:11]:>12s}" for v in ORDER))
    eps = [last(v, "val/psnr/bbox_mean")[1] for v in ORDER]
    print(f"{'@epoch':14s}" + "".join(f"{e:>12d}" for e in eps))
    for key, label, nd in METRICS:
        row = [last(v, key)[0] for v in ORDER]
        print(f"{label:14s}" + "".join(f"{fmt(x, nd):>12s}" for x in row))

    print("\n=== PSNR identity baselines baked into each run's key (differ => different val draw) ===")
    for v in ORDER:
        print(f"  {v:13s} {H[v]['psnr_baselines']}")

    print("\n=== EF slope: every logged point (single-epoch reads are noise) ===")
    for v in ORDER:
        pts = H[v]["series"].get("val/ef/slope", [])
        sl = [x for _, x in pts]
        band = f"min={min(sl):.3f} max={max(sl):.3f}" if sl else ""
        print(f"  {v:13s} " + " ".join(f"ep{s//1000}:{x:.3f}" for s, x in pts) + f"   [{band}]")


if __name__ == "__main__":
    main()
