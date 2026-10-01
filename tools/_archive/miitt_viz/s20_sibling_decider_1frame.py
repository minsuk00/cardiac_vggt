"""Follow-up to s20_sibling_decider.py: feed ALL THREE checkpoints the IDENTICAL '1frame'
plane-following batch (build_1frame_batch: 1 frame/plane, snapped z, slot 0 = swept reference,
NO companion pile-up on the reference plane). This removes the multiframe frozen-reference EVAL
artifact so ref-conditioning is clean for all three -> the ONLY variable is the weights.

Adds the beat-correlation probe to ALL three so we can tell REAL motion propagation
(corr(recon, GT_beat) high) from spurious aliveness (corr ~ 0).

Run: micromamba run -n svr env PYTHONPATH=training:. python -u tools/miitt_viz/s20_sibling_decider_1frame.py
Out: result/s20_decider/summary_1frame.json
"""
import os, sys, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "training")
import tools.miitt_viz.s20_sibling_decider as D   # reuse load_bundles/eval_model/alive/_probe/_PROBE

# force ALL models into the identical 1frame regime + probe everything
D.CKPTS = {
    "gather05": ("216539845_*ftgather05*1frame*", "1frame"),
    "s20":      ("216949759_*s20_dynamic*",       "1frame"),
    "s20contz": ("216949414_*s20contz*",          "1frame"),
}


def main():
    print("Loading CMRx val bundles ...", flush=True)
    bundles, rcfg = D.load_bundles()
    results = {}
    for name, (pat, regime) in D.CKPTS.items():
        results[name] = D.eval_model(name, pat, regime, bundles, rcfg, save_probe=True)
    with open(os.path.join(D.OUT, "summary_1frame.json"), "w") as f:
        json.dump({"config": dict(subjs=D.SUBJS, breathing=D.BREATHING, regime="1frame_ALL"),
                   "results": results, "probe": D._PROBE}, f, indent=2)
    print("\n===== FINAL TABLE (CMRx in-dist, IDENTICAL 1frame batch, clean, n=%d) =====" % len(D.SUBJS), flush=True)
    print(f"{'model':10s} {'ref%':>6s} {'nonref%':>8s} {'full dB':>8s} {'corr(recon,GTbeat)':>20s} {'|dz|mm':>8s}", flush=True)
    for name in ("gather05", "s20", "s20contz"):
        r = results[name]
        pr = [p for p in D._PROBE if p["name"] == name]
        cg = float(np.mean([p["r_recon_vs_gt"] for p in pr])) if pr else float("nan")
        dz = float(np.mean([p["dz_abs_mm"] for p in pr])) if pr else float("nan")
        print(f"{name:10s} {r['ref_pct']:6.1f} {r['nonref_pct']:8.1f} {r['full_db']:8.2f} {cg:20.3f} {dz:8.2f}", flush=True)
    print("\nDONE -> result/s20_decider/summary_1frame.json", flush=True)


if __name__ == "__main__":
    main()
