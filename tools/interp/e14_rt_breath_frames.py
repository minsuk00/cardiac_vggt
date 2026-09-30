"""E14 — real RT scan, non-reference frames chosen to maximise breathing misalignment (illustration only).

Same pipeline as E8. Step 1: for each non-reference slice s and every `step`-th frame f (all other slices at
frame 0, reference at frame 0), read the full model's predicted mean through-plane shift of slice s (mm).
Step 2: alternate slices take the frame with the largest vs smallest predicted shift. Step 3: reconstruct that
input (reference at frame 0) under full / ko_ref_L3 / self_L12_17 and save the volumes + chosen frames.
"""
import argparse
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import torch                # noqa: E402
import harness as H         # noqa: E402
sys.path.insert(0, os.path.join(H.ROOT, "evaluation", "src", "engine"))
sys.path.insert(0, os.path.join(H.ROOT, "evaluation", "src", "analysis"))
import run_vggt_rt as rt    # noqa: E402
from data.gpu_aug import gpu_augment_batch   # noqa: E402
from e2_breathing import slot_allow          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="MIITT_Volunteer1")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--frame", type=int, default=None,
                    help="skip the search: every slot (reference included) takes this frame (paper Fig. 6 naive stack)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    model, cfg = H.load("cuda:0")
    splat_res = ((cfg.get("loss") or {}).get("volume") or {}).get("splat_res")
    rt_path = os.path.join(rt.RT_ROOT, args.subject.replace("MIITT_", ""), "realtime/sax/4d_recon.nii.gz")
    bundle = rt.build_rt_bundle(rt_path)
    n_frames, D = bundle.shape[:2]
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = rt.rv.make_dataset(cfg, rt.make_rt_scaffold(rt_path, args.subject, tmpdir), "val", tmpdir)
        b0 = ds.get_data(seq_index=0, img_per_seq=ds.num_slices)
        dz_mm = float(np.asarray(b0["dz_mm"]).reshape(-1)[0])
        batch = rt.build_batch_rt(ds, rt.rv.name_seed("miitt", args.subject), bundle, dev)
    S = batch["timesteps"].shape[1]
    z_scale = float(batch["z_scale"].reshape(-1)[0])
    fake = SimpleNamespace(R=int(batch["scanner_coords"].shape[-2]))

    def fwd(frames, allow=None, layers=None):
        batch["timesteps"][0, :] = torch.as_tensor(frames, device=batch["timesteps"].device)
        batch.pop("images", None)
        gpu_augment_batch(batch, None, dev, respiratory_cfg=None, train=False)
        b = {"images": batch["images"], "z_indices": batch["z_indices"], "scanner_coords": batch["scanner_coords"]}
        return H.forward(model, fake, b, allow=allow, layers=layers)["wp"], b

    base = np.zeros(S, dtype=int)
    frames = [] if args.frame is not None else list(range(0, n_frames, args.step))
    shift = np.zeros((S, len(frames)))
    for s in (range(1, S) if args.frame is None else []):
        for k, f in enumerate(frames):
            fr = base.copy(); fr[s] = f
            wp, b = fwd(fr)
            dvf = (wp - b["scanner_coords"])[0, s, ..., 2]                    # normalized z
            m = b["images"][0, s].mean(0) > 0.05
            shift[s, k] = float(dvf[m].mean()) * H.Z_HALF_MM
        print(f"slot {s}: predicted shift range {shift[s].min():.1f} .. {shift[s].max():.1f} mm", flush=True)
    chosen = base.copy()
    if args.frame is not None:
        chosen[:] = args.frame
    else:
        for s in range(1, S):
            chosen[s] = frames[int(np.argmax(shift[s]) if s % 2 else np.argmin(shift[s]))]
    ko3 = torch.ones(2 * S, 2 * S, dtype=torch.bool); ko3[2:, 0:2] = False
    eye = slot_allow(torch.eye(S, dtype=torch.bool))
    rec = {"frames": chosen, "shift_curves": shift, "frame_grid": np.asarray(frames), "dz_mm": dz_mm}
    for arm, (allow, layers) in {"full": (None, None), "ko_ref_L3": (ko3, {3}),
                                 "self_L12_17": (eye, set(range(12, 18)))}.items():
        wp, b = fwd(chosen, allow, layers)
        V, _ = rt._splat_preds_native({"world_points": wp}, batch, (D, 256, 256), z_scale, splat_res=splat_res)
        rec[arm] = V[0].float().cpu().numpy().astype(np.float16)
        if arm == "full":
            dvf = (wp - b["scanner_coords"])[0, ..., 2]
            rec["pred_shift_chosen"] = np.array([float(dvf[s][b["images"][0, s].mean(0) > 0.05].mean()) * H.Z_HALF_MM
                                                 for s in range(S)])
    os.makedirs(args.out, exist_ok=True)
    tag = "" if args.frame is None else f"_f{args.frame:03d}"
    np.savez_compressed(os.path.join(args.out, f"{args.subject}{tag}.npz"), **rec)
    print("chosen frames", chosen.tolist(), "pred shift", np.round(rec["pred_shift_chosen"], 1).tolist(), flush=True)


if __name__ == "__main__":
    main()
