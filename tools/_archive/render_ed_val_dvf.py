"""Per-slice DVF (Δx/Δy/Δz in mm) for the 3 reference-conditioned models on the CMRxRecon val
set at ED (t=0), respiration ON — the displacement-field companion to result/ed_val_many_io.

Per (model, val subject): one trainer-style panel = rows [input intensity, Δx, Δy, Δz] × the S
input slots (slot 0 = the target-phase reference, marked [ref]). Δ = world_points - scanner_coords,
scaled to mm (in-plane (256-1)/2·1.4; through-plane (12-1)/2·12). Inputs are breathing-corrupted
(resp ON, deterministic per seq_index), so Δz shows how the model corrects the SI breath shift.

Reuses render_dvf / forward from render_reference_5dataset_io and the val adapter from
five_row_compare. Each model is loaded once.

Run: micromamba run -n svr python tools/render_ed_val_dvf.py
"""
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from tools.render_reference_5dataset_io import MODELS, forward, render_dvf
from tools.five_row_compare import DEV, val_batch, build_val_dataset
from vggt.models.vggt import VGGT

OUT = os.path.join(_ROOT, "result", "ed_val_dvf")
N_SUBJECTS = 30


def build_model(ckpt, head):
    m = VGGT(img_size=518, patch_size=14, embed_dim=1024,
             enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
             use_z_pose_embedding=True, use_t_pose_embedding=False,
             use_target_t_pose_embedding=False, use_reference_token=True,
             train_on_residual_dvf=True, warp_head_type=head).to(DEV).eval()
    ck = torch.load(os.path.join(_ROOT, ckpt), map_location="cpu", weights_only=False)
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    assert not miss and not unexp, f"missing={miss[:5]} unexpected={unexp[:5]}"
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    val_ds, rcfg = build_val_dataset()
    val_ds.t_target_fixed = 0  # ED

    for name, ckpt, head in MODELS:
        print(f"=== {name} ({head}) ===", flush=True)
        model = build_model(ckpt, head)
        mdir = os.path.join(OUT, name); os.makedirs(mdir, exist_ok=True)
        for seq in range(N_SUBJECTS):
            try:
                batch = val_batch(val_ds, rcfg, seq, breathing=True)
            except Exception as e:
                print(f"  skip seq{seq}: {e}", flush=True); continue
            _V_can, dvf = forward(model, batch)
            render_dvf(batch, dvf,
                       f"DVF — {name} · val seq{seq} (ED, resp ON)",
                       os.path.join(mdir, f"val_seq{seq:02d}_ED_resp_dvf.png"))
        del model
        torch.cuda.empty_cache()
    print(f"\ndone -> {os.path.relpath(OUT, _ROOT)}", flush=True)


if __name__ == "__main__":
    main()
