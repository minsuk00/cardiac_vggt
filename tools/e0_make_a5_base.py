"""ARM A5 (docs/66): build a base-weights file carrying the HISTORICAL z-embedder init.

The only randomly initialized tensors in a fresh run are the z-embedder's (every other
weight comes from VGGT-1B; the two z tensors are missing from the base and keep their
random init after the strict=false load). 4wok's effective GPU seed was
seed_value(42) x max_epochs(200) = 8400; the current hub's is 42 x 300 = 12600. This
script replays the OLD tree's exact init path — set_seeds(42, 200, 0) then
instantiate(cfg.model) — captures `aggregator.z_embedder.*`, injects them into a COPY of
vggt1b_base.pt, and writes it as a new file. The A5 arm then only overrides
`checkpoint.resume_checkpoint_path`: the injected tensors are present in the ckpt, so the
strict=false load OVERWRITES the current-seed random init with the 8400 one, while every
other RNG stream (sampling, augmentation, dropout) keeps the hub seed.

Run:
  PYTHONPATH=/home/minsukc/vggt-oldcode-p0/training:/home/minsukc/vggt-oldcode-p0 \
  micromamba run -n svr python tools/e0_make_a5_base.py \
      --tree /home/minsukc/vggt-oldcode-p0 --config mri_volume_diffusion \
      --base scratch/base_weights/vggt1b_base.pt \
      --out scratch/base_weights/vggt1b_base_a5_z8400.pt
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-value", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=200)
    args = ap.parse_args()

    tree = os.path.abspath(args.tree)
    sys.path.insert(0, os.path.join(tree, "training"))
    sys.path.insert(1, tree)
    os.chdir(tree)

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from train_utils.general import set_seeds

    for name, fn in [("rev_ts", lambda: "0"),
                     ("basename", lambda p: os.path.basename(p)),
                     ("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}")]:
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception:
            pass

    with initialize_config_dir(version_base=None, config_dir=os.path.join(tree, "training", "config")):
        cfg = compose(config_name=args.config)

    # Replay the old Trainer's init sequence: set_seeds -> instantiate(model). The z-embedder's
    # nn.Linear init lands at the same point of the seeded RNG stream as in the historical run.
    set_seeds(args.seed_value, args.max_epochs, 0)   # -> effective GPU seed 42*200 = 8400
    model = instantiate(cfg.model, _recursive_=False)

    z_tensors = {k: v.detach().clone() for k, v in model.state_dict().items()
                 if "z_embedder" in k}
    assert z_tensors, "no z_embedder tensors found in the old model"
    print(f"[a5] captured {list(z_tensors.keys())} "
          f"(norms: {[round(float(v.norm()), 6) for v in z_tensors.values()]})")

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    state = base["model"] if "model" in base else base
    for k in z_tensors:
        assert k not in state, f"{k} unexpectedly already present in the base ckpt"
    state.update({k: v for k, v in z_tensors.items()})
    out = os.path.abspath(os.path.join("/home/minsukc/vggt", args.out)) \
        if not os.path.isabs(args.out) else args.out
    torch.save({"model": state} if "model" not in base else base, out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"[a5] wrote {out}")


if __name__ == "__main__":
    main()
