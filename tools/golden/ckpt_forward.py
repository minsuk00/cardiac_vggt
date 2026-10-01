"""G3: load each paper checkpoint the way eval does and fingerprint its forward outputs.

Uses inference/load_run.load_model_from_run + evaluation/src/engine/run_vggt.py (make_dataset,
reconstruct) unmodified, on the subject's own cine (no breathing bundle), in deterministic mode.
"""
import argparse
import os
import shutil
import tempfile

from common import REPO, dump, sha

import torch

torch.use_deterministic_algorithms(True)
import sys  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "evaluation", "src", "engine"))
from inference.load_run import load_model_from_run  # noqa: E402
from run_vggt import make_dataset, reconstruct  # noqa: E402

LOGS = "/home/minsukc/vggt/scratch/logs"
RUNS = {
    "diff1000": "210823094_final518_diff1000_curated898",
    "base": "210823094_final518_base_curated898",
    "nogather": "210498476_final518_nogather_curated898",
    "hw0": "210484679_final518_hw0_curated898",
}
STAGE = "/tmp/vggt_golden"
SUBJECTS = [
    "CMRxRecon2025/Cine_combined/CMRx25_R1val_Center004_Siemens_15T_Aera_P011",
    "CMRxRecon2025/Cine_combined/CMRx25_R1val_Center004_Siemens_15T_Aera_P017",
]
PHASES = [0, 4]


def staged_run_dir(arm):
    """<STAGE>/<arm>/{run_meta.jsonl, ckpts/checkpoint_last.pt} so load_run finds its run_meta."""
    d = os.path.join(STAGE, arm)
    os.makedirs(os.path.join(d, "ckpts"), exist_ok=True)
    shutil.copyfile(os.path.join(LOGS, RUNS[arm], "run_meta.jsonl"), os.path.join(d, "run_meta.jsonl"))
    link = os.path.join(d, "ckpts", "checkpoint_last.pt")
    if not os.path.exists(link):
        os.symlink(os.path.join(STAGE, "ckpts", f"{arm}.pt"), link)
    return link


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--arms", nargs="+", default=list(RUNS))
    a = p.parse_args()

    out = {}
    for arm in a.arms:
        ckpt = staged_run_dir(arm)
        model, cfg = load_model_from_run(ckpt, device="cuda")
        keys = sorted(model.state_dict().keys())
        res = {"n_keys": len(keys), "keys": sha(keys),
               "weights": sha([sha(v) for _, v in sorted(model.state_dict().items())])}
        splat_res = cfg["loss"]["volume"].get("splat_res")
        for subj in SUBJECTS:
            with tempfile.TemporaryDirectory() as tmp:
                ds = make_dataset(cfg, subj, "val", tmp)
                phases = ds.get_data(seq_index=12345, img_per_seq=ds.num_slices)["phases"]
                phases = torch.as_tensor(phases).float().numpy()
                vols, _, ed = reconstruct(model, ds, 12345, phases, "cuda", None,
                                          splat_res=splat_res, phases=PHASES)
            res[subj.rsplit("/", 1)[-1]] = {"V_canon": sha(vols), "delta_ed": sha(ed["delta"]),
                                           "images_ed": sha(ed["images"])}
        out[arm] = res
        print(arm, res, flush=True)
        del model
        torch.cuda.empty_cache()
    dump(out, a.out)


if __name__ == "__main__":
    main()
