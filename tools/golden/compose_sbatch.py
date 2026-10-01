"""G0b: compose every ARM of sbatch/train_final_518.sh exactly as the script would, and dump the
resolved configs. Fails if any arm no longer composes (e.g. an override of a deleted key).

  python tools/golden/compose_sbatch.py --out temp/golden/<tag>/g0b.json
"""
import argparse
import os
import re

from common import REPO, dump

from omegaconf import OmegaConf


def sbatch_overrides(txt):
    peak = re.search(r'^PEAK_LR="(.*?)"', txt, re.M).group(1)
    recipe = re.search(r'^RECIPE_OVERRIDES="(.*?)"', txt, re.M | re.S).group(1)
    aug = re.search(r'^AUG_OVERRIDES="(.*?)"', txt, re.M).group(1)
    arms = dict(re.findall(r'^\s*(\w+)\)\s*ARM_OVERRIDES="(.*?)"', txt, re.M))
    out = {}
    for arm, arm_ov in arms.items():
        s = (recipe.replace("${PEAK_LR}", peak).replace("${ARM_OVERRIDES}", arm_ov)
             .replace("${EXPERIMENT_OVERRIDES:-}", "").replace("\\\n", " "))
        out[arm] = s.split() + aug.split() + ["exp_name=golden"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from resolvers import register_all
    from hydra import compose, initialize_config_dir
    register_all()
    txt = open(os.path.join(REPO, "sbatch", "train_final_518.sh")).read()
    resolved = {}
    for arm, ov in sbatch_overrides(txt).items():
        with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "training", "config")):
            cfg = compose(config_name="default", overrides=ov)
        resolved[arm] = OmegaConf.to_container(cfg, resolve=True)
    dump(resolved, a.out)
    print(f"composed {len(resolved)} arms: {sorted(resolved)}")


if __name__ == "__main__":
    main()
