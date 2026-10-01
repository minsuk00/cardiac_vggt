"""Run golden gates on the current tree and diff each against the reference outputs.

  python tools/golden/check.py --out temp/golden/<tag> [--gates g0b g1 g1b g4 g6 g7]
  python tools/golden/check.py --out temp/golden/<tag> --gates g5 g3 --gpu 0     # GPU gates

Reference = temp/golden/ref (recorded on the pre-B/C code, ../vggt-golden-final @ 2ba8f23).
Prints every differing JSON path (first 30 per gate). Exit 1 if any gate differs. A difference
is a stop signal: explain it or fix it, never wave it through.
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Differences that are deliberate (one regex per line, matched against a diff line).
EXPECTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected_diffs.txt")
G1_VARIANTS = ["paper", "multiframe", "static", "tfixed0", "noref"]
CPU = ["g0b", "g1", "g1b", "g4", "g6", "g7"]


def jobs(gate, out):
    """-> [(json name, argv)]"""
    py = ["python"]
    if gate == "g0b":
        return [("g0b.json", py + ["tools/golden/compose_sbatch.py"])]
    if gate == "g1":
        return [(f"g1_{v}.json", py + ["tools/golden/sample_trace.py", "--variant", v]) for v in G1_VARIANTS]
    if gate == "g1b":
        return [("g1b_cpu.json", py + ["tools/golden/aug_trace.py", "--device", "cpu"])]
    if gate == "g1b_cuda":
        return [("g1b_cuda.json", py + ["tools/golden/aug_trace.py", "--device", "cuda"])]
    if gate == "g4":
        return [("g4.json", py + ["tools/golden/cpu_grad.py"])]
    if gate == "g6":
        return [("g6.json", py + ["tools/golden/model_init.py", "--ckpts", "/tmp/vggt_golden/ckpts/diff1000.pt"])]
    if gate == "g7":
        return [("g7.json", py + ["tools/golden/runmeta_sweep.py"])]
    if gate == "g3":
        return [("g3.json", py + ["tools/golden/ckpt_forward.py"])]
    if gate == "g5":
        return [(f"g5_{v}/golden.json", py + ["tools/golden/run_val.py", "--variant", v,
                                                "--out", os.path.join(out, f"g5_{v}")])
                for v in ("off", "on")]
    raise SystemExit(f"unknown gate {gate}")


def diff(a, b, path="", acc=None):
    acc = [] if acc is None else acc
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a:
                acc.append(f"{path}/{k}: <absent> -> present")
            elif k not in b:
                acc.append(f"{path}/{k}: present -> <absent>")
            else:
                diff(a[k], b[k], f"{path}/{k}", acc)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", acc)
    elif a != b or type(a) is not type(b):
        acc.append(f"{path}: {str(a)[:80]!s} -> {str(b)[:80]!s}")
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ref", default=os.path.join(REPO, "temp", "golden", "ref"))
    p.add_argument("--gates", nargs="+", default=CPU)
    p.add_argument("--gpu", default="")
    a = p.parse_args()

    import re
    expected = [re.compile(l.strip()) for l in open(EXPECTED)
                if l.strip() and not l.startswith("#")] if os.path.exists(EXPECTED) else []
    env = {**os.environ, "PYTHONPATH": f"{REPO}/training:{REPO}", "CUDA_VISIBLE_DEVICES": a.gpu}
    failed = []
    for gate in a.gates:
        for name, argv in jobs(gate, a.out):
            path = os.path.join(a.out, name)
            if gate != "g5":
                argv = argv + ["--out", path]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # g5's run dir must start empty, so its log goes next to it, not inside.
            log = (os.path.dirname(path) if gate == "g5" else path.replace(".json", "")) + ".log"
            r = subprocess.run(argv, cwd=REPO, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
            if r.returncode != 0:
                failed.append(name)
                print(f"[{name}] CRASHED (see {log})")
                continue
            ref, new = json.load(open(os.path.join(a.ref, name))), json.load(open(path))
            if gate == "g6":
                ref.pop("strict_load", None)
                new.pop("strict_load", None)
            d = diff(ref, new)
            n_expected = len(d)
            d = [l for l in d if not any(e.search(f"{name}:{l}") for e in expected)]
            n_expected -= len(d)
            note = f" ({n_expected} expected diffs ignored)" if n_expected else ""
            print(f"[{name}] {'PASS' if not d else f'{len(d)} DIFFERENCES'}{note}")
            for line in d[:30]:
                print("   ", line)
            if d:
                failed.append(name)
    print("ALL PASS" if not failed else f"FAILED: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
