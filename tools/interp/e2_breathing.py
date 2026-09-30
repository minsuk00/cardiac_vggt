"""E2 — where does the through-plane (breathing) correction come from?

A. Pathway knockouts on global attention, K breathing draws per subject (val-style sampler):
     full / identity / self_only / self_ref / nbr1 / nbr1_ref / no_nbr1 / ko_ref /
     self_only restricted to global-layer blocks (self_L{a}_{b}) and prefixes/suffixes.
   'nbr1' = a slot may attend only to itself and to slots whose plane is adjacent (|dz|=1 plane).
B. Single-slot content dose-response: no breathing anywhere except slot k, shifted by δ in z only.
C. Single-slot z-token lie: no breathing; slot k's z-token moved by δ mm, image unchanged.
D. Common-mode shift: every slot (optionally except the reference) shifted by the same δ.

Per forward we store the per-slot predicted (dz, dy, dx) mm over the in-FOV pixels.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import torch                # noqa: E402
import harness as H         # noqa: E402

DELTAS = [-24, -18, -12, -6, -3, 0, 3, 6, 12, 18, 24]


def slot_allow(slot_ok):
    """(S,S) bool slot->slot -> (2S,2S) group matrix."""
    return slot_ok.repeat_interleave(2, 0).repeat_interleave(2, 1)


def pathway_conds(slot_z):
    S = len(slot_z)
    z = torch.as_tensor(slot_z)
    eye = torch.eye(S, dtype=torch.bool)
    adj = (z[:, None] - z[None, :]).abs() == 1
    ref = torch.zeros(S, S, dtype=torch.bool); ref[:, 0] = True
    full = torch.ones(S, S, dtype=torch.bool)
    ko_ref = full.clone(); ko_ref[1:, 0] = False
    c = {"full": None, "self_only": slot_allow(eye), "self_ref": slot_allow(eye | ref),
         "nbr1": slot_allow(eye | adj), "nbr1_ref": slot_allow(eye | adj | ref),
         "no_nbr1": slot_allow(~adj), "ko_ref": slot_allow(ko_ref)}
    out = {k: (v, None) for k, v in c.items()}
    for a in range(0, 24, 6):
        out[f"self_L{a}_{a + 5}"] = (slot_allow(eye), set(range(a, a + 6)))
    for a in (6, 12, 18):
        out[f"self_from{a}"] = (slot_allow(eye), set(range(a, 24)))
        out[f"self_upto{a - 1}"] = (slot_allow(eye), set(range(a)))
    return out


def fwd_disp(model, subj, b, allow=None, layers=None):
    o = H.forward(model, subj, b, allow=allow, layers=layers)
    return H.slot_disp_mm(subj, b, o["dvf"], "fov")


def run_subject(model, cfg, ds, idx, dev, K=3):
    subj = H.Subject(ds, idx, dev)
    rec = {"subject": subj.name, "idx": idx, "D": subj.D, "S": subj.S, "dz": subj.dz,
           "slot_z": subj.slot_z.tolist(), "A": [], "B": {}, "C": {}, "D_": {}}
    rng = np.random.default_rng(3000 + idx)
    conds = pathway_conds(subj.slot_z)
    rcfg = H.resp_cfg(cfg)
    # ---- A: pathway knockouts under realistic breathing
    for k in range(K):
        ts = rng.integers(0, subj.T, subj.S)
        disp, _ = subj.sample_breathing(rcfg, seed=4000 + 10 * idx + k)
        b = H.build_batch(subj, ts, disp)
        row = {"t": ts.tolist(), "disp": disp.cpu().numpy().tolist(), "pred": {}}
        for name, (allow, layers) in conds.items():
            row["pred"][name] = fwd_disp(model, subj, b, allow, layers).tolist()
        rec["A"].append(row)
    # ---- B/C/D use one fixed phase draw, no breathing baseline
    ts = rng.integers(0, subj.T, subj.S)
    rec["BCD_t"] = ts.tolist()
    # probe slot: the companion whose plane is closest to 1/4 of the stack from the reference
    order = [s for s in range(1, subj.S)]
    k = min(order, key=lambda s: abs(subj.slot_z[s] - (subj.ref_plane + subj.D // 4) % subj.D))
    rec["probe_slot"] = int(k)
    sub = {"full": None, "self_only": conds["self_only"][0], "nbr1": conds["nbr1"][0],
           "self_ref": conds["self_ref"][0]}
    for name, allow in sub.items():
        rec["B"][name] = []
        for dlt in DELTAS:
            disp = np.zeros((subj.S, 3), np.float32); disp[k, 0] = dlt
            b = H.build_batch(subj, ts, disp)
            rec["B"][name].append(fwd_disp(model, subj, b, allow).tolist())
    rec["C"]["full"] = []
    for dlt in DELTAS:
        zo = np.zeros(subj.S, np.float32); zo[k] = dlt
        b = H.build_batch(subj, ts, None, z_override=zo)
        rec["C"]["full"].append(fwd_disp(model, subj, b).tolist())
    for variant in ("all", "all_but_ref"):
        rec["D_"][variant] = []
        for dlt in DELTAS:
            disp = np.zeros((subj.S, 3), np.float32); disp[:, 0] = dlt
            if variant == "all_but_ref":
                disp[0, 0] = 0
            b = H.build_batch(subj, ts, disp)
            rec["D_"][variant].append(fwd_disp(model, subj, b).tolist())
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda:0"
    model, cfg = H.load(dev)
    ds = H.make_test_dataset(cfg)
    idxs = list(range(len(ds.subjects)))[args.shard::args.nshard]
    if args.limit:
        idxs = idxs[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    for i in idxs:
        fp = os.path.join(args.out, f"s{i:03d}.json")
        if os.path.exists(fp):
            continue
        t0 = time.time()
        rec = run_subject(model, cfg, ds, i, dev)
        json.dump(rec, open(fp, "w"))
        print(f"[{i}] {rec['subject']} S={rec['S']} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
