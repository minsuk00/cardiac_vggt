"""E1b — double dissociation + what part of the reference is read.

Same protocol and outputs as E1 (reference-phase sweep, fixed companions + breathing), new conditions:
  full              paired control
  self_L12_17       no cross-slice attention in global layers 12-17 (kills breathing, E2) — cardiac?
  self_L6_17        same for 6-17
  ko_refheart_L0_5  companions may not see the reference's HEART-ROI patch tokens, layers 0-5
  ko_refbg_L0_5     ... the reference's NON-heart patch tokens (the complement), layers 0-5
  ref_offmid        the reference token is put on a different plane (a companion 1/4-stack away);
                    tracking is scored on all planes except the new reference plane
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
from e2_breathing import slot_allow   # noqa: E402


def token_sets(subj, b, P):
    """query_sel = all companion tokens; heart/bg = slot-0 patch tokens inside / outside the heart."""
    S = len(b["slot_z"])
    N = S * P
    hp = H.heart_patch_mask(subj, b)                       # (S, G*G)
    qsel = torch.zeros(N, dtype=torch.bool); qsel[P:] = True
    heart = torch.zeros(N, dtype=torch.bool)
    bg = torch.zeros(N, dtype=torch.bool)
    heart[H.N_SPECIAL:P] = hp[0].cpu()
    bg[H.N_SPECIAL:P] = ~hp[0].cpu()
    return qsel, heart, bg


def score(subj, b, wp, t, comp_planes, ref_plane, common_planes=None):
    dvf = wp - b["scanner_coords"]
    V, _ = H.splat(subj, b, wp)
    Vc, _ = H.splat_companions(subj, b, wp)                          # reference slice's pixels removed
    far = [z for z in comp_planes if abs(z - ref_plane) >= 2]      # >=2 planes from the ref (most, not all, leakage)
    return {"t": t, "disp_fov": H.slot_disp_mm(subj, b, dvf, "fov").tolist(),
            "ncc_conly": H.phase_ncc_profile(subj, Vc, comp_planes).tolist(),   # leakage-free primary metric
            "ncc_conly_common": H.phase_ncc_profile(subj, Vc, common_planes).tolist() if common_planes else None,
            "obs": H.obs_err(subj, b, wp, t).tolist(),
            "ncc_comp": H.phase_ncc_profile(subj, V, comp_planes).tolist(),
            "ncc_far": H.phase_ncc_profile(subj, V, far).tolist() if far else None,
            # same plane set for every condition (excludes BOTH the mid plane and the off-mid plane)
            "ncc_common": H.phase_ncc_profile(subj, V, common_planes).tolist() if common_planes else None,
            "ncc_ref": H.phase_ncc_profile(subj, V, [ref_plane]).tolist()}


def run_subject(model, cfg, ds, idx, dev):
    subj = H.Subject(ds, idx, dev)
    rng = np.random.default_rng(1000 + idx)                 # SAME companions/breathing as E1
    comp = rng.integers(0, subj.T, subj.S)
    disp, _ = subj.sample_breathing(H.resp_cfg(cfg), seed=2000 + idx)
    ref_plane = subj.ref_plane
    comp_planes = [z for z in range(subj.D) if z != ref_plane]
    rec = {"subject": subj.name, "idx": idx, "D": subj.D, "S": subj.S, "T": subj.T, "dz": subj.dz,
           "slot_z": subj.slot_z.tolist(), "comp_t": comp.tolist(), "ref_plane": ref_plane,
           "disp": disp.cpu().numpy().tolist(), "cond": {}}
    if subj.heart is None:
        return rec
    P = H.N_SPECIAL + (subj.R // 14) ** 2
    eye = slot_allow(torch.eye(subj.S, dtype=torch.bool))
    # off-mid reference: swap slot 0 with the companion whose plane is ~D/4 away
    j = min(range(1, subj.S), key=lambda s: abs(subj.slot_z[s] - (ref_plane + max(1, subj.D // 4))))
    perm = list(range(subj.S)); perm[0], perm[j] = j, 0
    new_ref_plane = int(subj.slot_z[j])
    common = [z for z in range(subj.D) if z not in (ref_plane, new_ref_plane)]
    rec["offmid_plane"] = new_ref_plane
    names = ["full", "self_L12_17", "self_L6_17", "ko_refheart_L0_5", "ko_refbg_L0_5", "ref_offmid",
             "ko_ref_L3", "ko_ref", "identity", "ko_refheart_L3", "ko_refbg_L3"]
    S = subj.S
    ko_all = torch.ones(2 * S, 2 * S, dtype=torch.bool); ko_all[2:, 0:2] = False
    for name in names:
        rows = []
        for t in range(subj.T):
            ts = comp.copy(); ts[0] = t
            if name == "ref_offmid":
                sz = subj.slot_z[perm]; ts2 = comp[perm].copy(); ts2[0] = t
                d2 = disp[perm]
                b = H.build_batch(subj, ts2, d2, slot_z=sz)
                wp = H.forward(model, subj, b)["wp"]
                row = score(subj, b, wp, t, [z for z in range(subj.D) if z != new_ref_plane], new_ref_plane,
                            common)
                row["disp_slot"] = d2.cpu().numpy().tolist()      # slot order of THIS run (permuted)
                rows.append(row)
                continue
            b = H.build_batch(subj, ts, disp)
            if name == "full":
                wp = H.forward(model, subj, b)["wp"]
            elif name == "identity":
                wp = b["scanner_coords"].clone()
            elif name == "ko_ref_L3":
                wp = H.forward(model, subj, b, allow=ko_all, layers={3})["wp"]
            elif name == "ko_ref":
                wp = H.forward(model, subj, b, allow=ko_all)["wp"]
            elif name == "self_L12_17":
                wp = H.forward(model, subj, b, allow=eye, layers=set(range(12, 18)))["wp"]
            elif name == "self_L6_17":
                wp = H.forward(model, subj, b, allow=eye, layers=set(range(6, 18)))["wp"]
            else:
                qsel, heart, bg = token_sets(subj, b, P)
                hide = heart if name.startswith("ko_refheart") else bg
                lay = {3} if name.endswith("_L3") else set(range(6))
                wp = H.forward(model, subj, b, layers=lay, key_hide=hide, query_sel=qsel)["wp"]
            rows.append(score(subj, b, wp, t, comp_planes, ref_plane, common))
        rec["cond"][name] = rows
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
