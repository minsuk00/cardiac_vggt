#!/usr/bin/env python
"""Rank tools/sweep_input_frame.py scores: per (rhythm, subject, plane), the frame maximising
min(ED motion, ES motion); prints the top --k per rhythm. With --specs, also writes the best plane of each
of the top --k SUBJECTS per rhythm, one line '<rhythm> <source>:<subject>:<z>:<frame> <score> <ED> <ES>',
the input of tools/sweep_input_frame.py (--subjects, 4-field form, to save fields) and
tools/render_motion_edes_gallery.py.

Usage: python tools/rank_input_frame_sweep.py scratch/motion_edes/sweep80 --k 8 --specs temp/gallery_specs.txt
"""
import argparse
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--specs", help="write the top --k subjects (best plane each) per rhythm here")
    a = ap.parse_args()
    lines = []
    for rh in ("af12", "hrv12"):
        rows = []
        for f in glob.glob(f"{a.sweep}/scores/{rh}/*.json"):
            s = json.load(open(f))
            fr, m = max(s["frames"].items(), key=lambda kv: min(kv[1].values()))
            fz = s["frames"][str(s["frozen_frame"])]
            rows.append((min(m.values()), s["subject"], s["z"], int(fr), m["ED"], m["ES"],
                         s["frozen_frame"], min(fz.values()), s["cohort"][:-len(rh) - 1]))
        rows.sort(reverse=True)
        print(f"== {rh} ({len(rows)} planes)  score = min(ED, ES) mm")
        for sc, sid, z, fr, ed, es, ff, fsc, _ in rows[:a.k]:
            print(f"  {sc:4.2f}  {sid} z{z} f{fr}  ED {ed:.2f}  ES {es:.2f}   (frozen f{ff}: {fsc:.2f})")
        seen = set()
        for sc, sid, z, fr, ed, es, _, _, src in rows:
            if sid not in seen and len(seen) < a.k:
                seen.add(sid)
                lines.append(f"{rh} {src}:{sid}:{z}:{fr} {sc:.2f} {ed:.2f} {es:.2f}")
    if a.specs:
        open(a.specs, "w").write("\n".join(lines) + "\n")
        print("wrote", a.specs)


if __name__ == "__main__":
    main()
