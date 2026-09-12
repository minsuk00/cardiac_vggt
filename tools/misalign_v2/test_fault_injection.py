"""Fault-injection test for the misalignment metric (read-only; shifts are applied IN MEMORY to a clean
subject, nothing is written under scratch/data). Each case must be recovered by the seg+img consensus of
analyze.py's rules, otherwise the test FAILS. Cases:
  A  single-slice shift inside the search window (+5.4/-2.8 mm)        -> spike ~= |shift| at z, both estimators
  B  single-slice 28 mm shift (outside the OLD 20 mm window)            -> with MAXSHIFT_MM=40: recovered by both;
                                                                          with MAXSHIFT_MM=20: registration pinned,
                                                                          seg_fallback must still score it >= 25 mm
  C  3-slice block shifted 8.1 mm                                       -> step ~= 8 at both block edges (spike halved)
  D  direction check: seg and img both large but pointing apart          -> step keeps min(seg,img), not 0
Usage: python test_fault_injection.py [rel]   (default: a clean Z>=10 subject)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import metric_v2 as m
from analyze import gate_spikes, compute_steps

REL = sys.argv[1] if len(sys.argv) > 1 else "MNMs_sax/MNMs_A8I1U6"   # Z=12, severity 1.1 mm in the v3 sweep
seg0, img0, roi0, sp = m.load_subject(REL); dx, dy, dz = sp; Z = seg0.shape[2]
print(f"{REL}: shape {seg0.shape} spacing {sp}")

def shifted(zs, mm_x, mm_y):
    px, py = int(round(mm_x / dx)), int(round(mm_y / dy))
    seg, img, roi = seg0.copy(), img0.copy(), roi0.copy()
    for z in zs:
        seg[:, :, z] = np.roll(np.roll(seg[:, :, z], px, 0), py, 1); img[:, :, z] = np.roll(np.roll(img[:, :, z], px, 0), py, 1)
        roi[:, :, z] = np.roll(np.roll(roi[:, :, z], px, 0), py, 1)
    return (seg, img, roi), float(np.hypot(px * dx, py * dy))

def run(arrays, window):
    m.MAXSHIFT_MM = window
    df = pd.DataFrame(m.analyze_arrays(REL, *arrays, sp)); df["split"] = "test"
    gate_spikes(df); st = compute_steps(df)
    return df.set_index("z"), st.set_index("z")

fails = []
def check(name, cond, msg):
    print(("PASS" if cond else "FAIL"), name, msg); fails.append(name) if not cond else None

zm = Z // 2
# baseline
d0, s0 = run((seg0, img0, roi0), 40.0)
base_spike = d0.spike[d0.spike_ok].max(); base_step = s0.step[s0.step_ok].max()
print(f"baseline: max spike {base_spike:.2f}, max step {base_step:.2f} mm (subject must be clean for the test to be meaningful)")
check("baseline_clean", base_spike < 3 and base_step < 3, f"spike {base_spike:.2f} step {base_step:.2f}")

# A: in-window single slice
arr, mag = shifted([zm], 5.4, -2.8); d, s = run(arr, 40.0)
check("A_seg", abs(d.epi_spike[zm] - mag) < 1.5, f"seg spike {d.epi_spike[zm]:.2f} vs {mag:.2f}")
check("A_img", abs(d.img_spike[zm] - mag) < 1.5, f"img spike {d.img_spike[zm]:.2f} vs {mag:.2f}")
check("A_sign", np.hypot(d.epi_spike_x[zm] - d.img_spike_x[zm], d.epi_spike_y[zm] - d.img_spike_y[zm]) < 2.0, f"vector disagreement {np.hypot(d.epi_spike_x[zm] - d.img_spike_x[zm], d.epi_spike_y[zm] - d.img_spike_y[zm]):.2f} mm")
check("A_consensus", d.spike_ok[zm] and d.spike_src[zm] == "consensus" and abs(d.spike[zm] - mag) < 1.5, f"spike {d.spike[zm]:.2f} src {d.spike_src[zm]}")

# B: 28 mm single slice, both windows
arr, mag = shifted([zm], 28.0, 0.0)
d, s = run(arr, 40.0)
check("B40_img", abs(d.img_spike[zm] - mag) < 2.0, f"img spike {d.img_spike[zm]:.2f} vs {mag:.2f} (window 40)")
check("B40_consensus", d.spike_ok[zm] and abs(d.spike[zm] - mag) < 2.0, f"spike {d.spike[zm]:.2f} src {d.spike_src[zm]}")
check("B40_step", s.step[zm - 1] > 25 and s.step[zm] > 25, f"steps {s.step[zm-1]:.2f}/{s.step[zm]:.2f} src {s.step_src[zm-1]}/{s.step_src[zm]}")
d, s = run(arr, 20.0)
check("B20_pinned", d.img_pin_prev[zm] >= 0.5 and d.img_pin_next[zm] >= 0.5 and np.isnan(d.img_spike[zm]), f"pin prev/next {d.img_pin_prev[zm]:.2f}/{d.img_pin_next[zm]:.2f}, img spike {d.img_spike[zm]}")
check("B20_fallback", d.spike_ok[zm] and d.spike_src[zm] == "seg_fallback" and d.spike[zm] > 25, f"spike {d.spike[zm]:.2f} src {d.spike_src[zm]!r} (this is the ACDC_patient081 failure mode)")
check("B20_step_fallback", s.step_ok[zm - 1] and s.step[zm - 1] > 25, f"step z{zm-1} {s.step[zm-1]:.2f} src {s.step_src[zm-1]!r}")

# C: 3-slice block
blk = [zm - 1, zm, zm + 1]; arr, mag = shifted(blk, 8.1, 0.0); d, s = run(arr, 40.0)
check("C_step_edges", abs(s.step[zm - 2] - mag) < 2.0 and abs(s.step[zm + 1] - mag) < 2.0, f"steps {s.step[zm-2]:.2f}/{s.step[zm+1]:.2f} vs {mag:.2f}")
check("C_step_inside", s.step[zm - 1] < 3 and s.step[zm] < 3, f"inside-block steps {s.step[zm-1]:.2f}/{s.step[zm]:.2f}")
check("C_spike_halved", d.spike[zm] < 3 and 2 < d.spike[zm - 1] < 6.5, f"spike inside {d.spike[zm]:.2f}, edge {d.spike[zm-1]:.2f} (~{mag/2:.1f} expected)")

# D: direction check -- take case A's rows and rotate the image vector 90 deg in place (synthetic)
arr, mag = shifted([zm], 12.0, 0.0); d, s = run(arr, 40.0)
df = d.reset_index(); i = int(np.nonzero(df.z.values == zm - 1)[0][0])
df.loc[i, "img_dnext_x"], df.loc[i, "img_dnext_y"] = -df.loc[i, "img_dnext_y"], df.loc[i, "img_dnext_x"]   # rotate seg-vs-img by 90 deg
s2 = compute_steps(df).set_index("z")
check("D_both_large", s2.step_src[zm - 1] == "both_large" and s2.step[zm - 1] > 8, f"step {s2.step[zm-1]:.2f} src {s2.step_src[zm-1]!r} (seg {s2.step_seg[zm-1]:.1f} img {s2.step_img[zm-1]:.1f} agree {s2.step_agree[zm-1]:.1f})")
df.loc[i, "img_dnext_x"] *= 0.2; df.loc[i, "img_dnext_y"] *= 0.2   # img small & pointing elsewhere -> still contradict -> 0
s3 = compute_steps(df).set_index("z")
check("D_contradict", s3.step_src[zm - 1] == "contradict" and s3.step[zm - 1] == 0.0, f"step {s3.step[zm-1]:.2f} src {s3.step_src[zm-1]!r}")

print("\nRESULT:", "ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
