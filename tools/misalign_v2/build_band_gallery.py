"""Full gallery for ONE severity band (clean/mild/moderate/severe) -- every measurable subject in that
band, sorted by severity, slab reslice panels. Companion to build_gallery.py (which is severe-only,
kept as _html/95b for backward compat) and build_categorized_examples.py (the curated 16-example page).
Usage: python build_band_gallery.py <clean|mild|moderate|severe>"""
import sys, os, base64, html
import pandas as pd
from PIL import Image
ROOT = "/home/minsukc/vggt"; OUT = f"{ROOT}/temp/misalign_v2"; SL = f"{OUT}/slabs"
BAND = sys.argv[1]
RANGES = {"clean": (-1, 3), "mild": (3, 5), "moderate": (5, 8), "severe": (8, 999), "unmeasurable": (None, None)}   # clean lower bound -1 so a severity of exactly 0.0 is included
lo, hi = RANGES[BAND]
sub = pd.read_csv(f"{OUT}/subjects_v3.csv", index_col=0)
if BAND == "unmeasurable":   # too few gated slices/pairs for an automatic score: kept by the rule, must be judged by eye
    s = sub[~sub.measurable].sort_values("severity", ascending=False, na_position="last")
else:
    sub = sub[sub.measurable]
    s = sub[(sub.severity > lo) & (sub.severity <= hi)].sort_values("severity", ascending=False)

def b64(p, maxw=1500):
    im = Image.open(p).convert("RGB")
    if im.width > maxw: im = im.resize((maxw, int(im.height * maxw / im.width)))
    import io; buf = io.BytesIO(); im.save(buf, "JPEG", quality=70); return base64.b64encode(buf.getvalue()).decode()

parts = []; missing = []
for src, g in s.groupby("source", sort=True):
    parts.append(f'<h2>{src} — {len(g)} subjects</h2>')
    for rel, r in g.iterrows():
        p = f"{SL}/{rel.split('/')[-1]}.png"
        if not os.path.exists(p):
            missing.append(rel); continue
        parts.append(f'<div class="ex"><div class="cap">{html.escape(rel)} · <b>severity {r.severity:.1f} mm</b> '
                     f'(spike {r.max_spike:.1f}, step {r.max_step if pd.notna(r.max_step) else float("nan"):.1f}'
                     f'{"; † seg-only: registration beyond its search window" if (r.n_spike_fallback > 0 or r.n_step_fallback > 0) else ""}) · '
                     f'{r.centre_key} · split {r.split}</div><img src="data:image/jpeg;base64,{b64(p)}"></div>')

label = {"clean": "clean (≤3 mm)", "mild": "mild (3–5 mm)", "moderate": "moderate (5–8 mm)", "severe": "severe (>8 mm)", "unmeasurable": "unmeasurable (no automatic score — fewer than 3 gated slices and pairs; kept by the rule, judge by eye)"}[BAND]
H = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>All {label} subjects ({len(s)})</title>
<style>body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1600px;margin:20px auto;padding:0 16px;color:#222}}.ex{{margin:10px 0 18px}}.cap{{font-size:12px;margin-bottom:2px}}img{{max-width:100%;border:1px solid #ddd}}h2{{border-bottom:1px solid #ccc;margin-top:30px}}.nav{{font-size:13px;margin-bottom:16px}}</style></head><body>
<div class="nav">Categories: <a href="95d_gallery_clean.html">clean</a> · <a href="95d_gallery_mild.html">mild</a> · <a href="95d_gallery_moderate.html">moderate</a> · <a href="95d_gallery_severe.html">severe</a> · <a href="95d_gallery_unmeasurable.html">unmeasurable</a> · curated examples: <a href="95c_misalignment_examples_by_category.html">95c</a> · main report: <a href="95_slice_misalignment_reanalysis.html">95</a></div>
<h1>All {len(s)} <b>{label}</b> subjects</h1>
<p>Every native SAX slice stacked as an equal-height band, back to back, no gap, no interpolation (each band is a flat unsmoothed fill of that slice's own pixels). Yellow ticks = epicardial-mask extent on that cut. Red ▶ = slice with consensus spike &gt;5 mm (most moderate/mild subjects here are flagged by a slice-pair step instead, which has no separate marker — check the per-slice numbers on the right for the actual flagged transition). In the number column, † marks a value taken from the segmentation alone because the image registration was pinned at its 40 mm search bound (offset beyond the window — treated as agreement, not as "no shift"). Left = coronal cut, right = sagittal cut, two perpendicular views through the LV centre. {len(missing)} subjects failed to render.</p>
{"".join(parts)}</body></html>"""
open(f"{ROOT}/_html/95d_gallery_{BAND}.html", "w").write(H)
print(BAND, "gallery", len(s), "missing", len(missing), "bytes", len(H))
