"""Gallery HTML of EVERY subject above a severity threshold (default >8 mm), slab panels embedded,
grouped by source, sorted by severity. Also writes contact sheets (grids) for quick visual QA.
Usage: python build_gallery.py [thr_mm]"""
import sys, os, base64, html, numpy as np, pandas as pd
from PIL import Image
ROOT = "/home/minsukc/vggt"; OUT = f"{ROOT}/temp/misalign_v2"; SL = f"{OUT}/slabs"
thr = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
sub = pd.read_csv(f"{OUT}/subjects_v3.csv", index_col=0); s = sub[sub.measurable & (sub.severity > thr)].sort_values("severity", ascending=False)
def b64(p, maxw=1500):
    im = Image.open(p).convert("RGB")
    if im.width > maxw: im = im.resize((maxw, int(im.height * maxw / im.width)))
    import io; buf = io.BytesIO(); im.save(buf, "JPEG", quality=70); return base64.b64encode(buf.getvalue()).decode()
parts = []; missing = []
for src, g in s.groupby("source", sort=True):
    parts.append(f'<h2>{src} — {len(g)} subjects &gt;{thr:g} mm (of {int((sub.source==src)&sub.measurable).sum() if False else (sub[(sub.source==src)&sub.measurable]).shape[0]} measurable)</h2>')
    for rel, r in g.iterrows():
        p = f"{SL}/{rel.split('/')[-1]}.png"
        if not os.path.exists(p): missing.append(rel); continue
        parts.append(f'<div class="ex"><div class="cap">{html.escape(rel)} · <b>severity {r.severity:.1f} mm</b> (spike {r.max_spike:.1f}, step {r.max_step:.1f}) · {r.centre_key} · split {r.split}</div><img src="data:image/jpeg;base64,{b64(p)}"></div>')
H = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Every subject with a &gt;{thr:g} mm inter-slice discontinuity ({len(s)})</title>
<style>body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1600px;margin:20px auto;padding:0 16px;color:#222}}.ex{{margin:10px 0 18px}}.cap{{font-size:12px;margin-bottom:2px}}img{{max-width:100%;border:1px solid #ddd}}h2{{border-bottom:1px solid #ccc;margin-top:30px}}</style></head><body>
<h1>All {len(s)} subjects with a &gt;{thr:g} mm inter-slice in-plane discontinuity</h1>
<p>Companion to <code>_html/95_slice_misalignment_reanalysis.html</code>. A curated 16-example version organized by category, spanning each band's own severity range, is <a href="95c_misalignment_examples_by_category.html">here</a> — start there. Each panel below: real long-axis cine frames when the source ships them (CMRx23/24 only; no scanner geometry, so they cannot be overlaid — visual reference only) · coronal and sagittal reslices through the LV centre where every native SAX slice is stacked as an equal-height band, back to back, with <b>no gap and no interpolation</b> (each band is a flat unsmoothed fill of that slice's own pixels — this trades true physical spacing for legibility) · yellow ticks = epicardial-mask extent on that cut (a straight column = aligned, a jump = shifted) · red arrow = slice with consensus spike &gt;5 mm · right column: per-slice spike and step-to-next-slice in mm (nan = not measurable; † = segmentation-only, registration beyond its search window). Sorted by severity within source. {len(missing)} subjects failed to render.</p>
{"".join(parts)}</body></html>"""
open(f"{ROOT}/_html/95b_misalignment_severe_gallery.html", "w").write(H); print("gallery", len(s), "missing", len(missing), "bytes", len(H))
# contact sheets: 3x4 thumbnails per sheet for eyeballing
os.makedirs(f"{OUT}/sheets", exist_ok=True); rels = list(s.index); per = 12
for i in range(0, len(rels), per):
    ims = [Image.open(f"{SL}/{r.split('/')[-1]}.png").convert("RGB") for r in rels[i:i + per] if os.path.exists(f"{SL}/{r.split('/')[-1]}.png")]
    if not ims: continue
    tw = 1000; ims = [im.resize((tw, int(im.height * tw / im.width))) for im in ims]; th_ = max(im.height for im in ims)
    sheet = Image.new("RGB", (2 * tw, ((len(ims) + 1) // 2) * th_), "white")
    for j, im in enumerate(ims): sheet.paste(im, ((j % 2) * tw, (j // 2) * th_))
    sheet.save(f"{OUT}/sheets/sheet_{i//per:02d}.jpg", quality=60)
print("sheets", (len(rels) + per - 1) // per)
