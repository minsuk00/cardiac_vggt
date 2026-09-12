"""Focused gallery: 4 examples spanning low->high severity within each category (clean/mild/moderate/
severe), so the boundary between bands reads as a gradient rather than one arbitrary point, plus the
subjects the v3.1 rule fixes moved UP the most (the ACDC_patient081 class of miss). Uses the no-gap
equal-height slab reslice. Examples are chosen automatically from subjects_v3.csv (Z>=8, measurable) at
fixed quantiles of each band's own severity range, so they track the data after a re-sweep.
Usage: python build_categorized_examples.py"""
import base64, html, os
import numpy as np, pandas as pd
ROOT = "/home/minsukc/vggt"; OUT = f"{ROOT}/temp/misalign_v2"; SL = f"{OUT}/slabs"
sub = pd.read_csv(f"{OUT}/subjects_v3.csv", index_col=0)
S = sub[sub.measurable & (sub.Z >= 8)]

BANDS = [("clean (≤3 mm)", 0, 3), ("mild (3–5 mm) — keep", 3, 5), ("moderate (5–8 mm)", 5, 8), ("severe (>8 mm)", 8, 999)]
QUANTS = (0.05, 0.35, 0.65, 0.95)   # within-band severity quantiles -> near low edge ... near high edge

def pick(lo, hi):
    g = S[(S.severity > lo) & (S.severity <= hi)].sort_values("severity")
    out = []
    for q in QUANTS:
        target = g.severity.quantile(q); cand = g[~g.index.isin(out)]
        # prefer a source not already used in this band, then nearest severity
        used = {sub.loc[r].source for r in out}
        cand = cand.assign(d=(cand.severity - target).abs() + np.where(cand.source.isin(used), 0.4, 0.0))
        out.append(cand.d.idxmin())
    return out

PICKS = {name: pick(lo, hi) for name, lo, hi in BANDS}
# subjects the v3.1 fixes moved up the most (previous band -> new band); band_changes_vs_v3.csv is written by analyze.py
chg_p = f"{OUT}/band_changes_vs_v3.csv"
moved = []
if os.path.exists(chg_p):
    chg = pd.read_csv(chg_p, index_col=0)
    order = {"unmeasurable": -1, "clean": 0, "mild": 1, "moderate": 2, "severe": 3}
    chg = chg[[order[b] > order[p] for b, p in zip(chg.band, chg.band_prev)]]
    chg = chg.assign(jump=chg.severity - chg.sev_prev.fillna(0)).sort_values("jump", ascending=False)
    moved = list(chg.index[:6])
    if "ACDC_sax/ACDC_patient081" in chg.index and "ACDC_sax/ACDC_patient081" not in moved: moved = ["ACDC_sax/ACDC_patient081"] + moved[:5]

def b64(rel):
    p = f"{SL}/{rel.split('/')[-1]}.png"
    return base64.b64encode(open(p, "rb").read()).decode()

def card(rel, extra=""):
    r = sub.loc[rel]
    fb = " · <b>†</b> seg-only value used (registration beyond its search window)" if (r.n_spike_fallback > 0 or r.n_step_fallback > 0) else ""
    return (f'<div class="ex"><div class="cap"><b>{html.escape(str(r.source))}</b> · {html.escape(rel)} · severity <b>{r.severity:.1f} mm</b> '
            f'(spike {r.max_spike:.1f}, step {r.max_step if pd.notna(r.max_step) else float("nan"):.1f}) · {html.escape(str(r.centre_key))}{fb}{extra}</div>'
            f'<img src="data:image/png;base64,{b64(rel)}"></div>')

parts = []
for band, items in PICKS.items():
    parts.append(f'<h2>{html.escape(band)}</h2><p class="note">4 subjects spanning this band\'s own severity range, low to high — not just its median.</p>')
    parts += [card(rel) for rel in items]
if moved:
    parts.append('<h2>Reclassified by the v3.1 fixes</h2><p class="note">The subjects that moved up the most after (a) widening the registration search window 20→40 mm and treating a registration pinned at the bound as "offset beyond the window" rather than "no answer", and (b) keeping min(seg, img) for a step whose two estimates disagree in direction but are both large. These were previously sitting in a lower band; the first was the by-eye catch that exposed the bug.</p>')
    for rel in moved:
        r = chg.loc[rel]
        parts.append(card(rel, f' · <span style="color:#c0392b">was {r.sev_prev:.1f} mm ({r.band_prev}) → now {r.band}</span>'))

H = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Misalignment examples by category</title>
<style>body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1500px;margin:20px auto;padding:0 16px;color:#222}}
.ex{{margin:10px 0 18px}}.cap{{font-size:13px;margin-bottom:3px}}img{{max-width:100%;border:1px solid #ddd}}
h2{{border-bottom:2px solid #ccc;margin-top:36px;padding-bottom:4px}}.note{{font-size:12px;color:#666;margin:2px 0 10px}}
.tldr{{background:#eef6ee;border-left:5px solid #4c9f70;padding:10px 16px;margin-bottom:20px}}.nav{{font-size:13px;margin-bottom:16px}}</style></head><body>
<div class="nav">Full galleries: <a href="95d_gallery_clean.html">clean</a> · <a href="95d_gallery_mild.html">mild</a> · <a href="95d_gallery_moderate.html">moderate</a> · <a href="95d_gallery_severe.html">severe</a> · main report: <a href="95_slice_misalignment_reanalysis.html">95</a></div>
<h1>Inter-slice misalignment — examples spanning each severity category</h1>
<div class="tldr">Companion to <code>_html/95_slice_misalignment_reanalysis.html</code> and the per-band galleries
<code>_html/95d_gallery_*.html</code>. Each band below shows 4 subjects spread across that
band's <i>own</i> severity range (e.g. moderate runs 5→8&nbsp;mm; the first example here is near 5, the last
near 8) so the transition between bands is visible as a gradient, not a single snapshot. Panels: every native
SAX slice stacked as an equal-height band, back to back, with <b>no gap and no interpolation</b> — each band is
a flat, unsmoothed fill of that slice's own pixels (trades true physical spacing for legibility). Left = coronal
cut, right = sagittal cut — two perpendicular side-views through the same LV centre, so a shift in either
direction is visible in at least one panel. Yellow ticks = left/right extent of the epicardial mask on that cut
(a straight column = aligned; a jump = shifted). Red ▶ marks a slice whose measured offset (segmentation and
image-registration methods agree) exceeds 5&nbsp;mm. In the number column, † = value from the segmentation alone
because the registration was pinned at its 40&nbsp;mm search bound. Left column (CMRx23/24 only): real long-axis cine frames
for anatomical reference — the files carry no scanner geometry, so they cannot be aligned to the SAX stack.</div>
{"".join(parts)}
</body></html>"""
open(f"{ROOT}/_html/95c_misalignment_examples_by_category.html", "w").write(H)
print("picks", {k: [r.split("/")[-1] for r in v] for k, v in PICKS.items()}, "moved", [r.split("/")[-1] for r in moved], "bytes", len(H))
