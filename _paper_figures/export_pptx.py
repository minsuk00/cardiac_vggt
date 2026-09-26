#!/usr/bin/env python
"""Export the paper figures to ONE editable PowerPoint deck, one slide per figure, at 1:1 physical size.

On each slide:
  - every piece of text the figure draws (tick labels, axis labels, titles, legend entries, headers, numbers)
    is a native PowerPoint text box (Arial, same size / colour / weight / rotation / position; mathtext such as
    $\\mathcal{L}_{\\mathrm{obs}}$ becomes real text with sub/superscript runs);
  - each axes panel (MRI image, arrows, curves, markers, ticks, spines) is its own transparent PNG, and each inset
    axes is a separate picture on top;
  - figure-level graphics (legend handles, header rules, colour bars) are one more transparent PNG.

Each renderer runs unchanged (commands = README.md). savefig is patched: right after the renderer writes its
.pdf, the figure is decomposed from that exact state (tick locators frozen to the PDF draw, layout frozen).

python-pptx is not in the svr env; install it anywhere and put it on PYTHONPATH:
    pip install --target /tmp/pylib python-pptx
    PYTHONPATH=/tmp/pylib:training:. micromamba run -n svr python _paper_figures/export_pptx.py
    (writes _paper_figures/paper_figures.pptx; scratch in temp/pptx_work)
"""
import argparse
import os
import re
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = {   # Overleaf name -> (renderer, argv with {o} = output stem); the same commands as README.md, paper order
    "reference_conditioning": ("render_motion_edes.py", [
        "--dump", "scratch/motion_edes/sweep80_save", "--af", "cmrx2024/CMRx24_Train_P007,6,CMRx24_Train_P007_z6f9",
        "--hrv", "mnms/MNMs_E3L8U8,7,MNMs_E3L8U8_z7f0", "--no-map", "--arrow-color", "#F39078", "--crop-window-af",
        "--gamma-af", "0.9", "--out", "{o}.pdf"]),
    "runtime_vtc_af_hrv": ("render_vtc_2x2.py", ["--out", "{o}_dir"]),
    "ablation_motion": ("render_ablation_motion.py", [
        "--subject", "cmrx2023_af12/CMRx23_Train_P093", "--tight", "--flip-v", "--no-map", "--arrow-color", "#F39078",
        "--crop-margin", "-12", "--label-color", "black", "--label-box", "--out", "{o}.pdf"]),
    "real_rt": ("make_rt_figure.py", [
        "--root", "scratch/temp/miitt_afib_rt", "--row", "MIITT_Volunteer1:20:volunteer1:Healthy volunteer:0",
        "MIITT_Patient_2024Jan04_Cardiomyopathy_AFib:170:afib:AF patient:90", "--gamma", "1.3", "--out", "{o}"]),
}
DPI = 600                     # raster resolution of the graphic layers
BASE_FROM_TOP = 1.0           # first-line baseline below a top-anchored text box's top, in font sizes (Arial;
                              # measured in LibreOffice at 7/12.6/30 pt: 1.003/1.000/1.000; PowerPoint unverified)
SYM = {r"\mathcal{L}": "\u2112", r"\dagger": "\u2020", r"\downarrow": "\u2193", r"\uparrow": "\u2191",
       r"\rightarrow": "\u2192", r"\leftarrow": "\u2190", r"\times": "\u00d7", r"\pm": "\u00b1"}
_savefig, _text_draw = Figure.savefig, Text.draw
SLIDES = []                   # one dict per figure: size, layers [(png, x, y, w, h)], texts [dict]


def math_runs(s):
    """'w/o $\\mathcal{L}_{\\mathrm{obs}}$' -> [('w/o ', 0), ('\u2112', 0), ('obs', -1)]; 0 normal, -1 sub, +1 sup."""
    out = []
    for i, part in enumerate(re.split(r"(?<!\\)\$", s)):
        if i % 2 == 0:
            out.append((part.replace(r"\$", "$"), 0))
            continue
        for k, v in SYM.items():
            part = part.replace(k, v)
        part = re.sub(r"\\(?:mathdefault|mathrm|mathit|mathbf|text)\{([^{}]*)\}", r"\1", part)
        for tok in re.finditer(r"([_^])(\{[^{}]*\}|.)|([^_^]+)", part):
            if tok.group(3):
                out.append((tok.group(3), 0))
            else:
                out.append((tok.group(2).strip("{}"), -1 if tok.group(1) == "_" else 1))
    return [(re.sub(r"[\\{}]", "", t), b) for t, b in out if t]


def decompose(fig, kw, name, workdir):
    """Split the figure into per-axes transparent PNG layers + text records (inches from the slide's top-left)."""
    axes_all = [a for a in fig.findobj(Axes)]
    for ax in axes_all:                                       # keep the PDF draw's ticks at every later draw
        lim = (ax.get_xlim(), ax.get_ylim())
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(axis.get_majorticklocs()))
            axis.set_minor_locator(FixedLocator(axis.get_minorticklocs()))
        ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
    fig.set_layout_engine("none")
    # Every render below goes through the PDF backend, like the paper figure itself: Agg measures text slightly
    # differently, which repacks legends and moves the tight bbox (runtime/VTC: 0.7 pt). Layers are PDFs
    # rasterised by MuPDF; text boxes are recorded in PDF points during the same kind of save.
    import io
    import pymupdf
    save_kw = {k: v for k, v in kw.items() if k not in ("bbox_inches", "pad_inches")}
    pad = kw.get("pad_inches", matplotlib.rcParams["savefig.pad_inches"])
    if kw.get("bbox_inches") == "tight":                      # the exact bbox the paper PDF used
        got_tb, _gtb = [], fig.get_tightbbox
        def rec_tb(*a, **k):
            b = _gtb(*a, **k); got_tb.append(b); return b
        fig.get_tightbbox = rec_tb
        try:
            _savefig(fig, io.BytesIO(), format="pdf", bbox_inches="tight", pad_inches=pad, **save_kw)
        finally:
            del fig.get_tightbbox
        tb = got_tb[-1].padded(pad)
    else:
        tb = fig.bbox_inches
    W_in, H_in = tb.width, tb.height

    def snapshot():
        """RGBA raster (DPI) of exactly the slide area, rendered via the PDF backend + MuPDF."""
        buf = io.BytesIO()
        _savefig(fig, buf, format="pdf", bbox_inches=tb, pad_inches=0, facecolor="none", edgecolor="none",
                 **save_kw)
        pm = pymupdf.open(stream=buf.getvalue(), filetype="pdf")[0].get_pixmap(dpi=DPI, alpha=True)
        return np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, 4).copy()

    def capture():
        """Every Text drawn in a bbox-cropped PDF save, with its box in PDF points (origin bottom-left)."""
        got = {}
        def rec(self, renderer):
            out = _text_draw(self, renderer)
            if self.get_visible() and self.get_text().strip():
                s, ismath = self._preprocess_math(self.get_text())
                prop = self.get_fontproperties()
                d = renderer.get_text_width_height_descent(s.splitlines()[-1], prop, ismath)[2]
                # matplotlib's text bbox always reserves the descent of "lp" (Text._get_layout), so the baseline
                # sits max(d_text, d_lp) above the bbox bottom
                d = max(d, renderer.get_text_width_height_descent("lp", prop, False)[2])
                got[self] = (self.get_window_extent(renderer), d)   # bbox + baseline height above bottom (pt)
            return out
        Text.draw = rec
        try:
            _savefig(fig, io.BytesIO(), format="pdf", bbox_inches=tb, pad_inches=0, **save_kw)
        finally:
            Text.draw = _text_draw
        return got

    drawn = capture()
    H = tb.height * 72                                        # page height in points
    texts = []
    for t, (bb, desc) in drawn.items():
        rot = t.get_rotation() % 360
        cx, cy = (bb.x0 + bb.x1) / 2 / 72, (H - (bb.y0 + bb.y1) / 2) / 72
        w, h = bb.width / 72, bb.height / 72
        # baseline, measured along the text's own "down" direction, as an offset from the bbox centre (inches)
        if 45 < rot % 180 < 135:                              # vertical text: the box is unrotated, then turned
            w, h = h, w
        base = (h / 2 - desc / 72)                            # centre -> baseline, along "down"
        fw = t.get_fontweight()
        texts.append(dict(runs=math_runs(t.get_text()), bb=bb.bounds, base=base, cx=cx, cy=cy, w=w, h=h,
                          size=t.get_fontsize(),
                          color=to_hex(t.get_color()), rot=(360 - rot) % 360, mrot=rot,
                          bold=fw == "bold" or (isinstance(fw, (int, float)) and fw >= 600),
                          italic=t.get_fontstyle() == "italic", ha=t.get_horizontalalignment()))

    # make all text fully transparent: hiding it (set_visible) would change legend packing and move the handles;
    # alpha 0 keeps every layout extent. Annotation.set_alpha affects only the text, not its arrow.
    hidden = [(t, t.get_alpha()) for t in drawn]
    for t, _ in hidden:
        t.set_alpha(0.0)
    for f in [fig] + [s for s in fig.subfigs]:
        f.patch.set_alpha(0)

    top = [a for a in axes_all if not any(a in b.child_axes for b in axes_all)]
    from matplotlib.figure import SubFigure
    sfs = [s for s in fig.findobj(SubFigure) if s is not fig]
    fig_level = [c for f in [fig] + sfs for c in f.get_children()
                 if c is not f.patch and not isinstance(c, (Axes, SubFigure))]

    def crop(img, fname):
        """Crop a layer render to its drawn (non-transparent) pixels -> saved PNG + slide position/size in inches.
        Only one layer is visible per render, so this also catches legends/markers outside the axes box."""
        ys, xs = np.nonzero(img[..., 3])
        if len(ys) == 0:
            return None
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        Image.fromarray(img[y0:y1, x0:x1]).save(fname)
        return fname, x0 / DPI, y0 / DPI, (x1 - x0) / DPI, (y1 - y0) / DPI

    layers = []
    vis0 = {a: a.get_visible() for a in axes_all}
    fl0 = {c: c.get_visible() for c in fig_level}
    for c in fig_level:
        c.set_visible(False)
    for i, A in enumerate(top):                               # one layer per top-level axes (+ one per inset)
        if not vis0[A]:
            continue
        for a in top:
            a.set_visible(a is A)
        kids = [c for c in A.child_axes]
        for c in kids:
            c.set_visible(False)
        L = crop(snapshot(), os.path.join(workdir, f"{name}_ax{i}.png"))
        if L:
            layers.append(L)
        for j, C in enumerate(kids):
            if not vis0[C]:
                continue
            own = [c for c in A.get_children() if c is not C]
            ov = {c: c.get_visible() for c in own}
            for c in own:
                c.set_visible(False)
            C.set_visible(True)
            L = crop(snapshot(), os.path.join(workdir, f"{name}_ax{i}_inset{j}.png"))
            if L:
                layers.append(L)
            for c, v in ov.items():
                c.set_visible(v)
            C.set_visible(False)
        for c in kids:
            c.set_visible(vis0[c])
    for a in top:                                             # figure-level graphics (legend handles, rules)
        a.set_visible(False)
    for c, v in fl0.items():
        c.set_visible(v)
    L = crop(snapshot(), os.path.join(workdir, f"{name}_figure.png"))
    if L:
        layers.append(L)
    for a, v in vis0.items():
        a.set_visible(v)
    for t, v in hidden:
        t.set_alpha(v)
    again = capture()                                         # the layout must not have moved during the layer renders
    drift = max(max(abs(a - b) for a, b in zip(again[t][0].bounds, bb.bounds)) for t, (bb, _) in drawn.items())
    assert drift < 0.25, f"{name}: text moved {drift:.2f} pt between the capture and the layer renders"
    SLIDES.append(dict(name=name, W=W_in, H=H_in, layers=layers, texts=texts))


def savefig_capture(self, fname, *a, **k):
    _savefig(self, fname, *a, **k)
    if str(fname).endswith(".pdf"):
        decompose(self, k, CURRENT["name"], CURRENT["workdir"])


CURRENT = {}


def render(name, stem, workdir):
    import matplotlib.pyplot as plt
    script, argv = FIGS[name]
    plt.close("all")
    matplotlib.rcdefaults()                                   # renderers share one process: no rcParams leakage
    CURRENT.update(name=name, workdir=workdir)
    Figure.savefig = savefig_capture
    sys.argv = [script] + [s.format(o=stem) for s in argv]
    sys.path.insert(0, HERE)
    try:
        runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    finally:
        Figure.savefig = _savefig


def build_pptx(out, margin=0.25):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.util import Emu, Inches, Pt

    prs = Presentation()
    prs.slide_width = Inches(max(s["W"] for s in SLIDES) + 2 * margin)
    prs.slide_height = Inches(max(s["H"] for s in SLIDES) + 2 * margin)
    align = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}
    for s in SLIDES:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        for png, x, y, w, h in s["layers"]:
            slide.shapes.add_picture(png, Inches(margin + x), Inches(margin + y), Inches(w), Inches(h))
        for t in s["texts"]:
            # Baseline placement: a top-anchored box puts the first baseline BASE_FROM_TOP font-sizes below its
            # top edge; centring the box instead sat text ~0.13 font-size too low. Offsets are taken along the
            # text's own "down" direction, so rotated labels work the same way.
            em, slack = t["size"] / 72, 0.02                           # slack: against metric rounding
            w, h = t["w"] + slack, (BASE_FROM_TOP + 0.25) * em
            o = BASE_FROM_TOP * em - h / 2                             # box centre -> baseline, along "down"
            th = np.radians(t["mrot"])
            down = (np.sin(th), np.cos(th))                            # slide coords (y down)
            along = (np.cos(th), -np.sin(th))                          # reading direction
            # the slack goes on the side away from the alignment edge (unrotated text; rotated labels are centred)
            sh = {"left": slack / 2, "right": -slack / 2}.get(t["ha"], 0) if t["mrot"] % 180 == 0 else 0
            ccx = t["cx"] + (t["base"] - o) * down[0] + sh * along[0]
            ccy = t["cy"] + (t["base"] - o) * down[1] + sh * along[1]
            box = slide.shapes.add_textbox(Inches(margin + ccx - w / 2), Inches(margin + ccy - h / 2),
                                           Inches(w), Inches(h))
            box.rotation = t["rot"]
            tf = box.text_frame
            tf.word_wrap = False
            tf.auto_size = MSO_AUTO_SIZE.NONE
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Emu(0)
            tf.vertical_anchor = MSO_ANCHOR.TOP
            p = tf.paragraphs[0]
            p.alignment = align.get(t["ha"], PP_ALIGN.CENTER)
            for txt, base in t["runs"]:
                run = p.add_run()
                run.text = txt
                f = run.font
                f.size = Pt(t["size"])
                f.bold, f.italic = t["bold"], t["italic"]
                f.color.rgb = RGBColor.from_string(t["color"][1:].upper())
                f.name = "Cambria Math" if txt == "\u2112" else "Arial"
                if base:
                    run._r.get_or_add_rPr().set("baseline", "-25000" if base < 0 else "30000")
        slide.notes_slide.notes_text_frame.text = (
            f"{s['name']}: rendered by _paper_figures/{FIGS[s['name']][0]} (see _paper_figures/README.md). "
            "Text = native text boxes; each panel = its own PNG layer.")
    prs.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE, help="directory for paper_figures.pptx (default: _paper_figures/)")
    ap.add_argument("--work", default="temp/pptx_work", help="scratch dir for the renders and layer PNGs")
    ap.add_argument("--only", nargs="+", choices=list(FIGS), default=list(FIGS))
    a = ap.parse_args()
    work = a.work
    os.makedirs(work, exist_ok=True)
    for name in a.only:
        render(name, os.path.join(work, name), work)
    out = os.path.join(a.out, "paper_figures.pptx")
    build_pptx(out)
    print("wrote", out, "|", [(s["name"], len(s["layers"]), len(s["texts"])) for s in SLIDES])


if __name__ == "__main__":
    main()
