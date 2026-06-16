"""Build the self-contained 3D UNet refiner report → _html/09_unet_refiner.html.

How the refiner works + architecture + the two runs + bugs found & fixed + future steps +
a NOTES-FOR-FUTURE-AGENTS evaluation guide (what numbers to check after the runs finish).

Run: micromamba run -n svr python _html/build_refiner_report.py
"""
import base64
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

OUT = "/home/minsukc/vggt/_html/09_unet_refiner.html"


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_dimflow():
    """Anisotropic UNet dim-flow: D preserved, H/W pooled."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    stages = [("input\n[Vc,cov]", "2", "12", "256"), ("enc0", "16", "12", "256"),
              ("pool", "16", "12", "128"), ("enc1", "32", "12", "128"),
              ("pool", "32", "12", "64"), ("bottleneck", "64", "12", "64"),
              ("up+skip", "32", "12", "128"), ("up+skip", "16", "12", "256"),
              ("out 1×1\n(zero-init)", "1", "12", "256")]
    x = 0
    for i, (name, c, d, hw) in enumerate(stages):
        h = 0.5 + 1.6 * (int(hw) / 256)   # box height ∝ H/W
        col = "#9ecae1" if "pool" not in name and "up" not in name else "#fdd0a2"
        if "out" in name:
            col = "#a1d99b"
        ax.add_patch(patches.FancyBboxPatch((x, 1.9 - h / 2), 0.92, h, boxstyle="round,pad=0.02",
                                            fc=col, ec="#333", lw=0.7))
        ax.text(x + 0.46, 1.9, f"{name}\nC={c}", ha="center", va="center", fontsize=7.5)
        ax.text(x + 0.46, 1.9 - h / 2 - 0.18, f"D={d} H/W={hw}", ha="center", va="top", fontsize=6.5, color="#444")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 1.06, 1.9), xytext=(x + 0.92, 1.9),
                        arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))
        x += 1.16
    ax.text(x + 0.1, 1.9, "+ V_canon\n(residual)", ha="left", va="center", fontsize=8, color="#0a7d28", fontweight="bold")
    ax.set_xlim(-0.2, x + 1.4); ax.set_ylim(0, 3.4); ax.axis("off")
    ax.set_title("Anisotropic 3D UNet — D=12 preserved through every pool; only H/W downsample (256→128→64)",
                 fontsize=10)
    return fig_b64(fig)


def main():
    f_dim = fig_dimflow()
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>VGGT-MRI: 3D UNet refiner (v2)</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1040px;margin:0 auto;padding:32px 24px;color:#1a1a1a;line-height:1.55}}
h1{{font-size:25px;margin-bottom:4px}} h2{{margin-top:32px;border-bottom:2px solid #eee;padding-bottom:6px;font-size:20px}}
h3{{margin-top:20px;font-size:16px}}
.sub{{color:#666;margin-top:0}}
code,.mono{{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:#f3f4f6;padding:1px 5px;border-radius:4px}}
img{{max-width:100%;border:1px solid #e2e2e2;border-radius:6px;margin:10px 0}}
.callout{{background:#f0f6ff;border-left:4px solid #1f77b4;padding:12px 16px;margin:16px 0;border-radius:0 6px 6px 0}}
.callout.key{{background:#eafaf0;border-color:#0a7d28}}
.callout.warn{{background:#fff8e6;border-color:#d9a300}}
.note{{font-size:13px;color:#666}}
table{{border-collapse:collapse;margin:14px 0;font-size:13px;width:100%}}
th,td{{border:1px solid #e2e2e2;padding:5px 8px;text-align:left;vertical-align:top}}
th{{background:#f7f7f7}} td.num{{text-align:right;font-variant-numeric:tabular-nums}}
pre{{background:#f7f7f9;border:1px solid #e2e2e2;border-radius:6px;padding:10px 12px;font-size:12.5px;overflow-x:auto}}
ul{{margin-top:6px}} li{{margin:5px 0}}
</style></head><body>

<h1>3D UNet refiner on the splat (v2)</h1>
<p class="sub">Optional, default-OFF refiner that deblurs the splatted reconstruction
<span class="mono">V_canon → V_refined</span>. Implemented + verified 2026-06-16; runs prepared, not
yet launched. Companion to <span class="mono">08_breathing_failure_mode.html</span> (the why) and
<span class="mono">docs/version_history/v2_unet_refiner.md</span> (the full record).</p>

<div class="callout key"><b>TL;DR.</b><ul>
<li><b>Why:</b> report 08 proved ~75% of the breathing reconstruction blur is the <i>splat renderer
itself</i> — the model fixes <i>where</i> anatomy goes but can't add detail the coverage-averaging
discards. A small 3D UNet can deblur the splat output.</li>
<li><b>What:</b> a 0.35M-param <b>anisotropic residual 3D UNet</b> runs inside <span class="mono">VGGT.forward</span>;
the loss adds a <b>deep-supervised second term</b> <span class="mono">L = |V_canon−V_gt| + λ·|V_refined−V_gt|</span>
(λ=1). L_pre keeps the point head honest; L_post trains the refiner. Input is
<span class="mono">[V_canon, coverage]</span>.</li>
<li><b>Safety:</b> default OFF ⇒ the pipeline is <b>bitwise identical</b> to before (verified). All refiner
logging is <b>additive</b> (new <span class="mono">refiner_viz/</span> + <span class="mono">*_refined</span>
panels; nothing new when off).</li>
<li><b>Status:</b> 163 tests green; OFF / ON-joint / ON-frozen smoke all pass; reviewed by 5 agents (two
rounds). Two real bugs were found &amp; fixed (see §6).</li>
</ul></div>

<h2>1. How it works</h2>
<pre>images + scanner_coords → Aggregator → Point head → world_points = scanner_coords + Δ
   splat_predictions(world_points, images, grid) → V_canon, coverage   [now INSIDE VGGT.forward]
   VolumeRefiner([V_canon, coverage]) → V_refined = V_canon + Δ_refine
   loss:  L_pre = |V_canon − V_gt|     (keeps the point head's geometry supervised)
          L_post = λ·|V_refined − V_gt| (trains the refiner)
          objective += (L_pre + L_post)·volume.weight</pre>
<p>The splat moved from the loss into <span class="mono">VGGT.forward</span> (gated on
<span class="mono">enable_refiner</span>) so the refiner's params are used inside the
<b>DDP-wrapped forward</b> — required for correct gradient sync. The OFF path still splats in the loss
via the <i>same</i> <span class="mono">splat_predictions</span> helper, so <span class="mono">V_canon</span>
is byte-identical either way.</p>

<h2>2. Architecture — anisotropic residual 3D UNet</h2>
<img src="data:image/png;base64,{f_dim}">
<ul>
<li><b>DoubleConv</b> = (Conv3d 3×3×3 → GroupNorm → GELU)×2 — wolny/pytorch-3dunet style. GroupNorm (not
BatchNorm) because the volume batch is <b>B=1</b>.</li>
<li><b>Anisotropic pooling</b> <span class="mono">MaxPool3d((1,2,2))</span> — pools H/W only, <b>D=12 is
preserved</b> (Z is 8&nbsp;mm / 12 planes — too coarse to downsample). The standard references are fully
isotropic; this is the key adaptation.</li>
<li><b>Residual + zero-init output conv</b> ⇒ <span class="mono">V_refined = V_canon</span> at init (starts
as the identity; won't disrupt a good V_canon early).</li>
<li><b>fp32</b> (runs under <span class="mono">autocast(enabled=False)</span>), input
<span class="mono">[V_canon, coverage]</span>. ~0.35M params.</li>
</ul>

<h2>3. Logging (additive — only when ON)</h2>
<p>New wandb keys, all in new namespaces; <b>no existing panel is replaced</b>:
<span class="mono">val_psnr_bbox_refined/*</span>, <span class="mono">val_motion_refined/*</span> (same
identity baselines as the V_canon panels → directly comparable), <span class="mono">Train_Loss/*_refined</span>,
<span class="mono">refiner_viz/{{Train,Val_Visuals_subj0,7}}_Volume</span> (per-z V_gt/V_canon/V_refined/diff),
<span class="mono">refiner_viz/cardiac_cycle_gif</span> (V_gt | V_refined), <span class="mono">Grad/refiner</span>.
When OFF, none of these appear (verified: an OFF smoke has zero refiner mentions).</p>

<h2>4. The two prepared runs</h2>
<table><tr><th>run</th><th>seed</th><th>trainable</th><th>purpose</th></tr>
<tr><td><b>A frozen</b><br><span class="mono">train_refiner_frozen.sh</span></td><td>t59w6nqy weights-only<br>(resp, z, no-t, aggft)</td><td><b>only refiner</b></td><td>isolate the pure splat-deblur gain (geometry fixed)</td></tr>
<tr><td><b>B joint</b><br><span class="mono">train_refiner_joint.sh</span></td><td>VGGT-1B base</td><td>aggregator+point_head+refiner</td><td>let geometry co-adapt with the refiner</td></tr>
</table>
<p class="note">Both: breathing ON, z+target_t on, no input-t, λ=1, find_unused_parameters=true, 200 epochs. Prepared — not launched.</p>

<h2>5. Verification</h2>
<ul>
<li>163/163 unit tests green (incl. OFF bitwise, splat==helper byte-identical, residual-identity-at-init,
two-term λ scaling, freeze isolation for both modes).</li>
<li>Smoke (real data, A40): <b>OFF</b> byte-identical (Grad/aggregator+Grad/point, 0 refiner mentions);
<b>ON-joint</b> trains (Grad/refiner≈0.41, two-term objective, refined panels log); <b>ON-frozen</b> trains
only the refiner. All RC=0, no DDP errors.</li>
<li>5 review subagents across two rounds — the 2nd round caught bug #2 below (since fixed).</li>
</ul>

<h2>6. Problems found &amp; how they were fixed</h2>
<div class="callout warn"><b>#1 — Gradient-clip crash (caught by the ON smoke, missed by static review).</b>
<span class="mono">GradientClipper.setup_clipping</span> requires every trainable param to match a
configured <span class="mono">module_name</span>; the new <span class="mono">refiner.*</span> params
matched none → <span class="mono">ValueError: Some parameters are not configured for gradient clipping</span>.
<b>Fix:</b> added a <span class="mono">refiner</span> group to <span class="mono">optim.gradient_clip.configs</span>.
Empty when OFF ⇒ skipped ⇒ no effect on the OFF path. <i>Lesson: a runtime smoke is essential — static
review can't see config/runtime interactions.</i></div>
<div class="callout warn"><b>#2 — OFF console regression (caught by an adversarial review agent).</b> My first
fix created <span class="mono">Grad/</span> meters only for clip groups with trainable params — but the
<b>aggregator</b> group is <i>also</i> always-empty (fully frozen) yet its
<span class="mono">Grad/aggregator: 0.0000</span> meter was historically displayed. The broad guard
silently dropped that column. <b>Fix:</b> narrowed the guard to skip <i>only</i> the refiner group when
empty → OFF console byte-identical again.</div>
<div class="callout warn"><b>#3 — Spurious OFF metrics (caught early).</b> Putting the refiner scalars in the
val <span class="mono">scalar_keys_to_log</span> allowlist made unupdated meters log
<span class="mono">0.0</span> when OFF. <b>Fix:</b> removed them from the allowlist; refiner train scalars
are logged directly only when present, val via the per-phase <span class="mono">*_refined</span> panels.</div>

<h2>7. Notes for the next agent — how to evaluate after the runs finish</h2>
<div class="callout key"><b>The question: does the refiner deblur V_canon (recover detail) WITHOUT
hallucinating?</b></div>
<ol>
<li><b>Primary signal — refiner beats the raw splat, same run.</b> Compare
<span class="mono">val_psnr_bbox_refined/mean</span> vs <span class="mono">val_psnr_bbox/mean</span>, and
<span class="mono">val_motion_refined/mean</span> vs <span class="mono">val_motion/mean</span>.
<b>V_refined should be HIGHER than V_canon.</b> If refined ≤ V_canon, the refiner isn't helping (λ too low /
under-trained).</li>
<li><b>Reference numbers</b> (breathing val = the deployment task; from <span class="mono">docs/05</span> +
reports 07/08):
<table><tr><th>quantity (breathing val)</th><th>bbox PSNR</th><th>motion PSNR</th></tr>
<tr><td>identity floor (do-nothing)</td><td class="num">23.23</td><td class="num">16.59</td></tr>
<tr><td>seed model v2 (resp, no refiner, ep 59) = V_canon</td><td class="num">26.74</td><td class="num">19.28</td></tr>
<tr><td><b>target for val_psnr_*_refined</b></td><td class="num">&gt; ~26.7</td><td class="num">&gt; ~19.3</td></tr>
</table>
Frozen mode = pure deblur on a fixed geometry (should clear ~26.7 bbox); joint mode may lift both V_canon
and V_refined.</li>
<li><b>Sharpness — the real point.</b> Baseline (report 08): V_canon is <b>0.65× GT</b> sharpness (breathing),
0.74× (clean). Extend <span class="mono">tools/measure_sharpness.py</span> to also measure
<span class="mono">V_refined</span> (the model now returns it) — the ratio should rise toward 1.0. Higher
PSNR <i>and</i> higher sharpness = genuine deblur.</li>
<li><b>Hallucination check (critical).</b> Clean ≠ correct. Confirm the gain is on <b>held-out / val</b>
subjects (PSNR actually rises), not just crisper-looking output. Eyeball
<span class="mono">refiner_viz/Val_Visuals_subj{{0,7}}_Volume</span> (V_refined vs V_gt) for fabricated
structure — especially in the ~5% under-covered regions flagged in report 08.</li>
<li><b>Frozen vs joint.</b> Frozen isolates splat-deblur; joint lets the point head co-adapt. If joint's
<span class="mono">val_psnr_bbox</span> (V_canon) <i>also</i> improves, the deep supervision is helping
geometry too.</li>
<li><b>Sanity.</b> <span class="mono">Grad/refiner</span> finite &amp; non-zero; <span class="mono">loss_refiner</span>
falling; <span class="mono">loss_volume</span> (L_pre) NOT degrading (the point head stays supervised).</li>
</ol>
<p class="note">Pointers: <span class="mono">docs/version_history/v2_unet_refiner.md</span> (full record),
<span class="mono">_html/08_breathing_failure_mode.html</span> (why the refiner), <span class="mono">_html/07</span>
+ <span class="mono">docs/05</span> (the 5-variant results behind the t59w6nqy seed).</p>

<h2>8. Future steps</h2>
<ul>
<li><b>Add a sharpness-aware loss term</b> (gradient L1 and/or 1−SSIM) on V_refined — L1 alone is
mean-seeking and caps high-frequency recovery. Then optionally a light perceptual/adversarial term
(higher sharpness, higher hallucination risk).</li>
<li><b>Refiner vs learned decoder</b> — the refiner deblurs the splat output; a higher-ceiling alternative
replaces the splat with a learned decoder (regress V from features). Run head-to-head.</li>
<li><b>Ablate the coverage channel</b> (<span class="mono">refiner_use_coverage=false</span>) to confirm it
helps (less hallucination in under-covered regions).</li>
<li><b>Bigger / deeper refiner</b> if 0.35M underfits — but watch hallucination and the in-distribution
val ceiling.</li>
</ul>

<p class="note">Reproduce this page: <span class="mono">_html/build_refiner_report.py</span>. Code:
<span class="mono">vggt/models/refiner.py</span>, <span class="mono">vggt/models/vggt.py</span>,
<span class="mono">training/loss.py</span>, <span class="mono">training/trainer.py</span>,
<span class="mono">sbatch/train_refiner_{{frozen,joint}}.sh</span>.</p>
</body></html>"""
    with open(OUT, "w") as f:
        f.write(html)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
