"""Build the self-contained _html/15 report (images base64-embedded)."""
import base64, os

ROOT = "/home/minsukc/vggt"
OUT = os.path.join(ROOT, "_html/15_mnms_segmentation_eval.html")


def img(path, w="100%"):
    p = os.path.join(ROOT, path)
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f'<img style="max-width:{w}" src="data:image/png;base64,{b}">'


HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>M&Ms nnU-Net cardiac segmentation as a recon-quality metric</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1080px;
margin:0 auto;padding:28px 30px;color:#202124;line-height:1.55;font-size:15.5px}}
h1{{font-size:25px;border-bottom:3px solid #1a73e8;padding-bottom:8px}}
h2{{font-size:20px;margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:5px}}
h3{{font-size:16.5px;margin-top:22px;color:#174ea6}}
.tldr{{background:#e8f0fe;border-left:5px solid #1a73e8;padding:14px 18px;border-radius:5px;margin:18px 0}}
.tldr b{{color:#174ea6}}
figure{{margin:18px 0;text-align:center}} img{{border:1px solid #e0e0e0;border-radius:5px}}
.cap{{font-size:13px;color:#5f6368;margin-top:5px;text-align:center}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}}
th,td{{border:1px solid #dadce0;padding:7px 10px;text-align:center}}
th{{background:#f1f3f4}} td.l,th.l{{text-align:left}}
.good{{color:#188038;font-weight:bold}} .bad{{color:#c5221f;font-weight:bold}} .mid{{color:#e8710a;font-weight:bold}}
.box{{background:#f8f9fa;border:1px solid #e0e0e0;border-radius:6px;padding:12px 16px;margin:14px 0}}
.win{{background:#e6f4ea;border-left:5px solid #188038}}
.warn{{background:#fef7e0;border-left:5px solid #f9ab00}}
code{{background:#f1f3f4;padding:1px 5px;border-radius:3px;font-size:13.5px}}
small{{color:#5f6368}}
</style></head><body>

<h1>A pretrained cardiac segmenter as an anatomical quality metric for our SAX reconstructions</h1>
<p><small>2026-06-20 · in-session · M&Ms challenge-winner nnU-Net (Zenodo <code>Task114_heart_mnms</code>)
run on the ACDC test set (vs human GT) and on our VGGT reconstruction output.
Repro: <code>tools/nnunet_mnms_eval/</code> · design record <code>docs/15_mnms_nnunet_segmentation_eval.md</code>.</small></p>

<div class="tldr">
<b>TL;DR.</b> We added a <b>pretrained cardiac segmenter</b> (the 2020 M&Ms challenge winner) as an
<b>anatomical</b> quality metric on top of PSNR — it labels LV / myocardium / RV on any short-axis
volume. Two things were verified:
<ul>
<li><b>It's genuinely good.</b> On the <b>ACDC test set</b> (50 patients, 100 ED+ES frames, human GT) it
scores <b class="good">mean Dice 0.902</b> (LV 0.93 / MYO 0.88 / RV 0.90) — and this is a
<b>cross-dataset</b> test (trained on M&Ms, never saw ACDC), on par with in-domain ACDC-trained
models. Robust across all 5 pathologies.</li>
<li><b>It's a sensitive recon metric.</b> Run on our current best model's reconstruction (joint refiner,
ep~88), the predicted volume's segmentation agrees with the GT volume's segmentation at
<b>Dice LV 0.93 / MYO 0.81 / RV 0.88 at ED</b>, dropping to <b>0.85 / 0.74 / 0.83 across all 12
cardiac phases</b> — the metric correctly penalizes systole (hard motion) and splat blur.</li>
</ul>
<b>Engineering:</b> it's a <b>nnU-Net v1</b> model (the lab's <code>nnunetv2</code> env cannot load it), installed
into a <b>fresh isolated env</b> — the training env <code>svr</code> was left <b>bit-identical</b>. Model + data on GPFS.
</div>

<h2>1. What the model is &amp; where it's from</h2>
<p>It is the winning entry of the <b>M&amp;Ms challenge</b> — "Multi-Centre, Multi-Vendor &amp;
Multi-Disease Cardiac Image Segmentation" (MICCAI/STACOM 2020) — by <b>Full, Isensee, Jäger &amp;
Maier-Hein</b> (DKFZ). Published as Full et al., <i>Studying Robustness of Semantic Segmentation Under
Domain Shift in Cardiac MRI</i>, STACOM 2020 (doi:10.1007/978-3-030-68107-4_24); weights on Zenodo
(<code>Task114_heart_mnms</code>, record 4134721, 2.3&nbsp;GB, md5 <code>e6613f33…</code> — verified on download).</p>
<ul>
<li><b>Task:</b> segment a short-axis (SAX) cine-MRI volume into <b>1 = LV blood pool, 2 = LV
myocardium, 3 = RV blood pool</b> (background 0). Exactly the anatomy of our SAX-stack output.</li>
<li><b>Architecture:</b> an <b>nnU-Net</b> ensemble — <b>five 2D</b> + <b>five 3D-fullres</b> folds
(trainer <code>nnUNetTrainerV2_MMS</code>). nnU-Net (Isensee 2021, <i>Nat. Methods</i>) is a
self-configuring U-Net: it derives patch size, spacing, normalization and architecture from the
dataset fingerprint, and is the long-standing state of the art for cardiac SAX segmentation.</li>
<li><b>Why M&amp;Ms specifically:</b> it was trained <b>multi-vendor / multi-centre / multi-disease</b>
on purpose for robustness to domain shift — ideal for a model we will point at out-of-distribution
reconstructions.</li>
</ul>

<h3>How it runs on our data (and the gotchas)</h3>
<p>nnU-Net reads each NIfTI with SimpleITK, <b>resamples to its trained spacing</b> using the header,
and <b>z-scores intensities internally</b> — so our percentile-normalized [-1,1] volumes pass through
fine. Two things had to be right:</p>
<ul>
<li><b>Geometry.</b> Our <code>val_volumes</code> are saved in splat order <code>(Z,Y,X)</code> with an
identity affine. We transpose to nibabel <code>(X,Y,Z)</code> and write the true canonical spacing
<code>(1.4, 1.4, 12.0)</code> mm before inference (else nnU-Net resamples wrong).</li>
<li><b>Label convention (critical).</b> The model emits <code>1=LV, 2=MYO, 3=RV</code>; <b>ACDC GT is
<code>1=RV, 2=MYO, 3=LV</code></b>. We remap per structure when scoring against ACDC. Forget this and
LV/RV Dice collapse to ~0.</li>
</ul>

<div class="box"><b>Isolation — the training env was protected.</b> The M&amp;Ms model is
<b>nnU-Net v1</b> (commands <code>nnUNet_predict -t 114 -tr nnUNetTrainerV2_MMS</code>); the lab's
<code>koalai</code> env has the incompatible <b>nnunetv2</b> and could not load it. We built a fresh
<code>nnunet</code> micromamba env (<b>nnunet 1.7.1 + torch 2.3.1+cu121</b>) and never ran <code>pip</code>
in <code>svr</code>. Verified before &amp; after: <code>svr</code> = torch 2.3.1+cu121 / monai 1.4.0 / numpy 1.26.4,
<b>unchanged</b>. Model + predictions live on GPFS (<code>scratch/data/nnunet_mnms/</code>).</div>

<h2>2. Is it actually good? — validation on ACDC vs human ground truth</h2>
<p>ACDC ships expert segmentations at ED and ES. We ran the model on the <b>official 50-patient test
split</b> (100 ED+ES frames) and Dice'd against GT. This is a true <b>cross-dataset generalization</b>
test (M&amp;Ms→ACDC), not home turf.</p>

<table>
<tr><th class="l">mode</th><th>LV (ED / ES / all)</th><th>MYO</th><th>RV</th><th>mean</th></tr>
<tr><td class="l">2D (5 folds)</td><td>0.949 / 0.892 / 0.921</td><td>0.868</td><td>0.899</td><td>0.896</td></tr>
<tr><td class="l">3D-fullres (5 folds)</td><td>0.947 / 0.901 / 0.924</td><td>0.869</td><td>0.888</td><td>0.893</td></tr>
<tr><td class="l"><b>ensemble (2D+3D)</b></td><td><b>0.950 / 0.906 / 0.928</b></td><td><b>0.875</b></td><td><b>0.903</b></td><td><b class="good">0.902</b></td></tr>
</table>
<p>For reference, <b>in-domain</b> ACDC-trained nnU-Net scores ~0.93 / 0.89 / 0.90. A M&amp;Ms model
reaching <b>0.93 / 0.88 / 0.90 with zero ACDC training</b> confirms the segmenter is trustworthy. The
weakest cell is <b>ES-LV (0.906)</b> — the small, contracted end-systolic cavity, as expected.</p>

<figure>{img("result/acdc_dice_summary.png")}
<div class="cap"><b>Left:</b> ACDC Dice vs human GT by mode and structure (ensemble best everywhere).
<b>Right:</b> ensemble Dice by pathology — NOR / MINF / DCM / HCM / abnormal-RV — all 0.89–0.91
(robust; HCM LV lowest at 0.887 because the hypertrophic cavity is small).</div></figure>

<figure>{img("result/acdc_mnms_pred_vs_gt.png", w="62%")}
<div class="cap">Model prediction (left) vs human GT (right) on three ACDC cases (colors unified:
red=LV, yellow=MYO, cyan=RV). Close agreement across ED and ES.</div></figure>

<h2>3. Applying it to our reconstruction (CMRxRecon)</h2>
<p>Now the intended use: a quality signal on our own output. We segmented <b>both</b> the GT volume and
the VGGT-<b>predicted</b> volume from our <b>current best model</b> (the live joint-refiner run
<code>218349151_mri_refiner_joint</code>, epoch ~88; predicted volume = <code>V_canon</code>, the splatted
reconstruction) and measured <b>Dice( seg(pred), seg(GT) )</b> — "does our reconstruction yield the
same heart the GT does?". 30 val subjects, each at a different cardiac phase (stratified t0…t11).</p>

<table>
<tr><th class="l">subset</th><th>n</th><th>LV</th><th>MYO</th><th>RV</th></tr>
<tr><td class="l">ED (t00) only</td><td>3</td><td class="good">0.93</td><td>0.81</td><td>0.88</td></tr>
<tr><td class="l"><b>all 12 phases (ensemble)</b></td><td>30</td><td>0.848</td><td class="mid">0.739</td><td>0.828</td></tr>
</table>
<p>The <b>ED number reproduces an independent earlier run</b> (the ED-only <code>dynamic_axial</code> model:
Dice 0.94 / 0.81 / 0.93, n=6) — consistent. Across <b>all phases</b> it drops by design: this set
includes systole (the hard motion regime) and <code>V_canon</code> is the <b>blurry splat</b> (≈0.77× GT
sharpness, per <code>docs/10</code>). So the metric is <b>phase-resolved and recon-sensitive</b> — exactly
what makes it useful. <b>MYO</b> (thin walls) and <b>RV</b> (large, deforms most) degrade first, matching
the project's known failure modes.</p>

<figure>{img("result/cmrx_joint_seg_panel.png", w="64%")}
<div class="cap">M&amp;Ms nnU-Net on our reconstruction. <b>Left:</b> GT volume + its segmentation.
<b>Right:</b> our reconstruction <code>V_canon</code> + its segmentation. The recon is visibly blurrier
(the splat renderer), yet the segmenter still recovers coherent anatomy; Dice tracks recon quality —
clean (t02) 0.93/0.85/0.84 → mid-systole (t11) 0.90/0.68/0.87 → hard (t06) 0.78/0.74/0.60.</div></figure>

<div class="warn box"><b>What this metric is (and isn't).</b> For our reconstructions there is no human
GT, so this is <b>seg(pred) vs seg(GT-volume)</b> — an <b>anatomy-preservation / internal-consistency</b>
measure, not Dice against a radiologist. That's the right tool for a <i>relative</i> recon-quality
signal (and the ACDC result above licenses trusting the segmenter itself). It also inherits the
segmenter's own ~0.1 Dice error, so treat it as a comparative metric across models/phases, not an
absolute anatomical truth.</div>

<h2>4. 2D vs 3D vs ensemble</h2>
<p>All three modes ran end-to-end on both datasets. On ACDC they are within ~0.01 Dice of each other;
the <b>2D+3D ensemble is best on every structure</b> (mean 0.902 vs 0.896 / 0.893). 2D edges 3D on RV
(thin through-plane), 3D edges 2D on ES-LV (3D context helps the small cavity). <b>Recommendation:</b>
use the <b>ensemble</b> for headline numbers (it is the published M&amp;Ms model and strictly best
here); 3D-fullres alone is a fine ~2× cheaper proxy when iterating.</p>

<h2>5. Reproduce</h2>
<div class="box"><pre style="margin:0;white-space:pre-wrap;font-size:13px">
# isolated env (svr untouched): nnunet 1.7.1 + torch 2.3.1+cu121
micromamba run -n nnunet bash -c 'source tools/nnunet_mnms_eval/env.sh && \\
  nnUNet_predict -i IN -o OUT -t 114 -m 3d_fullres -tr nnUNetTrainerV2_MMS'   # -m 2d for 2D; --save_npz + nnUNet_ensemble for ensemble

# ACDC vs human GT
python tools/nnunet_mnms_eval/prep_acdc.py  --acdc_split scratch/data/ACDC/testing --img_dir .../inputs --gt_dir .../gt
python tools/nnunet_mnms_eval/eval_acdc.py  --pred_dir .../seg_ensemble --gt_dir .../gt    # remaps ACDC<->M&Ms labels

# our reconstruction (anatomy preservation)
python tools/nnunet_mnms_eval/prep_inputs.py  --val_dir &lt;log&gt;/val_volumes --out_dir .../inputs
python tools/nnunet_mnms_eval/analyze_segs.py --seg_dir .../seg_ensemble --input_dir .../inputs
</pre></div>
<p><small>Model + data: <code>scratch/data/nnunet_mnms/</code> (GPFS). ACDC: <code>scratch/data/ACDC</code>.
Figures: <code>result/acdc_dice_summary.png</code>, <code>result/acdc_mnms_pred_vs_gt.png</code>,
<code>result/cmrx_joint_seg_panel.png</code>. Design record: <code>docs/15_mnms_nnunet_segmentation_eval.md</code>.</small></p>

</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(HTML)
print("wrote", OUT, f"({len(HTML)//1024} KB source, images embedded)")
