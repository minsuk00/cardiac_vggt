"""Cohort analysis of metric_v2 output (read-only). Produces, under temp/misalign_v2/:
  slices_v3.csv, steps_v3.csv, subjects_v3.csv, summary.json, curated_*.txt, figs/F*.png
Every number quoted in docs/95 and _html/95 comes from summary.json written here.

Two per-subject severity measures, each requiring the segmentation and the image estimator to agree:
  spike : |c_z - (c_{z-1}+c_{z+1})/2|      -> single-slice slips (halved on block edges)  [was "kink"]
  step  : |(c_{z+1}-c_z) - median_z(c_{z+1}-c_z)| -> discontinuity between adjacent slices (block edges at
          full magnitude; also fires on both pairs around a single slip)
  severity = max over measurable slices/pairs of max(spike, step).
Consensus rules (v3.1, 2026-09-11):
  * value = min(seg, img) when both estimators answer;
  * registration pinned at the search bound (img_pin_* >= PIN_MIN) is "offset >= MAXSHIFT_MM", not "no answer":
    the segmentation estimate is used as-is (seg_fallback) -- a 28 mm block shift in ACDC_patient081 was
    previously discarded this way and the subject scored 2.4 mm "clean";
  * a flat/featureless registration (img_flat_*) or low NCC is still "no answer" -> the slice/pair is not scored;
  * step direction check: if the two vectors disagree in direction but BOTH magnitudes are >= BIG_MM the pair
    keeps the conservative min(seg, img) (previously zeroed: CMRx25 Center001 P003 z1, seg 20.2 / img 15.2 -> 0.0);
    a direction disagreement with one small magnitude is still treated as the seg false-positive it usually is.
Usage: python analyze.py <full_v3.csv>
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

ROOT = "/home/minsukc/vggt"; OUT = f"{ROOT}/temp/misalign_v2"; FIG = f"{OUT}/figs"
AREA_MM2 = 400.0; NPH_MIN = 6; NCC_MIN = 0.4
PIN_MIN = 0.5      # fraction of phases pinned at the search bound to count a registration as "beyond the window"
BIG_MM = 8.0       # both-large threshold for keeping a direction-disagreeing step
THRS = (3, 4, 5, 8, 10)

def gate_spikes(df):
    """Adds spike_ok / spike / spike_src / spike_agree / spike_mean columns in place."""
    df["epi_area_mm2"] = df.epi_area_ed * df.dx * df.dy
    seg_ok = (df.epi_area_mm2 >= AREA_MM2) & df.epi_spike.notna() & (df.epi_spike_nph >= NPH_MIN)
    img_ok = df.img_spike.notna() & (df.img_ncc_prev >= NCC_MIN) & (df.img_ncc_next >= NCC_MIN)
    # pinned on either side: registration says the offset is beyond the window on that side; not "no shift"
    pinned = (df.img_pin_prev.fillna(0) >= PIN_MIN) | (df.img_pin_next.fillna(0) >= PIN_MIN)
    flat = (df.img_flat_prev.fillna(0) >= PIN_MIN) | (df.img_flat_next.fillna(0) >= PIN_MIN)
    fallback = seg_ok & ~img_ok & pinned & ~flat
    df["spike_ok"] = (seg_ok & img_ok) | fallback
    df["spike_src"] = np.where(fallback, "seg_fallback", np.where(seg_ok & img_ok, "consensus", ""))
    df["spike_agree"] = np.hypot(df.epi_spike_x - df.img_spike_x, df.epi_spike_y - df.img_spike_y)   # same sign convention
    df["spike"] = np.where(seg_ok & img_ok, np.minimum(df.epi_spike, df.img_spike), np.where(fallback, df.epi_spike, np.nan))
    df["spike_mean"] = np.where(seg_ok & img_ok, 0.5 * (df.epi_spike + df.img_spike), np.nan)
    return df

def compute_steps(df):
    """Per adjacent pair (z, z+1): step from phase-median centroids + the one-sided z->z+1 registration."""
    rows = []
    for rel, g in df.groupby("rel", sort=False):
        g = g.sort_values("z"); z = g.z.values
        cx = g.epi_cx_med.values; cy = g.epi_cy_med.values; area = g.epi_area_mm2.values
        okc = np.isfinite(cx) & (area >= AREA_MM2)
        seg_d = []; idx = []
        for i in range(len(g) - 1):
            if okc[i] and okc[i + 1] and z[i + 1] - z[i] == 1:
                seg_d.append((cx[i + 1] - cx[i], cy[i + 1] - cy[i])); idx.append(i)
        if len(seg_d) < 3: continue
        seg_d = np.array(seg_d); med = np.median(seg_d, axis=0)
        pin = g.img_pin_next.fillna(0).values; flat = g.img_flat_next.fillna(0).values
        for k, i in enumerate(idx):
            # img: d_next = c_z - c_{z+1}  =>  c_{z+1}-c_z = -d_next   (sign fixed in metric_v2)
            img_d = -np.array([g.img_dnext_x.values[i], g.img_dnext_y.values[i]])
            ncc = g.img_ncc_next.values[i]
            r_seg = float(np.hypot(*(seg_d[k] - med))); r_img = float(np.hypot(*(img_d - med))) if np.isfinite(img_d).all() else np.nan
            agree = float(np.hypot(*(seg_d[k] - img_d))) if np.isfinite(img_d).all() else np.nan
            both = bool(np.isfinite(r_img) and ncc >= NCC_MIN)
            fallback = (not both) and pin[i] >= PIN_MIN and flat[i] < PIN_MIN
            if both:
                lo = min(r_seg, r_img)
                dir_ok = agree <= max(2.5, 0.5 * lo)
                step = lo if (dir_ok or lo >= BIG_MM) else 0.0
                src = "consensus" if dir_ok else ("both_large" if lo >= BIG_MM else "contradict")
            elif fallback:
                step = r_seg; src = "seg_fallback"
            else:
                step = np.nan; src = ""
            rows.append(dict(rel=rel, z=int(z[i]), step_seg=r_seg, step_img=r_img, step_agree=agree, step_ok=both or fallback,
                             step=step, step_src=src, drift_mm=float(np.hypot(*med))))
    return pd.DataFrame(rows)

def subject_table(df, st):
    g = df.groupby("rel")
    sub = pd.DataFrame(dict(Z=g.Z.first(), split=g.split.first(), n_spike=g.spike_ok.sum(), max_spike=g.spike.max(),
                            max_spike_mean=g.spike_mean.max(), max_epi_spike_gated=df[df.spike_ok].groupby("rel").epi_spike.max(),
                            max_img_spike_gated=df[df.spike_ok].groupby("rel").img_spike.max(), lineres_ed_max=g.lv_lineres_ed.max(),
                            n_spike_fallback=g.spike_src.apply(lambda x: (x == "seg_fallback").sum())))
    gs = st[st.step_ok].groupby("rel")
    sub["n_step"] = gs.size().reindex(sub.index).fillna(0).astype(int); sub["max_step"] = gs.step.max().reindex(sub.index)
    sub["n_step_fallback"] = gs.step_src.apply(lambda x: (x == "seg_fallback").sum()).reindex(sub.index).fillna(0).astype(int)
    sub["n_step_both_large"] = gs.step_src.apply(lambda x: (x == "both_large").sum()).reindex(sub.index).fillna(0).astype(int)
    # weak step evidence: every scored pair was zeroed as a contradiction, so the step score carries no information
    # and severity rests on the spike consensus alone (flagged, not excluded)
    sub["weak_step_evidence"] = gs.step_src.apply(lambda x: (x == "contradict").all()).reindex(sub.index).fillna(False).astype(bool)
    sub["severity"] = np.fmax(sub.max_spike, sub.max_step)
    sub["measurable"] = (sub.n_spike >= 3) | (sub.n_step >= 3)
    return sub

def band(sev):
    return np.where(sev <= 3, "clean", np.where(sev <= 5, "mild", np.where(sev <= 8, "moderate", "severe")))

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/full_v3.csv")
    man = pd.read_csv(f"{ROOT}/training/splits/manifest.csv")
    def centre_of(r):
        if r.source == "CMRxRecon2025":
            m = re.search(r"(Center\d+)_(\w+?)_(\d+T)_(\w+?)_P", r.rel_path); return f"CMRx25/{m.group(1)}/{m.group(2)}-{m.group(4)}-{m.group(3)}"
        if r.source == "MNMs": return f"MNMs/centre{r.centre}/{r.vendor}"
        if r.source == "ACDC": return "ACDC/Dijon/Siemens"
        return f"{r.source}/Fudan/Siemens-Vida"
    man["centre_key"] = man.apply(centre_of, axis=1)
    meta = man.set_index("rel_path")[["id", "source", "centre_key", "vendor", "pathology_label", "n_z", "pitch_mm"]]

    gate_spikes(df)
    st = compute_steps(df); st.to_csv(f"{OUT}/steps_v3.csv", index=False)
    sub = subject_table(df, st).join(meta, how="left")
    sub["band"] = band(sub.severity)
    sub.to_csv(f"{OUT}/subjects_v3.csv"); df.to_csv(f"{OUT}/slices_v3.csv", index=False)
    s = sub[sub.measurable]
    S = {}
    okk = df.spike_ok; oks = st.step_ok
    S["n_subjects"] = int(len(sub)); S["n_measurable"] = int(len(s)); S["n_slice_rows"] = int(len(df)); S["n_spike_slices"] = int(okk.sum()); S["n_step_pairs"] = int(oks.sum())
    S["spike_quantiles"] = {f"p{int(q*100)}": float(v) for q, v in df.spike[okk].quantile([.5, .75, .9, .95, .99]).items()}
    S["slice_frac_spike_gt"] = {t: float((df.spike[okk] > t).mean()) for t in THRS}
    S["pair_frac_step_gt"] = {t: float((st.step[oks] > t).mean()) for t in THRS}
    S["subj_frac_spike_gt"] = {t: float((s.max_spike > t).mean()) for t in THRS}
    S["subj_frac_step_gt"] = {t: float((s.max_step > t).mean()) for t in THRS}
    S["subj_frac_sev_gt"] = {t: float((s.severity > t).mean()) for t in THRS}
    S["subj_sev_quantiles"] = {f"p{int(q*100)}": float(v) for q, v in s.severity.quantile([.25, .5, .75, .9, .95]).items()}
    S["band_counts"] = {b: int((s.band == b).sum()) for b in ("clean", "mild", "moderate", "severe")}
    cons = okk & (df.spike_src == "consensus")
    S["r_seg_img_spike"] = float(np.corrcoef(df.epi_spike[cons], df.img_spike[cons])[0, 1])
    mid = cons & (df.epi_area_mm2 > 800) & (df.img_ncc_prev > 0.6) & (df.img_ncc_next > 0.6)
    S["r_seg_img_spike_mid"] = float(np.corrcoef(df.epi_spike[mid], df.img_spike[mid])[0, 1])
    # cost of the wider window: spurious far NCC peaks on consensus slices (img >10 mm where seg <3 mm); absorbed by min(seg, img)
    S["img_spurious_far_slices"] = int(((df.img_spike > 10) & (df.epi_spike < 3))[cons].sum())
    S["r_seg_img_spike_clip15"] = float(np.corrcoef(df.epi_spike[cons].clip(0, 15), df.img_spike[cons].clip(0, 15))[0, 1])
    S["spike_agree_median_mm"] = float(df.spike_agree[cons].median()); S["step_agree_median_mm"] = float(st.step_agree[oks & (st.step_src == "consensus")].median())
    S["xphase_noise_seg_med"] = float(df.epi_spike_noise[okk].median()); S["xphase_noise_img_med"] = float(df.img_spike_noise[cons].median())
    fl = okk & (df.spike > 5)
    S["xphase_noise_seg_flagged"] = float(df.epi_spike_noise[fl].median()); S["xphase_noise_img_flagged"] = float(df.img_spike_noise[fl & cons].median())
    S["frac_flagged_noise_gt_spike"] = float((df.epi_spike_noise[fl] > df.spike[fl]).mean())
    # v3.1 bookkeeping: how often each consensus rule fired, and what the fixes changed
    S["rules"] = dict(spike_seg_fallback_slices=int((df.spike_src == "seg_fallback").sum()),
                      step_seg_fallback_pairs=int((st.step_src == "seg_fallback").sum()),
                      step_both_large_pairs=int((st.step_src == "both_large").sum()),
                      step_contradict_pairs=int((st.step_src == "contradict").sum()),
                      step_contradict_frac_of_scored=float((st.step_src == "contradict").sum() / max(1, oks.sum())),
                      # the meaningful contradictions: segmentation claims >5 mm, image (finite, NCC ok) does not back it
                      step_contradict_pairs_seg_gt5=int(((st.step_src == "contradict") & (st.step_seg > 5)).sum()),
                      step_contradict_frac_of_seg_gt5=float(((st.step_src == "contradict") & (st.step_seg > 5)).sum() / max(1, ((st.step_seg > 5) & st.step_img.notna() & oks).sum())),
                      subj_with_any_fallback=int(((s.n_spike_fallback > 0) | (s.n_step_fallback > 0)).sum()),
                      subj_weak_step_evidence=int(s.weak_step_evidence.sum()),
                      subj_weak_step_evidence_by_band={b: int((s.weak_step_evidence & (s.band == b)).sum()) for b in ("clean", "mild", "moderate", "severe")},
                      subj_with_both_large=int((s.n_step_both_large > 0).sum()))
    prev_p = f"{OUT}/_v3_superseded/subjects_v3.csv"
    if os.path.exists(prev_p):
        prev = pd.read_csv(prev_p, index_col=0)
        jj = s.join(prev[["severity", "measurable"]].rename(columns={"severity": "sev_prev", "measurable": "meas_prev"}), how="left")
        jj["band_prev"] = band(jj.sev_prev.fillna(-1)); jj.loc[jj.sev_prev.isna(), "band_prev"] = "unmeasurable"
        chg = jj[jj.band != jj.band_prev]
        order = {"unmeasurable": -1, "clean": 0, "mild": 1, "moderate": 2, "severe": 3}
        up = chg[[order[b] > order[p] for b, p in zip(chg.band, chg.band_prev)]]
        S["vs_v3"] = dict(n_band_changed=int(len(chg)), n_moved_up=int(len(up)), n_moved_down=int(len(chg) - len(up)),
                          moved_up_from_clean=int((up.band_prev == "clean").sum()),
                          moved_up_to_severe=int((up.band == "severe").sum()),
                          prev_subj_frac_sev_gt={t: float((prev[prev.measurable].severity > t).mean()) for t in THRS},
                          transitions={f"{p}->{b}": int(n) for (p, b), n in chg.groupby(["band_prev", "band"]).size().items()})
        chg.assign(sev_prev=chg.sev_prev.round(2), severity=chg.severity.round(2))[["source", "centre_key", "split", "sev_prev", "band_prev", "severity", "band", "max_spike", "max_step", "n_spike_fallback", "n_step_fallback", "n_step_both_large"]].sort_values("severity", ascending=False).to_csv(f"{OUT}/band_changes_vs_v3.csv")
        # subjects whose numbers changed at all (for re-rendering the panels)
        num_chg = jj[(jj.sev_prev.isna()) | ((jj.severity - jj.sev_prev).abs() > 0.05)]
        S["vs_v3"]["n_severity_changed"] = int(len(num_chg))
    # sensitivity (all on the same gated slices, same measurable-subject denominator)
    S["sens_subj_gt5"] = dict(consensus_min=S["subj_frac_spike_gt"][5], mean=float((s.max_spike_mean > 5).mean()),
                              seg_only=float((s.max_epi_spike_gated > 5).mean()), img_only=float((s.max_img_spike_gated > 5).mean()),
                              spike_or_step=S["subj_frac_sev_gt"][5])
    # is spike still needed alongside step? subjects whose band differs when step alone defines severity
    band_step = band(s.max_step.fillna(0)); S["spike_needed"] = dict(n_band_differs_step_only=int((band_step != s.band).sum()),
                                                                     n_subj_spike_gt_step=int((s.max_spike > s.max_step.fillna(0)).sum()))
    # halo: flagged spike slices that are neighbours of a bigger flagged slice
    halo = 0; peak = 0
    for rel, gg in df[df.spike_ok].groupby("rel"):
        gg = gg.sort_values("z"); kk = gg.spike.values; zz = gg.z.values
        for i in range(len(gg)):
            if not kk[i] > 5: continue
            nb = [kk[j] for j in (i - 1, i + 1) if 0 <= j < len(gg) and zz[j] - zz[i] in (-1, 1)]
            if nb and max(nb) > kk[i]: halo += 1
            else: peak += 1
    S["spike_slices_gt5"] = dict(total=halo + peak, local_max=peak, halo=halo)
    # interleave: longest alternating run of |spike|>2 along the dominant axis, among severity>5
    runs = {}
    for rel in s[s.severity > 5].index:
        gg = df[(df.rel == rel) & df.spike_ok].sort_values("z")
        if len(gg) < 3: continue
        a = gg.epi_spike_x.values if np.abs(gg.epi_spike_x).sum() > np.abs(gg.epi_spike_y).sum() else gg.epi_spike_y.values; zz = gg.z.values
        best = cur = 1
        for i in range(1, len(a)):
            if zz[i] - zz[i - 1] == 1 and abs(a[i]) > 2 and abs(a[i - 1]) > 2 and a[i] * a[i - 1] < 0: cur += 1
            else: cur = 1
            best = max(best, cur)
        runs[rel] = best
    runs = pd.Series(runs); S["interleave"] = dict(n_severe=int(len(runs)), frac_run_ge4=float((runs >= 4).mean()), frac_run_ge5=float((runs >= 5).mean()))
    # per source / centre
    def agg(x): return pd.Series(dict(n=len(x), median_sev=x.severity.median(), p90_sev=x.severity.quantile(.9),
                                      gt3=(x.severity > 3).mean(), gt5=(x.severity > 5).mean(), gt8=(x.severity > 8).mean(),
                                      spike_gt5=(x.max_spike > 5).mean()))
    src = s.groupby("source").apply(agg); cent = s.groupby("centre_key").apply(agg)
    src.to_csv(f"{OUT}/per_source_v3.csv"); cent.to_csv(f"{OUT}/per_centre_v3.csv")
    S["per_source"] = src.round(3).to_dict("index"); S["per_centre"] = cent.round(3).to_dict("index")
    S["centres_all_affected_gt5"] = bool((cent[cent.n >= 5].gt5 > 0).all()); S["n_centres_ge5"] = int((cent.n >= 5).sum())
    # comparison with docs/94 (like-for-like: their interior-only ED column; ours = severity)
    pp = pd.read_csv(f"{ROOT}/temp/data_curation/full_sweep_per_phase.csv"); p0 = pp[pp.phase == 0].set_index("subject")
    j = s.join(p0[["max_jump_mm", "interior_max_jump_mm"]], how="inner").dropna(subset=["interior_max_jump_mm"])
    S["prior"] = dict(n=int(len(j)), flag_rate_asreported=float((j.max_jump_mm > 4).mean()), flag_rate_interior=float((j.interior_max_jump_mm > 4).mean()),
                      ours_sev_gt4=float((j.severity > 4).mean()), ours_sev_gt3=float((j.severity > 3).mean()),
                      r_interior_vs_sev=float(np.corrcoef(j.interior_max_jump_mm, j.severity)[0, 1]),
                      r_asreported_vs_sev=float(np.corrcoef(j.max_jump_mm, j.severity)[0, 1]),
                      prior_flagged_but_sev_le3=float((j[j.interior_max_jump_mm > 4].severity <= 3).mean()),
                      prior_flagged_but_sev_le5=float((j[j.interior_max_jump_mm > 4].severity <= 5).mean()),
                      sev_gt5_but_prior_clean=float((j[j.severity > 5].interior_max_jump_mm <= 4).mean()))
    # ---------------- curated lists (pooled.txt format; ALL THREE sections filtered -- val/test GT carries the same
    # corruption, so a misaligned GT would penalise a correct reconstruction; unmeasurable subjects KEPT, listed in the header)
    pooled = [l.rstrip("\n") for l in open(f"{ROOT}/training/splits/pooled.txt")]
    def write_list(name, keep, note):
        kept = {}; dropped = {}; sec = None
        with open(f"{OUT}/{name}.txt", "w") as f:
            f.write(f"# {note}\n# Derived from training/splits/pooled.txt; train, val AND test filtered. Generated by tools/misalign_v2/analyze.py\n")
            for l in pooled:
                if l.startswith("#") or not l.strip(): continue
                if l.startswith("["): sec = l.strip("[]"); kept.setdefault(sec, 0); dropped.setdefault(sec, 0); f.write(l + "\n"); continue
                if l in keep: kept[sec] += 1; f.write(l + "\n")
                else: dropped[sec] += 1
        return kept, dropped
    # manual overrides from by-eye review (tools/misalign_v2/manual_overrides.csv): exclude always drops, include always keeps
    ov_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manual_overrides.csv")
    ov = pd.read_csv(ov_p).set_index("rel") if os.path.exists(ov_p) else pd.DataFrame(columns=["action"])
    unknown = [r for r in ov.index if r not in sub.index]
    assert not unknown, f"manual_overrides.csv names unknown subjects: {unknown}"
    man_excl = set(ov[ov.action == "exclude"].index); man_incl = set(ov[ov.action == "include"].index)
    sub["manual"] = ov.action.reindex(sub.index).fillna("")
    S["manual_overrides"] = dict(n_exclude=len(man_excl), n_include=len(man_incl),
                                 exclude_by_band={b: int(sub.loc[list(man_excl)].band.eq(b).sum()) for b in ("clean", "mild", "moderate", "severe")},
                                 include_by_band={b: int(sub.loc[list(man_incl)].band.eq(b).sum()) for b in ("clean", "mild", "moderate", "severe")})
    tiers = {}
    for thr in (5, 8, 10):
        unmeas = set(sub[~sub.measurable].index)
        keep = ((set(s[s.severity <= thr].index) | unmeas) - man_excl) | man_incl
        k, d = write_list(f"curated_sev{thr}mm", keep, f"keep subjects (train, val and test) whose worst slice/pair discontinuity (max of spike, step; seg+img consensus, v3.1 rules) <= {thr} mm, "
                          f"then manual by-eye overrides from tools/misalign_v2/manual_overrides.csv ({len(man_excl)} excluded, {len(man_incl)} force-included). "
                          f"{len(unmeas)} UNMEASURABLE subjects (too few gated slices/pairs) are KEPT unless manually excluded; they are: " + ", ".join(sorted(unmeas)))
        tiers[f"sev{thr}"] = dict(thr=thr, train_kept=k["train"], train_dropped=d["train"], val_kept=k["val"], val_dropped=d["val"], test_kept=k["test"], test_dropped=d["test"],
                                  unmeasurable_kept=int((~sub.measurable).sum()),
                                  per_centre={c: [int((sub.centre_key == c).sum()), int((sub.centre_key == c)[sub.index.isin(keep)].sum())] for c in sorted(sub.centre_key.dropna().unique())},
                                  per_centre_valtest={c: [int(((sub.centre_key == c) & (sub.split != "train")).sum()), int(((sub.centre_key == c) & (sub.split != "train"))[sub.index.isin(keep)].sum())] for c in sorted(sub.centre_key.dropna().unique())})
    S["tiers"] = tiers
    S["unmeasurable_train"] = sorted(sub[(sub.split == "train") & ~sub.measurable].index.tolist())
    S["unmeasurable_valtest"] = sorted(sub[(sub.split != "train") & ~sub.measurable].index.tolist())
    # ---------------- figures
    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=120)
    ax.scatter(df.epi_spike[cons], df.img_spike[cons], s=4, alpha=0.35); ax.plot([0, 30], [0, 30], "k--", lw=0.8); ax.set_xlim(0, 30); ax.set_ylim(0, 30)
    ax.set_xlabel("segmentation spike (epicardial centroid vs neighbours), mm"); ax.set_ylabel("image-registration spike (NCC to neighbours), mm")
    ax.set_title(f"two independent estimators, per slice (n={int(cons.sum())}, r={S['r_seg_img_spike']:.2f})", fontsize=9); fig.tight_layout(); fig.savefig(f"{FIG}/F1_seg_vs_img.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=120)
    for srcn, gg in s.groupby("source"):
        x = np.sort(gg.severity.values); ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=f"{srcn} (n={len(gg)})")
    for t in (3, 5, 8): ax.axvline(t, color="gray", lw=0.6, ls=":")
    ax.set_xlim(0, 20); ax.set_xlabel("subject severity = worst slice/pair discontinuity, mm (seg+img consensus)"); ax.set_ylabel("cumulative fraction of subjects"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/F2_subject_cdf.png"); plt.close(fig)
    c2 = s.groupby("centre_key").agg(n=("Z", "size"), clean=("severity", lambda x: (x <= 3).mean()), mild=("severity", lambda x: ((x > 3) & (x <= 5)).mean()),
                                     moderate=("severity", lambda x: ((x > 5) & (x <= 8)).mean()), severe=("severity", lambda x: (x > 8).mean())).sort_values("clean")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(c2) + 1.2), dpi=120); left = np.zeros(len(c2))
    for col, colr in (("clean", "#4c9f70"), ("mild", "#c9c95a"), ("moderate", "#e08a3c"), ("severe", "#c0392b")):
        ax.barh(range(len(c2)), c2[col], left=left, color=colr, label=col); left += c2[col].values
    ax.set_yticks(range(len(c2))); ax.set_yticklabels([f"{k} (n={n})" for k, n in zip(c2.index, c2.n)], fontsize=7)
    ax.set_xlabel("fraction of subjects (severity: clean ≤3 | mild 3–5 | moderate 5–8 | severe >8 mm)"); ax.legend(fontsize=7, loc="lower right", ncol=4)
    fig.tight_layout(); fig.savefig(f"{FIG}/F3_per_centre.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=120)
    ax.scatter(j.interior_max_jump_mm, j.severity, s=6, alpha=0.4); ax.axvline(4, color="r", ls=":", lw=0.8); ax.axhline(5, color="g", ls=":", lw=0.8)
    ax.set_xlim(0, 40); ax.set_ylim(0, 25); ax.set_xlabel("docs/94 metric, interior slices, ED (mm)"); ax.set_ylabel("this analysis: subject severity (mm)")
    ax.set_title(f"per subject, train split (n={len(j)}, r={S['prior']['r_interior_vs_sev']:.2f})", fontsize=9); fig.tight_layout(); fig.savefig(f"{FIG}/F5_prior_vs_ours.png"); plt.close(fig)
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(json.dumps({k: v for k, v in S.items() if k not in ("per_centre", "tiers", "unmeasurable_train")}, indent=1, default=float))
    print(cent.round(3).to_string())
    for k, v in tiers.items(): print(k, v["train_kept"], v["train_dropped"], "unmeasurable kept", v["unmeasurable_kept"])
