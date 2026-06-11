# 04 — Inference information contract: what the model is allowed to know per input slice

> **TL;DR & takeaway** *(human-facing; the rest is the detailed record for agents)*
>
> **Decision: target the *blind-input* extreme — at inference, assume the model knows only `z`
> (the prescribed slice position) per input slice. Input cardiac phase `t` and input respiratory
> phase `r` are assumed UNAVAILABLE.** This follows from the headline goal of *one frame per slice*:
> a single frame carries no temporal signal, so per-slice self-gating is impossible, and we
> deliberately assume **no ECG and no respiratory device** either (we want a method that needs zero
> auxiliary hardware, to maximize generality and match true sparse real-time).
>
> **Key distinction that makes this coherent:** *input* phase and *target* phase are independent.
> - **Target** `target_t` / `target_r` are always available — they are *queries we choose*, with
>   ground truth *generated* in simulation. Dropping input phase never costs us the target query.
> - **Input** `t` / `r` are the only thing the deployment environment dictates. `z` is always known
>   (geometry). The contract is therefore: **`z` known; input `t`, `r` unknown.**
>
> **Asymmetry to respect:** cardiac phase is fairly recoverable from a single slice's *appearance*
> (chamber size, wall thickness); respiratory phase is **not** (it barely shows in a cropped
> short-axis slice). So a blind model can plausibly content-infer cardiac state but has *no cue* for
> respiratory state → don't ask it to *resolve* an arbitrary `target_r`; **pin `target_r` to the
> reference (4D, "correct-not-resolve")** so the model only has to *undo* respiratory displacement
> back to the breath-hold reference.
>
> **ECG / self-gating are the documented FALLBACK, not the assumption.** If the blind model
> underperforms, recover phase and feed it: ECG-label cardiac `t` (decoupled clock, works at any
> sparsity), self-gate `t`/`r` from a temporal stream, or add a respiratory navigator. Implement
> the bridge as **input-phase dropout (+ noise)** so ONE model uses phase when present and
> content-infers when absent.
>
> **Status: design stance, not yet implemented.** The current pipeline still *conditions on* input
> `t` (the `t_embedder`); respiratory `r` is not implemented at all. Moving to the blind contract
> means training input phase with dropout up to the fully-dropped case. See
> [[01_respiratory_motion_simulation]] for the respiratory-sim side and the "Future enhancements"
> bullets in CLAUDE.md.

---

## 1. The question

For each input slice the model could in principle be told three things: its cardiac phase `t`, its
respiratory phase `r`, and its slice position `z`. Which of these does the *real-world deployment
environment* actually hand us at inference? That set — the **information contract** — is the one
thing we don't get to choose by architecture; it's dictated by the acquisition. Everything else
(target queries, 4D-vs-5D output, GT definition) is free design, because that information is always
available by construction (we choose the query; we generate the GT in sim).

## 2. Why this is the *core* question

Train/inference **parity**: a model can only rely at inference on information it will actually have
at inference. If we train with clean input `t`/`r` (free in sim) but deploy with none, the model
learns to lean on a crutch that's gone at test time — silently invalidating everything trained on
that assumption. So the contract must be decided *first*, and the training setup must honor it.

## 3. Input vs target — the load-bearing split

| role | variable | known at inference? |
|---|---|---|
| **query** (we choose) | `target_t`, `target_r` | **always** — never measured, just set; GT generated in sim |
| **source state** (per input slice) | input `t`, input `r` | **the contested part** |
| **geometry** (per input slice) | input `z` | **yes** — prescribed slice position from scanner |

Dropping input `t`/`r` does **not** block target `t`/`r`. The only coupling: if the model can't
*observe* input `r`, don't ask it to *resolve* an arbitrary `target_r` (no cue) — pin it (§6).

## 4. Why one frame per slice ⇒ no self-gated input phase

A "phase" is a position within a *cycle*, and a cycle only exists across *time*. **Self-gating**
recovers phase by watching a 1-D motion signal (k-space DC, navigator line, region intensity)
*oscillate over time* and reading off where each frame sits in the recovered cardiac/respiratory
waveform. It needs a **temporal stream**:

- Cardiac (~1 Hz): need ≥~12–30 frames/cycle over a few beats.
- Respiratory (~0.2–0.3 Hz): the binding constraint — need to span **multiple breaths** (~5+),
  i.e. tens of seconds and **hundreds of frames per slice**.

Per-slice self-gating therefore needs the exact many-frames-per-slice density the project exists to
eliminate. A *single* frame has no cycle to locate yourself in → self-gating is **impossible**, not
just inaccurate. (A *different* mechanism — **content-inference**, reading phase from one image's
appearance — needs no stream; that's what a blind model does internally. §5.)

**Gating vs labeling (why ECG is special).** Retrospective *gating* = binning many frames by phase
(slow, many frames/slice — eliminated by the project). *Labeling* = attaching a phase to whatever
frame you have, which needs only a **clock** + timestamp. The **ECG is a clock decoupled from the
imaging-frame count**: one frame at wall-clock time τ → look up the ECG cardiac phase at τ. So ECG
labeling gives input `t` at *any* sparsity, including one frame per slice — *if* an ECG is recorded.

## 5. Why we assume no ECG / no respiratory device (and the asymmetry)

ECG is, in fact, standard and effectively free in cardiac MR (4-lead setup on every patient), so
assuming it is realistic. **But we deliberately target the harder extreme — no ECG, no respiratory
device — because we want a method that needs zero auxiliary hardware** and generalizes to arbitrary
sparse real-time cine (incl. datasets where no ECG was logged). ECG/self-gating then sit in reserve
as a fallback (§7).

Recoverability of input phase from slice *content* (the blind fallback inside the model) is
asymmetric:

- **Cardiac `t`: plausibly recoverable** from a single short-axis slice — chamber area / wall
  thickness strongly encode cardiac phase (some ambiguity between look-alike phases).
- **Respiratory `r`: near-unrecoverable** from a single *cropped* short-axis slice — the main cue is
  the heart's absolute SI position shift, much of which the canonical crop/registration removes, and
  the diaphragm is usually out of FOV.

Two risks of going fully blind: (a) respiratory correction is **underdetermined** (no cue); (b)
**phase-averaged blur** — if the model can't infer source phase it may regress the safe L1 minimizer
(a mush averaged over the unknown phase), same degeneracy family as the "fully-unsupervised
`V_canon≈0`" risk in CLAUDE.md.

## 6. Consequence for the output: pin `target_r` (4D, correct-not-resolve)

Because input `r` is unobservable, the respiratory axis enters **asymmetrically** from cardiac:

| | input (per slot) | output query |
|---|---|---|
| cardiac | `t` (recoverable-ish / fallback ECG) | `target_t` — **queried dial** |
| respiratory | `r` (assumed unknown) | `target_r` — **pinned to reference**, not queried |

Reconstruct the full cardiac cycle at **one** respiratory phase (end-expiration ≈ the breath-hold
reference) → 4D output (XYZ × cardiac), which also matches the breath-hold GT we actually have. Full
5D (queryable `target_r`) is cheap to *build* but its value is capped by sim fidelity (it would learn
to reproduce our own respiratory-sim motion model) and there's no real 5D reference — defer it. The
simulation gives respiratory-correction GT *for free precisely because* we only ever target the
single breath-hold respiratory phase. See [[01_respiratory_motion_simulation]].

## 7. Fallbacks if blind underperforms (kept in reserve)

Implement all of these as **input-phase dropout (+ noise)** on the existing/new embedders, so a
single model spans the spectrum from "phase available" to "fully blind":

1. **ECG-label cardiac `t`** — decoupled clock; works at any imaging sparsity. In-bore ECG can be
   degraded by the magnetohydrodynamic effect, so feed it with mild label noise.
2. **Self-gate `t`/`r`** — only if a *temporal stream* (and/or a cheap continuous navigator) exists;
   self-gate the dense stream, label frames, then sparsify.
3. **Respiratory navigator / bellows** — gives `r` for an upgrade to true 5D.

## 8. What this changes in the code (not yet done)

- Current: `t_embedder` conditions on input `t`; `target_t_embedder` provides the query; no `r`.
- Target stance: train input `t` (and a future input-`r` embedder) with **dropout up to fully
  dropped**, so the default inference path is blind; keep `target_t` (and a pinned `target_r`) as
  queries. The blind model must then content-infer cardiac state internally.
- Watch for the **phase-averaged blur** degeneracy (§5) — may need a coverage/completeness or
  stronger smoothness term, mirroring the fully-unsupervised note in CLAUDE.md.
