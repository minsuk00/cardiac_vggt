# 90 — NeSVoR native env (`nesvor-t2`) + torch-2 port: 2.78× faster, container retired

> **TL;DR & takeaway**
> We were always running **real upstream NeSVoR** — the pinned `junshenxu/nesvor:v0.5.0` image ships the
> full source and `evaluation/src/engine/run_nesvor.sh` invokes its own `nesvor reconstruct` CLI. What we
> lacked was an *editable* copy, and the container's CUDA was built for **sm_70 (V100) only**, so on our
> Ampere/Ada cards every kernel ran through PTX JIT. Fixed by going native: cloned upstream at `v0.5.0`
> to `baselines/nesvor/NeSVoR`, ported **10 lines** of CUDA (`AT_DISPATCH_FLOATING_TYPES(x.type(), …)` →
> `x.scalar_type()`, removed in torch 2), and built one micromamba env **`nesvor-t2`** (torch 2.13.0+cu130)
> with tiny-cuda-nn and both NeSVoR extensions compiled for **sm_86 AND sm_89** — one env covers A40 and
> L40S. **MEASURED, same subject/phase/config: A40 native 165 s vs A40 container 459 s = 2.78× (the A40
> ≈462 s in `baselines/nesvor/readme.md` reproduced at 459 s, and the speedup is now unconfounded — same
> GPU, only the build differs); L40S native 118 s = 3.9× vs the container A40.** Correctness gated on a
> **noise floor**: two runs of the *identical* container agree only to corr 0.9936 (NeSVoR randomly inits
> the INR and MC-samples the PSF), and native-vs-container lands at **0.9962** — *inside* the noise. Plus
> NeSVoR's own `test_cg_recon` passes (`atol=3e-5`) and the transform kernels match scipy to **6e-08**
> with gradcheck passing. **NOT DONE: nothing in `evaluation/` points at the new env**, so no published
> number has changed yet, and the cohort decision (re-run natively vs re-time only) is open — §7.

Companion: `baselines/nesvor/readme.md` (mechanism + operational flags), `docs/32` (first container run),
`docs/84` (the 144-subject baseline campaign that spent **90.8 GPU-h** on NeSVoR).

---

## 1. The starting misconception, corrected

The premise that prompted this ("we're not using NeSVoR's codebase to run, right?") was **false**. Evidence:

- The image installs NeSVoR as an **editable install**:
  `/usr/local/lib/python3.10/dist-packages/nesvor.egg-link → /usr/local/NeSVoR`, plus `easy-install.pth`.
- `singularity exec … python3 -c "import nesvor"` → `0.5.0` from `/usr/local/NeSVoR/nesvor`.
- `diff -rq` of the container's `/usr/local/NeSVoR` against upstream tag **`v0.5.0` (`730ddaa`)`**:
  **identical**, the only extra file being the Dockerfile-generated `install.sh`.

So we ran stock upstream code the whole time. The real gaps were (a) the source wasn't readable/editable,
and (b) the arch problem in §2.

**The `.sif` is the environment**, not a NeSVoR reimplementation: `singularity inspect` reports
`deffile.bootstrap: docker`, `deffile.from: junshenxu/nesvor:v0.5.0`. It carries Ubuntu 22.04 userland,
python 3.10.6, torch 1.13.1+cu117, tiny-cuda-nn, the compiled extensions, and the source.

**v0.5.0 vs upstream `master` (0.6.0rc1)** — checked, effectively nothing for us: a multi-GPU
`torch.cuda.set_device(args.device)` fix (issue #14; no-op since we always pass `--device 0`), two missing
spaces in a help string, a `setup.py` README guard, an off-by-one in a *test* phantom generator. Everything
else is Docker/packaging. **0.6.0rc1 is a packaging release, not a new method** — staying on `v0.5.0` costs
nothing scientifically and keeps us bit-comparable with published recons.

---

## 2. The real defect: everything CUDA in the container is sm_70

Measured with `cuobjdump` on the extracted binaries:

| Component | Built for |
|---|---|
| tiny-cuda-nn binding | `tinycudann_bindings/_70_C.cpython-310.so` |
| `slice_acq_cuda…so` | `sm_70` cubin + `compute_70` PTX |
| `transform_convert_cuda…so` | `sm_70` cubin + `compute_70` PTX |

`sm_70` is **Volta**. A40 is `sm_86` (Ampere), L40S is `sm_89` (Ada) — a **major**-version gap, so the
cubins can't run and the driver must JIT the `compute_70` PTX at load time. Functional and numerically
valid, but not optimized for the card. This is why `baselines/nesvor/readme.md` recorded **V100 195 s vs
A40 462 s** — the V100 being *faster* than a much newer card was the symptom.

That readme attributed the gap to the arch, but the evidence was **confounded** (two different GPUs). §5
de-confounds it.

> **Minor-version compatibility (why the L40S gap was never as bad).** Cubins ARE forward-compatible
> across *minor* versions within one major: `sm_86` SASS runs natively on `sm_89`. Visible in torch
> 2.13+cu130's own arch list — `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`, **no `sm_89`** — yet
> PyTorch fully supports Ada. So compiling for `sm_89` buys *Ada-tuned* code, not an escape from JIT;
> `sm_70 → sm_86` is the severe case because it crosses major versions.

---

## 3. What was built

### 3.1 Source clone
`baselines/nesvor/NeSVoR` — `git clone https://github.com/daviddmc/NeSVoR`, checked out at **`v0.5.0`**.
~16 MB. The 1.3 GB of fetal-brain checkpoints (`SVoRT_v2.pt`, seg, IQA) were **deliberately not extracted** —
reachable only via `--registration svort` / seg / IQA commands, none of which we run. If ever needed, copy
`/usr/local/NeSVoR/nesvor/checkpoints` out of the image to `scratch/` (never `$HOME`).

### 3.2 The torch-2 port — **10 lines, and the only change to NeSVoR's code**

```cuda
- AT_DISPATCH_FLOATING_TYPES(vol.type(), "slice_acquisition_forward_cuda", [&] {
+ AT_DISPATCH_FLOATING_TYPES(vol.scalar_type(), "slice_acquisition_forward_cuda", [&] {
```

6 sites in `nesvor/slice_acquisition/slice_acq_cuda_kernel.cu`, 4 in
`nesvor/transform/transform_convert_cuda_kernel.cu`. Without it the build dies:

```
error: no suitable conversion function from "const at::DeprecatedTypeProperties" to "c10::ScalarType" exists
```

**Why it is semantics-preserving.** `tensor.type()` returns `at::DeprecatedTypeProperties`. In torch 1.x
that class had an implicit conversion to `c10::ScalarType` — which is all `AT_DISPATCH_FLOATING_TYPES` ever
consumed — and that conversion returns `scalarType()`, exactly what `.scalar_type()` returns. Torch 2
deleted the conversion, not the semantics. `.type()` also carried backend (CPU/CUDA) info, which the macro
discards. Same dtype dispatched, same kernel template instantiated. Verified numerically in §6.

`git diff` in the clone shows the whole change; `git stash` reverts it.

> ⚠️ **An earlier audit of this claimed "no removed-API hits" and was WRONG** — it grepped `\.type\(\).`
> (trailing dot), which cannot match a bare `x.type()`. The build error is what caught it. Lesson: a grep
> that finds nothing is not evidence of absence until the pattern is shown to fire on a known positive.

### 3.3 The env

```bash
micromamba create -n nesvor-t2 python=3.10
pip install "setuptools<81" wheel                       # 81+ dropped pkg_resources
pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130
module load gcc/13.2.0 cuda/13.1.0                      # CUDA 13 needs gcc >= 12
TCNN_CUDA_ARCHITECTURES="86;89" pip install --no-build-isolation \
    "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
pip install "nibabel>=3.2.2" "scipy>=1.7.3" "scikit-image>=0.20.0" "SimpleITK>=2.2.1"
TORCH_CUDA_ARCH_LIST="8.6;8.9" pip install --no-build-isolation -e baselines/nesvor/NeSVoR
```

Build script: `temp/build_nesvor_t2.sh` (gitignored — promote if we keep this).
Resulting env is **5.1 GB**; tcnn is **2.0** (commit `749dd70`), vs the container's 1.7.

**Two build gotchas, both real:**
1. **`ModuleNotFoundError: No module named 'pkg_resources'`** — pip's build isolation installs the newest
   setuptools (84), which removed `pkg_resources`; tiny-cuda-nn's `setup.py` imports it at line 5. Fix:
   pin `setuptools<81` **and** pass `--no-build-isolation` (also required regardless, since tcnn needs
   `torch` importable at build time and an isolated env has none).
2. `torch==2.13.0+cu130` chosen because it is the **one torch-2 stack with positive local evidence** —
   the `svr` env already has sm_86 tcnn built against it.

**Deps deliberately skipped:** `tensorflow==2.8` and `monai==0.3.0`. Both are lazily imported with graceful
`ImportError`s and only used by brain-segmentation / 3D-IQA commands we never call.

**Why torch 2 at all** (upstream does *not* pin torch — `requirements.txt` says `torch>=1.10.2`; the
1.13.1 comes only from their Dockerfile): CUDA 11.7 **predates Ada and cannot emit `sm_89`**, so a cu117
env is A40-only. A cu117 env WAS built and validated first (`nesvor-native`, sm_86, 165 s — identical to
the torch-2 result) and then **deleted** in favour of one env covering both cards.

---

## 4. Method: how correctness was gated

**NeSVoR is stochastic** — fresh INR per scan with random init plus Monte-Carlo PSF sampling — so outputs
are not reproducible even container-vs-container. Any "the port matches" claim needs a **noise floor**:

- **Floor** = container-on-A40 vs container-on-V100 (identical code, different run).
- **Test** = native vs container-on-A40 (**same GPU**, so only the build differs).
- Pass iff test agreement ≥ floor.

Comparisons use the scorer's self-percentile normalization (`score/image_metrics.py` rule: recon's own
0.5/99.9 over its support), because NeSVoR's output gauge is arbitrary (~700 mean). **These PSNRs are
recon-vs-recon agreement, NOT quality — do not confuse them with the GT-referenced numbers in docs/84–88.**

Script: `temp/validate_nesvor_t2.sh`. Subject `CMRx24_Test_P012`, `breath`, `t00`, full NeSVoR defaults
(`--n-iter 6000`, `--n-samples 256`, `--thicknesses 8`, `--registration none`, `--output-resolution 1.4`).
Chosen because its container recon is stamped and its provenance records **V100, J=1, 191 s for `t00`**.

---

## 5. Results — MEASURED

### 5.1 Speed (same subject / phase / config)

| Build | GPU | Wall | vs container-A40 |
|---|---|---|---|
| container (sm_70, PTX-JIT) | A40 | **459 s** | 1.00× |
| container (sm_70, native) | V100 | 191 s | 2.40× |
| **native torch-2 (sm_86)** | **A40** | **165 s** | **2.78×** |
| **native torch-2 (sm_89)** | **L40S** | **118 s** | **3.89×** |

- The readme's 462 s reproduced at **459 s**.
- **2.78× is unconfounded** — both legs on the same A40, same subject, same flags.
- Native A40 (165 s) now **beats the V100** (191 s), 1.16×. A prior worry that the V100's HBM2 bandwidth
  (900 GB/s vs the A40's 696 GB/s) might dominate a memory-bound hash-grid workload did **not** materialize.
- L40S is fastest at 118 s (1.40× vs A40 native).

**Campaign implication** (docs/84 spent **90.8 GPU-h** on NeSVoR for 144 subjects × 12 phases on V100):
A40 native ≈ **79 GPU-h**, L40S native ≈ **57 GPU-h**. Not re-run — arithmetic from the per-phase times.

### 5.2 Agreement (recon-vs-recon; floor = 0.9936)

| Comparison | corr | PSNR | support IoU |
|---|---|---|---|
| **noise floor** — container A40 vs container V100 | 0.99360 | 32.34 dB | 1.000 |
| native A40 vs container A40 | **0.99616** | 34.56 dB | 1.000 |
| native A40 vs container V100 | 0.99418 | 32.78 dB | 1.000 |
| L40S vs native A40 | 0.99707 | 35.80 dB | — |
| L40S vs container V100 | 0.99630 | 34.74 dB | — |

Every native row is **at or above the floor** — the port is inside the method's own run-to-run noise.
Identical support (IoU 1.000). L40S job **58868955** (`spgpu2`, `jjparkcv_owned1`), COMPLETED in 2:06.

### 5.3 Unit-level verification of the patch

| Sites | Check | Result |
|---|---|---|
| 6 in `slice_acq_cuda_kernel.cu` | NeSVoR's own `tests/slice_acquisition/test_slice_acq.py::test_cg_recon` — phantom, 16 orientations, 20-iter CG SRR through forward + adjoint + backward | **PASS**, `atol=3e-5, rtol=1e-5` |
| 4 in `transform_convert_cuda_kernel.cu` | `axisangle2mat` vs scipy `Rotation` (independent impl) | max abs err **5.96e-08** |
| | `mat2axisangle(axisangle2mat(x))` round-trip | max abs err **1.19e-07** |
| | backward vs finite differences | **gradcheck PASS** |

6e-08 on float32 is machine epsilon.

**Pre-existing upstream breakage, NOT ours:** `tests/transform/test_transform_convert.py` fails to import
in v0.5.0 (`cannot import name 'axisangle2mat_torch' from 'nesvor.transform'`) — the test is stale against
the shipped API. It fails identically in the container. The scipy check above covers what it intended.

### 5.4 Residual risk, stated honestly

The 10 patched lines are verified at machine precision. The **unquantified** risk is everything else in
torch 1.13→2.13 and CUDA 11.7→13.1 that the diff does not touch: AMP/`GradScaler` semantics (NeSVoR emits
`FutureWarning: torch.cuda.amp.autocast(args...) is deprecated` — cosmetic, not fixed), RNG streams,
`grid_sample` edge handling, reduction ordering. Evidence against it is **n=1 subject × 1 phase** landing
above the noise floor. Real, but not exhaustive.

---

## 6. What this changes for editing NeSVoR

`baselines/nesvor/NeSVoR` is now the editable install for `nesvor-t2` — **both python (59 files) and the
4 CUDA/C++ sources are live**; a `pip install -e .` re-compile picks up kernel edits. The container had no
`nvcc`, so CUDA edits were impossible there. The clone is `v0.5.0 + the §3.2 port`, not stock upstream —
state that in the paper's methods.

---

## 7. NOT DONE — open items

1. **Nothing in `evaluation/` uses the new env.** `evaluation/src/engine/run_nesvor.sh` still calls the
   container via singularity. **No published number has changed**, and the 2.78× is not reflected in any
   baseline. A runner pointed at `nesvor-t2` is needed — per project rule, that goes to `tools/` and the
   user decides whether it belongs in `evaluation/`.
2. **Cohort decision, OPEN.** The 144-subject docs/84 recons came from the container. Either (a) re-run the
   cohort natively for internal consistency (≈57–79 GPU-h), or (b) keep existing recons for **quality** and
   use native only for **timing** — defensible now that equivalence is measured, but it must be stated that
   timing and reconstruction came from different runs.
3. **Which GPU to standardize on, OPEN.** A fair speed comparison needs *every* method on one card. L40S is
   fastest but needs `jjparkcv_owned1`/`spgpu2`; A40 is the default `spgpu`. SVRTK is CPU, so it needs its
   own consistent CPU allocation.
4. **Validation is n=1** (1 subject × 1 phase per GPU). Broadening it is cheap and would harden §5.4.
5. **Obsolete artifacts from the abandoned container-bind approach**, not yet removed:
   `baselines/nesvor/nesvor_src.sh` (wrapper that bind-mounted the clone over `/usr/local/NeSVoR`) and
   `scratch/nesvor/blobs_v050/` (sm_70 `.so` + egg-info extracted from the image). Keep only if the
   container path is wanted as a fallback.
6. **`baselines/nesvor/readme.md` §7 is stale** — it documents the bind-mount plan, superseded by this doc.
7. **Scripts live in gitignored `temp/`**: `build_nesvor_t2.sh`, `validate_nesvor_t2.sh`,
   `test_nesvor_l40s.sh`. Promote to `tools/` if we keep this route.
8. **AMP deprecation warnings** not addressed (cosmetic).
9. `/home` was at **94.8 % (75.85/80 GiB)** during this work; the cu117 env was deleted to free 4.1 GB.
   `nesvor-t2` costs 5.1 GB. Watch it.

---

## 8. Reproduce

```bash
bash temp/build_nesvor_t2.sh          # env + tcnn(86;89) + NeSVoR extensions
bash temp/validate_nesvor_t2.sh       # A40: native vs container, + noise floor
sbatch temp/test_nesvor_l40s.sh       # L40S (spgpu2, jjparkcv_owned1)

micromamba activate nesvor-t2 && module load gcc/13.2.0 cuda/13.1.0
cd baselines/nesvor/NeSVoR && python -m pytest tests/slice_acquisition -q
```

Artifacts: `temp/nesvor_t2_validate/compare.json`, `temp/nesvor_l40s_test/compare.json`,
`slurm_logs/nesvor_l40s_test_58868955.out`.
