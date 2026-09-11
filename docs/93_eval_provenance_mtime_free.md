# 93 — evaluation/ provenance is content-keyed, not mtime-keyed (GPFS touch-safe)

> **TL;DR & takeaway** (2026-09-01). GPFS purges untouched files, so we need to periodically
> `find … -exec touch` the eval volumes. The harness keyed two things on file mtimes and both
> would have broken: (1) **`ckpt_fingerprint = "<size>:<int(mtime)>"`** — every existing arm would
> read as a *different checkpoint* (`run_vggt.py check_overwrite` refuses; `aggregate.py` warns
> "mixes N checkpoints"); (2) **"derived file older than `gt_t00`"** freshness for
> `cine_gt` / `cine_{clean,breath}` and "seg older than `ef_manifest.json`" — a blanket touch
> lands in arbitrary order, so these flipped at random (arms skipped, `sys.exit`). Both are now
> **content ids**: `v2:<size>:<sha256(first+last 64 MiB)[:16]>` for checkpoints and container
> `.sif`s (same bytes from bash and Python — verified), `sha256(gt_t00)` recorded in
> `cine_gt.src.json` and `metrics.json["gt_sha256"]` for cine freshness, and a random `dump_id`
> (manifest ⇄ `<seg_dir>/ef_dump_id`, written by `run_seg.sh`) for seg provenance. Existing files
> were migrated in place by `tools/migrate_eval_fingerprints.py` (dry-run then `--apply`): 2882
> fingerprints + 288 container ids rewritten as substrings, 1011 `gt_sha256` keys and 144
> `cine_gt.src.json` sidecars added, 0 unverifiable checkpoints, 0 stale cines. **Touching
> `scratch/` is now safe.** MRI2CT/evaluation never read mtimes and needed nothing.

## What changed (all in `evaluation/`, plus the migration under `tools/`)

| Site | Before | After |
|---|---|---|
| `paths.py` | — | `ckpt_fingerprint`, `same_fingerprint`, `gt_sha256`, `file_sha256`, `cine_gt_src` |
| `src/engine/run_vggt.py` `_ckpt_fingerprint` / `_same_ckpt` | `size:int(mtime)`; string equality | `paths.ckpt_fingerprint`; legacy-vs-v2 pairs fall back to realpath |
| `src/engine/run_svrtk3d.sh`, `run_nesvor.sh` `container_id` | `stat -c '%s:%Y' $SIF` | bash `container_id()` — identical format/bytes to the Python id |
| `src/score/image_metrics.py` | regenerate `cine_gt` if mtime < `gt_t00` mtime | regenerate if `cine_gt.src.json["gt_sha256"] != sha256(gt_t00)`; writes the sidecar; adds `metrics["gt_sha256"]` |
| `src/score/ef_dice.py dump` | skip arm if `cine_*` mtime < `gt_t00` mtime | skip if `metrics.json["gt_sha256"]` (and `cine_gt.src.json`) ≠ live gt hash; manifest gets `meta.dump_id` |
| `src/score/ef_dice.py score` | `sys.exit` if any seg mtime < manifest mtime | `sys.exit` unless `<seg_dir>/ef_dump_id == manifest.dump_id` (legacy manifests without an id are refused) |
| `src/engine/run_seg.sh` | — | copies `dump_id` into `<seg_dir>/ef_dump_id`; refuses a seg_dir stamped with another id or holding unstamped segs |
| `src/score/aggregate.py` | key by fingerprint if all rows have one | … and all are the same format |

`metrics["recon_mtime"]` is still written (nothing reads it) — left alone.

## Why head+tail hash, not full-file

A full sha256 of a ~9 GB checkpoint is a sequential GPFS read (~1–2 min) *per run_vggt
invocation*; the first and last 64 MiB (~1 s) already differ between any two training checkpoints
(tensor bytes at both ends). Size is included as a cheap second discriminator.

## Migration semantics (why it had to run BEFORE any touch)

The script uses the *old* mtime rules exactly once to decide what the existing files are:
a legacy fingerprint is rewritten only if it still equals `size:int(mtime)` of the checkpoint on
disk (proof it is the same file); `gt_sha256` is added to a `metrics.json` only if every existing
`cine_*` beside it postdates `gt_t00`. Anything unverifiable is reported and left alone (none
were). Substring replacement for fingerprints, json re-dump for the added key only when the
re-dump reproduces the file byte-for-byte otherwise. Idempotent: a second dry run reports 0 work.

## Verification

- bash `container_id` vs `paths.ckpt_fingerprint` on a 150 MB and a 1 kB random file: identical.
- `_ckpt_fingerprint` / `_same_ckpt` unchanged by `os.utime`; v2 vs v2 mismatch → False;
  legacy vs v2 → realpath fallback (same path True, other path False); `gt_sha256` unchanged by touch.
- Migration applied to a copied subject fixture: unified diff showed only the fingerprint
  substrings, the added `gt_sha256` key, and the new sidecar; `timing.json`/`resp_diag.json` untouched.
