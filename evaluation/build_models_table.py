"""Harvest per-arm provenance into models.json + MODELS.md.

One row per arm present under volumes/ (union across datasets). VGGT arms carry a
metadata.json (identical across subjects — copied per run), harvested once. Classical
baselines (svrtk3d/nesvor) have no metadata.json, so they get a minimal row. The arm dir
name is the key (same name in volumes/<ds>/out/<subj>/<arm> and checkpoints/<arm>/).

Run: micromamba run -n svr python evaluation/build_models_table.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

# metadata.json fields we surface as columns (in order)
FIELDS = ["config", "regime", "frames_per_slice", "z_mode", "date", "git_commit",
          "wandb_id", "ckpt", "note"]


def epoch_of(arm, meta):
    """Best-effort epoch from the arm name (…_ep99), model_name, ckpt path, or note.
    (The 20260715 arms carry the epoch only in the ckpt filename / note, not the arm name.)"""
    for s in (arm, meta.get("model_name", ""), meta.get("ckpt", ""), meta.get("note", "")):
        m = re.search(r"(?:^|[_-])ep(\d+)", s)   # anchored: don't match 'ep' inside step2/deep3/prep10
        if m:
            return int(m.group(1))
    return None


def harvest():
    arms = {}  # arm name -> {"datasets": set, "meta": dict|None}
    for ds in paths.DATASETS:
        for arm in paths.arms(ds):
            rec = arms.setdefault(arm, {"datasets": set(), "meta": None})
            rec["datasets"].add(ds)
            if rec["meta"] is None:  # find any subject that carries this arm's metadata.json
                for subj in paths.subjects(ds):
                    mp = paths.metadata(ds, subj, arm)
                    if mp.is_file():
                        rec["meta"] = json.loads(mp.read_text())  # read_text closes the handle
                        break
    return arms


def to_rows(arms):
    rows = []
    for arm in sorted(arms):
        rec = arms[arm]
        meta = rec["meta"] or {}
        is_vggt = arm.startswith("vggt_")
        copied = (paths.CHECKPOINTS / arm / "checkpoint.pt").is_file()
        row = {
            "arm": arm,
            "type": "vggt" if is_vggt else "baseline",
            "datasets": sorted(rec["datasets"]),
            "epoch": epoch_of(arm, meta) if is_vggt else None,
            "ckpt_copied": copied,
        }
        for f in FIELDS:
            row[f] = meta.get(f) if is_vggt else None
        rows.append(row)
    return rows


def write_markdown(rows, out):
    cols = ["arm", "type", "epoch", "config", "regime", "frames_per_slice", "z_mode",
            "date", "wandb_id", "git_commit", "ckpt_copied", "datasets", "ckpt", "note"]
    hdr = {"arm": "arm", "type": "type", "epoch": "ep", "config": "config",
           "regime": "regime", "frames_per_slice": "fps", "z_mode": "z_mode",
           "date": "date", "wandb_id": "wandb", "git_commit": "commit",
           "ckpt_copied": "copied", "datasets": "datasets", "ckpt": "ckpt (source)",
           "note": "note"}

    def cell(row, c):
        v = row.get(c)
        if v is None:
            return "—"
        if c == "datasets":
            return " ".join(v)
        if c == "ckpt_copied":
            return "✓" if v else ""
        if c == "ckpt":
            return "`" + str(v).replace("`", "'") + "`"          # keep the code-span intact
        return str(v).replace("\n", " ").replace("|", "\\|")     # keep table columns intact

    lines = [
        "# Model provenance (arms under `evaluation/volumes/`)",
        "",
        "One row per arm dir name (same name in `volumes/<ds>/out/<subj>/<arm>` and, when a",
        "durable copy exists, `checkpoints/<arm>/`). VGGT arms are harvested from their",
        "`metadata.json`; classical baselines have none. Regenerate with",
        "`python evaluation/build_models_table.py`.",
        "",
        "Config deltas (gather05/contz/lowdiff100/…) are NOT in a resolved yaml — they live in",
        "the named base config (`training/config/<config>.yaml`) + the training sbatch",
        "(`sbatch/_archive/oneframe_*.sh`). `ckpt (source)` is the original path; `copied`=✓ means a",
        "durable copy sits in `checkpoints/<arm>/checkpoint.pt`.",
        "",
        "| " + " | ".join(hdr[c] for c in cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row, c) for c in cols) + " |")
    lines.append("")
    Path(out).write_text("\n".join(lines))


def main():
    arms = harvest()
    rows = to_rows(arms)
    (paths.EVAL_ROOT / "models.json").write_text(json.dumps(rows, indent=2))
    write_markdown(rows, paths.EVAL_ROOT / "MODELS.md")
    nv = sum(r["type"] == "vggt" for r in rows)
    nb = len(rows) - nv
    print(f"-> models.json + MODELS.md  ({len(rows)} arms: {nv} vggt, {nb} baseline)")
    print(f"   with metadata: {sum(bool(r['config']) for r in rows)}  "
          f"copied ckpts: {sum(r['ckpt_copied'] for r in rows)}")


if __name__ == "__main__":
    main()
