"""Aggregate reconstruction runtime without including evaluation metrics."""
import argparse
import json
from pathlib import Path
from .config import write_json


def summarize(run):
    run = Path(run)
    fit = json.loads((run / "timing_fit.json").read_text())
    export = json.loads((run / "reconstruction/timing_export.json").read_text())
    subject = json.loads((run / "subject.json").read_text())
    prep = subject.get("preparation_seconds")
    full_cine = fit["setup_seconds_this_invocation"] + fit["optimization_seconds"] + export["load_and_full_cine_seconds"]
    result = {"preparation_seconds": prep, "fitting_setup_seconds": fit["setup_seconds_this_invocation"],
              "simulation_generation_seconds_excluded": subject.get("simulation", {}).get("source_generation_seconds"),
              "optimization_seconds": fit["optimization_seconds"],
              "load_and_full_cine_export_seconds": export["load_and_full_cine_seconds"],
              "extra_reference_query_seconds": export["additional_reference_query_seconds"],
              "end_to_end_full_cine_seconds": None if prep is None else prep + full_cine,
              "end_to_end_with_reference_query_seconds": None if prep is None else prep + full_cine + export["additional_reference_query_seconds"],
              "smoke": fit["smoke"], "protocol": subject.get("protocol"),
              "excluded": "simulation/reference generation, environment installation, scheduler wait, metrics and plotting",
              "note": "For resumed jobs, fitting setup records only the latest invocation; repeated setup overhead is not accumulated."}
    write_json(run / "timing_total.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    summarize(**vars(p.parse_args()))
