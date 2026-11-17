"""Verify every revised method from an extracted supplementary bundle, offline.

Run: python verify_revision.py --bundle .
Only the supplied cache and fitted downstream models are required. No fitting or
published-file modification occurs. Output is written beneath rerun/.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def compare(fresh, saved_path, methods):
    with saved_path.open() as f:
        saved = {r["window_id"]: r for r in csv.DictReader(f)}
    assert {r["window_id"] for r in fresh} == set(saved), saved_path
    errors = {m: max(abs(float(r[m])-float(saved[r["window_id"]][m])) for r in fresh)
              for m in methods}
    assert max(errors.values()) < 2e-6, (saved_path, errors)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(); bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    for item in manifest["files"]:
        assert sha(bundle / item["path"]) == item["sha256"], item["path"]
    sys.path.insert(0, str(bundle / "reviewer_revision/source"))
    import models
    import evaluate as ev
    import evaluate_stable as stable
    import critical_audit as audit
    models.SOURCE = ev.SOURCE = stable.SOURCE = bundle / "published/dataset_c"
    models.DATA = ev.DATA = bundle / "cache"
    models.PAIRED = ev.PAIRED = bundle / "cache/features_2020"
    models.OUT = ev.OUT = stable.OUT = audit.OUT = bundle / "reviewer_revision/results"
    stable.FEATURES = audit.STABLE = bundle / "cache/stable_agriculture/features"
    torch.set_num_threads(4)
    frozen = json.loads((ev.OUT / "operating_points.json").read_text())["thresholds"]
    errors = {}; results = {}
    for split in ("validation", "test"):
        fresh = ev.predict(split)
        errors[split] = compare(fresh, ev.OUT / f"{split}_predictions.csv", ev.METHODS)
        if split == "validation":
            for method in ev.METHODS:
                assert abs(ev.choose_threshold(fresh, method)["threshold"] - frozen[method]["threshold"]) < 2e-6
        else:
            loss = [r for r in fresh if r["endpoint"] != "stable_forest"]
            results["loss_endpoints"] = ev.metrics(loss, frozen)
            published = json.loads((ev.OUT / "evaluation.json").read_text())["results"]["loss_endpoints"]
            for method in ev.METHODS:
                a,b = results["loss_endpoints"]["methods"][method], published["methods"][method]
                assert all(a[k] == b[k] for k in ("tp","fn","fp","tn"))
                assert abs(a["auroc"] - b["auroc"]) < 1e-10
    fresh = stable.predict()
    errors["stable"] = compare(fresh, ev.OUT / "stable_agriculture_predictions.csv", ev.METHODS)
    published = json.loads((ev.OUT / "stable_agriculture_evaluation.json").read_text())["groups"]["all"]
    results["stable_flags"] = {}
    for method in ev.METHODS:
        count = sum(r[method] >= frozen[method]["threshold"] for r in fresh)
        assert count == published["methods"][method]["false_positives"]
        results["stable_flags"][method] = int(count)
    added = json.loads((audit.OUT / "critical_audit.json").read_text())
    for split in ("validation","test","stable"):
        fresh = audit.predict(split)
        errors[f"spectral_PC_{split}"] = compare(fresh, audit.OUT / f"critical_audit_{split}_predictions.csv", [audit.METHOD])
        if split == "validation":
            assert abs(ev.choose_threshold(fresh,audit.METHOD)["threshold"] - added["operating_point"]["threshold"]) < 2e-6
        elif split == "test":
            loss = [r for r in fresh if r["endpoint"] != "stable_forest"]
            rerun = audit.bootstrap(loss)
            assert np.allclose(rerun["ci95"], added["bootstrap"]["ci95"], atol=1e-10)
            assert np.allclose(rerun["terrafm_minus_spectral_ci95"], added["bootstrap"]["terrafm_minus_spectral_ci95"], atol=1e-10)
    report = {"status":"passed", "scope":"all nine revised methods and exploratory spectral post-classification",
              "manifest_sha256":sha(bundle / "manifest.json"),
              "verified_files":len(manifest["files"]), "maximum_score_errors":errors,
              "results":results, "added_paired_bootstrap":"passed"}
    destination = bundle / "rerun/revision_verification.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report,indent=2)+"\n")
    print("PASS: revised scores, validation thresholds, loss metrics, stable flags and added paired bootstrap.")
    print(destination)


if __name__ == "__main__": main()
