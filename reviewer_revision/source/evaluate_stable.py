"""Apply validation-frozen operating points to stable-agriculture controls."""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from evaluate import (METHODS, LABELS, OUT, load_linear, probability, top9, wilson)
from models import indices

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
FEATURES = BENCH / "data/processed/reviewer_revision/stable_agriculture/features"
SOURCE = BENCH / "results/last_effort"


def spectral_features(s20: np.ndarray, s24: np.ndarray) -> np.ndarray:
    return np.concatenate([s20, s24, s24 - s20, np.abs(s24 - s20),
                           indices(s20), indices(s24), indices(s24) - indices(s20)], axis=1)


def predict() -> list[dict]:
    thresholds = json.loads((OUT / "operating_points.json").read_text())["thresholds"]
    token_head = load_linear(SOURCE / "tokens_landuse_head.pt", 768)
    spectral_head = load_linear(OUT / "spectral_landuse_converged.pt", 14)
    with (OUT / "bitemporal_terrafm.pkl").open("rb") as stream:
        direct_model = pickle.load(stream)
    with (OUT / "bitemporal_spectral_forest.pkl").open("rb") as stream:
        forest = pickle.load(stream)
    with (OUT / "stable_agriculture_reference.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    predictions = []
    for row in rows:
        with np.load(FEATURES / f"{row['window_id']}.npz") as saved:
            t20 = saved["tokens_2020"].astype(np.float32).reshape(768, -1).T
            t24 = saved["tokens_2024"].astype(np.float32).reshape(768, -1).T
            s20 = saved["spectral_2020"].astype(np.float32).reshape(14, -1).T
            s24 = saved["spectral_2024"].astype(np.float32).reshape(14, -1).T
            change = saved["change"].reshape(-1).astype(np.float32)
            optical = float(saved["optical_fraction"])
        p20, p24 = probability(*token_head, t20), probability(*token_head, t24)
        agri = p24[:, 2]
        linear_agri = probability(*spectral_head, s24)[:, 2]
        temporal = np.concatenate([t20, t24, t24 - t20, np.abs(t24 - t20)], axis=1)
        conventional = spectral_features(s20, s24)
        scores = {
            "joint_original": top9(agri * change),
            "endpoint_terrafm": top9(agri),
            "postclassification": top9(p20[:, 0] * agri),
            "bitemporal_terrafm": top9(direct_model.predict_proba(temporal)[:, 1]),
            "spectral_linear_endpoint": top9(linear_agri),
            "spectral_linear_joint": top9(linear_agri * change),
            "spectral_forest_bitemporal": top9(forest.predict_proba(conventional)[:, 1]),
            "change_original": top9(change),
            "optical_index": optical,
        }
        predictions.append({**row, **scores, **{
            f"flag_{method}": int(scores[method] >= thresholds[method]["threshold"])
            for method in METHODS}})
    return predictions


def main() -> None:
    predictions = predict()
    with (OUT / "stable_agriculture_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(predictions)
    groups = {"all": predictions}
    groups.update({region: [r for r in predictions if r["region"] == region]
                   for region in sorted({r["region"] for r in predictions})})
    report = {"n": len(predictions), "groups": {}}
    for name, records in groups.items():
        report["groups"][name] = {"n": len(records), "methods": {}}
        for method in METHODS:
            flagged = sum(int(r[f"flag_{method}"]) for r in records)
            report["groups"][name]["methods"][method] = {
                "label": LABELS[method], "false_positives": flagged,
                "false_positive_rate": flagged / len(records),
                "ci95_wilson": wilson(flagged, len(records)),
            }
    comparisons = []
    for first, second in [("endpoint_terrafm", "joint_original"),
                          ("endpoint_terrafm", "postclassification"),
                          ("endpoint_terrafm", "bitemporal_terrafm"),
                          ("postclassification", "spectral_forest_bitemporal"),
                          ("bitemporal_terrafm", "spectral_forest_bitemporal")]:
        a = np.asarray([r[f"flag_{first}"] for r in predictions], dtype=bool)
        b = np.asarray([r[f"flag_{second}"] for r in predictions], dtype=bool)
        a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
        comparisons.append({"first": first, "second": second,
                            "first_only_flagged": a_only, "second_only_flagged": b_only,
                            "exact_two_sided_p": float(binomtest(min(a_only, b_only),
                                                                  a_only + b_only, .5).pvalue)
                            if a_only + b_only else 1.0})
    report["paired_flag_comparisons"] = comparisons
    (OUT / "stable_agriculture_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["groups"]["all"], indent=2), flush=True)


if __name__ == "__main__":
    main()
