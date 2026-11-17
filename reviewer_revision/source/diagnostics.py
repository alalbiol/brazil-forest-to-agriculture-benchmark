"""Geographic and threshold-stability diagnostics from frozen predictions."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import BallTree

from evaluate import METHODS, OUT

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
SOURCE = BENCH / "results/last_effort"
SEED = 20261015
EARTH_RADIUS_M = 6_371_008.8


def read(path: Path) -> list[dict]:
    with path.open() as stream:
        records = list(csv.DictReader(stream))
    for record in records:
        for method in METHODS:
            if method in record:
                record[method] = float(record[method])
    return records


def coarse_id(row: dict, kilometres: int) -> str:
    x = float(row["lon"]) * 111.32 * math.cos(math.radians(-7))
    y = float(row["lat"]) * 111.32
    return f"{row['region']}:{math.floor(x / kilometres)}:{math.floor(y / kilometres)}"


def coarse_bootstrap(records: list[dict], kilometres: int, iterations: int = 4000) -> dict:
    groups = sorted({coarse_id(row, kilometres) for row in records})
    lookup = {group: [i for i, row in enumerate(records)
                      if coarse_id(row, kilometres) == group] for group in groups}
    y = np.asarray([r["endpoint"] == "agriculture" for r in records])
    score = {method: np.asarray([r[method] for r in records]) for method in METHODS}
    draws = {method: [] for method in METHODS}
    rng = np.random.default_rng(SEED + kilometres)
    for _ in range(iterations):
        indices = np.asarray([i for group in rng.choice(groups, len(groups), replace=True)
                              for i in lookup[group]])
        if len(set(y[indices])) < 2:
            continue
        for method in METHODS:
            draws[method].append(roc_auc_score(y[indices], score[method][indices]))
    return {"cluster_km": kilometres, "clusters": len(groups),
            "methods": {method: {"valid_draws": len(values),
                                  "ci95": np.percentile(values, [2.5, 97.5]).tolist()}
                        for method, values in draws.items()}}


def paired_loss_contrast(records: list[dict], first: str, second: str,
                         iterations: int = 4000) -> dict:
    blocks = sorted({r["block_id"] for r in records})
    lookup = {block: [i for i, row in enumerate(records) if row["block_id"] == block]
              for block in blocks}
    y = np.asarray([r["endpoint"] == "agriculture" for r in records])
    a = np.asarray([r[first] for r in records]); b = np.asarray([r[second] for r in records])
    estimate = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(SEED); draws = []
    for _ in range(iterations):
        indices = np.asarray([i for block in rng.choice(blocks, len(blocks), replace=True)
                              for i in lookup[block]])
        if len(set(y[indices])) < 2:
            continue
        draws.append(roc_auc_score(y[indices], a[indices]) -
                     roc_auc_score(y[indices], b[indices]))
    return {"first": first, "second": second, "estimate": estimate,
            "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
            "valid_draws": len(draws), "unit": "original 10 km block"}


def add_distances(test: list[dict]) -> dict:
    with (SOURCE / "reference.csv").open() as stream:
        train = [row for row in csv.DictReader(stream)
                 if row["reference_label"] == "accepted" and row["split"] == "train"]
    tree = BallTree(np.radians([[float(r["lat"]), float(r["lon"])] for r in train]),
                    metric="haversine")
    query = np.radians([[float(r["lat"]), float(r["lon"])] for r in test])
    distances = tree.query(query, k=1)[0][:, 0] * EARTH_RADIUS_M / 1000
    for row, distance in zip(test, distances):
        row["nearest_training_km"] = float(distance)
    edges = [0, 25, 50, 100, float("inf")]
    bins = []
    thresholds = json.loads((OUT / "operating_points.json").read_text())["thresholds"]
    for low, high in zip(edges[:-1], edges[1:]):
        records = [row for row in test if low <= row["nearest_training_km"] < high and
                   row["endpoint"] != "stable_forest"]
        if not records:
            continue
        y = np.asarray([r["endpoint"] == "agriculture" for r in records])
        bins.append({"low_km": low, "high_km": high if np.isfinite(high) else None,
                     "n": len(records), "positive_n": int(y.sum()),
                     "negative_n": int((~y).sum()),
                     "methods": {method: float(roc_auc_score(y, [r[method] for r in records]))
                                 if len(set(y)) == 2 else None for method in METHODS}})
    return {"minimum_km": float(distances.min()), "median_km": float(np.median(distances)),
            "maximum_km": float(distances.max()), "bins": bins}


def choose(records: list[dict], method: str) -> float:
    y = np.asarray([r["endpoint"] == "agriculture" for r in records])
    score = np.asarray([r[method] for r in records])
    fpr, tpr, threshold = roc_curve(y, score)
    valid = np.flatnonzero(fpr <= .1)
    return float(threshold[valid[np.argmax(tpr[valid])]])


def threshold_bootstrap(validation: list[dict], test: list[dict], stable: list[dict],
                        iterations: int = 2000) -> dict:
    validation = [r for r in validation if r["endpoint"] != "stable_forest"]
    test_loss = [r for r in test if r["endpoint"] != "stable_forest"]
    blocks = sorted({r["block_id"] for r in validation})
    lookup = {block: [r for r in validation if r["block_id"] == block] for block in blocks}
    rng = np.random.default_rng(SEED)
    draws = {method: {"threshold": [], "test_recall": [], "test_loss_fpr": [],
                      "stable_agriculture_fpr": []} for method in METHODS}
    for _ in range(iterations):
        sample = [record for block in rng.choice(blocks, len(blocks), replace=True)
                  for record in lookup[block]]
        if len({r["endpoint"] == "agriculture" for r in sample}) < 2:
            continue
        for method in METHODS:
            threshold = choose(sample, method)
            positive = [r[method] >= threshold for r in test_loss if r["endpoint"] == "agriculture"]
            negative = [r[method] >= threshold for r in test_loss if r["endpoint"] != "agriculture"]
            stable_flags = [float(r[method] >= threshold) for r in stable]
            draws[method]["threshold"].append(threshold)
            draws[method]["test_recall"].append(float(np.mean(positive)))
            draws[method]["test_loss_fpr"].append(float(np.mean(negative)))
            draws[method]["stable_agriculture_fpr"].append(float(np.mean(stable_flags)))
    return {method: {quantity: {"median": float(np.median(values)),
                                "interval95": np.percentile(values, [2.5, 97.5]).tolist()}
                     for quantity, values in quantities.items()}
            for method, quantities in draws.items()}


def main() -> None:
    validation = read(OUT / "validation_predictions.csv")
    test = read(OUT / "test_predictions.csv")
    stable = read(OUT / "stable_agriculture_predictions.csv")
    loss = [r for r in test if r["endpoint"] != "stable_forest"]
    report = {
        "class_by_state_and_split": {},
        "nearest_training_distance": add_distances(test),
        "coarse_spatial_bootstrap": [coarse_bootstrap(loss, 50), coarse_bootstrap(loss, 100)],
        "postclassification_minus_spectral_forest": paired_loss_contrast(
            loss, "postclassification", "spectral_forest_bitemporal"),
        "threshold_selection_bootstrap": threshold_bootstrap(validation, test, stable),
        "false_positives_by_endpoint": {},
    }
    with (SOURCE / "reference.csv").open() as stream:
        reference = [r for r in csv.DictReader(stream) if r["reference_label"] == "accepted"]
    for split in ("train", "validation", "test"):
        report["class_by_state_and_split"][split] = {
            region: dict(Counter(r["endpoint"] for r in reference
                                 if r["split"] == split and r["region"] == region))
            for region in sorted({r["region"] for r in reference})}
    thresholds = json.loads((OUT / "operating_points.json").read_text())["thresholds"]
    for method in METHODS:
        report["false_positives_by_endpoint"][method] = {
            endpoint: sum(r[method] >= thresholds[method]["threshold"] for r in test
                          if r["endpoint"] == endpoint)
            for endpoint in ("mining", "secondary_forest", "stable_forest")}
    (OUT / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"nearest_training_distance": report["nearest_training_distance"],
                      "false_positives_by_endpoint": report["false_positives_by_endpoint"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
