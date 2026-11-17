"""Reproduce supplemental date-transfer and top-k checks from cached features."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

BENCH = Path(__file__).resolve().parent.parent
DATA = BENCH / "data/processed/last_effort"
REVIEW = BENCH / "data/processed/reviewer_revision"
RESULTS = BENCH / "results/reviewer_revision"
K_VALUES = (1, 5, 9, 20)


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def probabilities(model, state, values):
    x = torch.from_numpy(values.astype(np.float32).reshape(768, -1).T)
    with torch.no_grad():
        return model((x - state["mean"]) / state["std"]).softmax(-1).numpy()


def labels_from_map(mapped):
    indicators = np.stack([mapped == 1, mapped == 2,
                           np.isin(mapped, [3, 4]), mapped == 10])
    fractions = F.interpolate(torch.from_numpy(indicators.astype(np.float32))[None],
                              size=(14, 14), mode="area")[0].numpy()
    labels = fractions.argmax(axis=0).reshape(-1)
    labels[fractions.max(axis=0).reshape(-1) < .8] = -1
    return labels


def topk(values, k):
    return float(np.sort(values.reshape(-1))[-k:].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path,
                        help="Extracted reproduction package; writes under rerun/")
    args = parser.parse_args()
    bundle = args.bundle.resolve() if args.bundle else None
    features24 = bundle / "cache/features" if bundle else DATA / "features"
    label_dir = bundle / "cache/labels" if bundle else DATA / "labels"
    features20 = bundle / "cache/features_2020" if bundle else REVIEW / "features_2020"
    stable_features = (bundle / "cache/stable_agriculture/features" if bundle else
                       REVIEW / "stable_agriculture/features")
    results = bundle / "reviewer_revision/results" if bundle else RESULTS
    head = (bundle / "published/dataset_c/tokens_landuse_head.pt" if bundle else
            BENCH / "results/last_effort/tokens_landuse_head.pt")
    reference_path = (bundle / "published/dataset_c/reference.csv" if bundle else
                      BENCH / "results/last_effort/reference.csv")
    torch.set_num_threads(4)
    state = torch.load(head, map_location="cpu", weights_only=False)
    model = torch.nn.Linear(768, 4)
    model.load_state_dict(state["state_dict"])
    model.eval()
    reference = {r["window_id"]: r for r in read(reference_path)
                 if r["reference_label"] == "accepted"}
    predictions = {name: read(results / f"{name}_predictions.csv")
                   for name in ("validation", "test", "stable_agriculture")}
    patch_counts = {date: {"correct": 0, "valid": 0, "forest_correct": 0,
                           "forest_valid": 0} for date in (2020, 2024)}
    scores = {part: {k: [] for k in K_VALUES} for part in predictions}
    for part, rows in predictions.items():
        for row in rows:
            wid = row["window_id"]
            if part == "stable_agriculture":
                with np.load(stable_features / f"{wid}.npz") as cache:
                    token20 = cache["tokens_2020"].copy()
                    token24 = cache["tokens_2024"].copy()
            else:
                assert reference[wid]["split"] == part
                with np.load(features20 / f"{wid}.npz") as cache:
                    token20 = cache["tokens"].copy()
                with np.load(features24 / f"{wid}.npz") as cache:
                    token24 = cache["tokens"].copy()
                    cached_labels24 = cache["token_labels"].reshape(-1).copy()
            p20 = probabilities(model, state, token20)
            p24 = probabilities(model, state, token24)
            if part == "validation":
                with np.load(label_dir / f"{wid}.npz") as cache:
                    label20 = labels_from_map(cache["before"])
                    label24 = labels_from_map(cache["after"])
                assert np.array_equal(label24, cached_labels24)
                for date, labels, probs in ((2020, label20, p20), (2024, label24, p24)):
                    valid = labels >= 0
                    forest = labels == 0
                    tally = patch_counts[date]
                    predicted = probs.argmax(axis=1)
                    tally["correct"] += int((predicted[valid] == labels[valid]).sum())
                    tally["valid"] += int(valid.sum())
                    tally["forest_correct"] += int((predicted[forest] == 0).sum())
                    tally["forest_valid"] += int(forest.sum())
            products = p20[:, 0] * p24[:, 2]
            for k in K_VALUES:
                scores[part][k].append(topk(products, k))
            if part != "validation":
                assert np.isclose(scores[part][9][-1], float(row["postclassification"]), atol=2e-6)
    out = {"date_agreement": {}, "topk": {}}
    for date, tally in patch_counts.items():
        out["date_agreement"][str(date)] = {**tally,
            "accuracy": tally["correct"] / tally["valid"],
            "forest_recall": tally["forest_correct"] / tally["forest_valid"]}
    validation = predictions["validation"]
    test = predictions["test"]
    stable = predictions["stable_agriculture"]
    for k in K_VALUES:
        y_val = np.array([r["endpoint"] == "agriculture" for r in validation])
        relevant = np.array([r["endpoint"] != "stable_forest" for r in validation])
        fpr, tpr, thresholds = roc_curve(y_val[relevant],
                                         np.array(scores["validation"][k])[relevant])
        allowed = np.flatnonzero(fpr <= .1)
        selected = allowed[np.argmax(tpr[allowed])]
        threshold = float(thresholds[selected])
        y_test = np.array([r["endpoint"] == "agriculture" for r in test])
        loss = np.array([r["endpoint"] != "stable_forest" for r in test])
        score_test = np.array(scores["test"][k])
        positive = score_test[y_test]
        negative = score_test[loss & ~y_test]
        controls = np.array(scores["stable_agriculture"][k])
        out["topk"][str(k)] = {
            "threshold": threshold,
            "loss_auroc": roc_auc_score(y_test[loss], score_test[loss]),
            "conversion_vs_established_auroc": roc_auc_score(
                np.r_[np.ones(len(positive)), np.zeros(len(controls))],
                np.r_[positive, controls]),
            "conversion_detected": int((positive >= threshold).sum()),
            "conversion_n": len(positive),
            "nonagricultural_loss_flagged": int((negative >= threshold).sum()),
            "nonagricultural_loss_n": len(negative),
            "established_flagged": int((controls >= threshold).sum()),
            "established_n": len(controls),
        }
    original_positives = [r for r in test if r["endpoint"] == "agriculture"]
    original_controls = stable
    for field in ("postclassification", "endpoint_terrafm"):
        records = original_positives + original_controls
        out.setdefault("conversion_vs_established_auroc", {})[field] = roc_auc_score(
            [1] * len(original_positives) + [0] * len(original_controls),
            [float(r[field]) for r in records])
    assert np.isclose(out["topk"]["9"]["conversion_vs_established_auroc"],
                      out["conversion_vs_established_auroc"]["postclassification"])
    target = bundle / "rerun/additional_checks.json" if bundle else RESULTS / "additional_checks.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
