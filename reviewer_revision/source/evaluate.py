"""Freeze added validation thresholds and evaluate all models on Dataset C."""
from __future__ import annotations

import csv
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
SOURCE = BENCH / "results/last_effort"
OUT = BENCH / "results/reviewer_revision"
DATA = BENCH / "data/processed/last_effort"
PAIRED = BENCH / "data/processed/reviewer_revision/features_2020"
sys.path.insert(0, str(HERE))
from models import chip_arrays, rows, sha256  # noqa: E402

SEED = 20261015
METHODS = [
    "joint_original", "endpoint_terrafm", "postclassification",
    "bitemporal_terrafm", "spectral_linear_endpoint",
    "spectral_linear_joint", "spectral_forest_bitemporal",
    "change_original", "optical_index",
]
LABELS = {
    "joint_original": "TerraFM endpoint × inherited change",
    "endpoint_terrafm": "TerraFM endpoint only",
    "postclassification": "TerraFM post-classification",
    "bitemporal_terrafm": "Direct bitemporal TerraFM",
    "spectral_linear_endpoint": "Linear spectral endpoint (500-epoch cap)",
    "spectral_linear_joint": "Linear spectral endpoint × change (500-epoch cap)",
    "spectral_forest_bitemporal": "Bitemporal spectral random forest",
    "change_original": "Inherited change head",
    "optical_index": "Optical-index change",
}


def top9(values: np.ndarray) -> float:
    flat = np.asarray(values).reshape(-1)
    return float(np.sort(flat)[-min(9, len(flat)):].mean())


def load_linear(path: Path, dimensions: int) -> tuple[torch.nn.Module, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = torch.nn.Linear(dimensions, 4)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state


def probability(model: torch.nn.Module, state: dict, x: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        return model((tensor - state["mean"]) / state["std"]).softmax(-1).numpy()


def event_diagnostics(window_id: str, agri: np.ndarray, joint: np.ndarray) -> dict[str, float]:
    with np.load(DATA / "labels" / f"{window_id}.npz") as saved:
        before, after = saved["before"], saved["after"]
    prior_agri = np.isin(before, [3, 4])
    event = (before == 1) & np.isin(after, [3, 4])
    event_grid = F.interpolate(torch.from_numpy(event.astype(np.float32))[None, None],
                               size=(14, 14), mode="area")[0, 0].numpy()
    top_agri = np.argpartition(agri.reshape(-1), -9)[-9:]
    top_joint = np.argpartition(joint.reshape(-1), -9)[-9:]
    inside = event_grid.reshape(-1) >= .5
    outside = ~inside
    return {
        "mapped_prior_agriculture_fraction_2020": float(prior_agri.mean()),
        "mapped_agriculture_transition_fraction": float(event.mean()),
        "endpoint_top9_mean_event_fraction": float(event_grid.reshape(-1)[top_agri].mean()),
        "joint_top9_mean_event_fraction": float(event_grid.reshape(-1)[top_joint].mean()),
        "endpoint_score_inside_event": top9(agri.reshape(-1)[inside]) if inside.any() else float("nan"),
        "endpoint_score_outside_event": top9(agri.reshape(-1)[outside]) if outside.any() else float("nan"),
    }


def predict(split: str) -> list[dict]:
    token_head = load_linear(SOURCE / "tokens_landuse_head.pt", 768)
    spectral_head = load_linear(OUT / "spectral_landuse_converged.pt", 14)
    with (OUT / "bitemporal_terrafm.pkl").open("rb") as stream:
        temporal_model = pickle.load(stream)
    with (OUT / "bitemporal_spectral_forest.pkl").open("rb") as stream:
        forest = pickle.load(stream)
    original = {row["window_id"]: row for row in csv.DictReader(
        (SOURCE / f"{split}_predictions.csv").open())}
    output = []
    for row in rows(split):
        wid = row["window_id"]
        with np.load(DATA / "features" / f"{wid}.npz") as saved:
            t24 = saved["tokens"].astype(np.float32).reshape(768, -1).T
            s24 = saved["spectral"].astype(np.float32).reshape(14, -1).T
            change = saved["change"].reshape(-1).astype(np.float32)
            optical = float(saved["optical_fraction"])
        with np.load(PAIRED / f"{wid}.npz") as saved:
            t20 = saved["tokens"].astype(np.float32).reshape(768, -1).T
        p20 = probability(*token_head, t20)
        p24 = probability(*token_head, t24)
        spectral_agri = probability(*spectral_head, s24)[:, 2]
        temporal, spectral, _ = chip_arrays(row)
        direct = temporal_model.predict_proba(temporal)[:, 1]
        conventional = forest.predict_proba(spectral)[:, 1]
        agri = p24[:, 2]
        joint = agri * change
        record = {
            **row,
            "joint_original": top9(joint),
            "endpoint_terrafm": top9(agri),
            "postclassification": top9(p20[:, 0] * agri),
            "bitemporal_terrafm": top9(direct),
            "spectral_linear_endpoint": top9(spectral_agri),
            "spectral_linear_joint": top9(spectral_agri * change),
            "spectral_forest_bitemporal": top9(conventional),
            "change_original": top9(change),
            "optical_index": optical,
        }
        # Assert the untouched original values are reproduced by the independent pass.
        source = original[wid]
        for new, old in (("joint_original", "joint_top9"),
                         ("endpoint_terrafm", "landuse_top9"),
                         ("change_original", "change_top9"),
                         ("optical_index", "optical_fraction")):
            if not np.isclose(record[new], float(source[old]), atol=2e-6):
                raise ValueError(f"Original score mismatch for {wid}: {new}")
        record.update(event_diagnostics(wid, agri, joint))
        output.append(record)
    return output


def choose_threshold(records: list[dict], method: str) -> dict:
    relevant = [r for r in records if r["endpoint"] != "stable_forest"]
    y = np.asarray([r["endpoint"] == "agriculture" for r in relevant])
    score = np.asarray([r[method] for r in relevant])
    fpr, tpr, thresholds = roc_curve(y, score)
    valid = np.flatnonzero(fpr <= .1)
    best = valid[np.argmax(tpr[valid])]
    return {"threshold": float(thresholds[best]), "validation_recall": float(tpr[best]),
            "validation_fpr": float(fpr[best]), "positive_n": int(y.sum()),
            "negative_n": int((~y).sum())}


def wilson(k: int, n: int) -> list[float | None]:
    if not n:
        return [None, None]
    z, p = 1.96, k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [centre - half, centre + half]


def metrics(records: list[dict], thresholds: dict) -> dict:
    y = np.asarray([r["endpoint"] == "agriculture" for r in records])
    result = {"n": len(y), "positive_n": int(y.sum()), "negative_n": int((~y).sum()),
              "methods": {}}
    for method in METHODS:
        score = np.asarray([r[method] for r in records])
        prediction = score >= thresholds[method]["threshold"]
        tp, fn = int((y & prediction).sum()), int((y & ~prediction).sum())
        fp, tn = int((~y & prediction).sum()), int((~y & ~prediction).sum())
        recall = tp / (tp + fn) if tp + fn else None
        fpr = fp / (fp + tn) if fp + tn else None
        result["methods"][method] = {
            "label": LABELS[method],
            "auroc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, score)) if y.any() else None,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": recall, "recall_ci95_wilson": wilson(tp, tp + fn),
            "fpr": fpr, "fpr_ci95_wilson": wilson(fp, fp + tn),
            "accuracy": float((prediction == y).mean()),
            "balanced_accuracy": (recall + 1 - fpr) / 2 if recall is not None and fpr is not None else None,
        }
    return result


def block_bootstrap(records: list[dict], iterations: int = 4000) -> dict:
    blocks = sorted({r["block_id"] for r in records})
    lookup = {block: [i for i, r in enumerate(records) if r["block_id"] == block]
              for block in blocks}
    y = np.asarray([r["endpoint"] == "agriculture" for r in records])
    scores = {method: np.asarray([r[method] for r in records]) for method in METHODS}
    draws = {method: [] for method in METHODS}
    contrasts = {f"{a}_minus_{b}": [] for a, b in [
        ("joint_original", "change_original"),
        ("endpoint_terrafm", "joint_original"),
        ("endpoint_terrafm", "spectral_linear_endpoint"),
        ("bitemporal_terrafm", "spectral_forest_bitemporal"),
        ("postclassification", "spectral_forest_bitemporal"),
    ]}
    rng = np.random.default_rng(SEED)
    for _ in range(iterations):
        indices = np.asarray([i for b in rng.choice(blocks, len(blocks), replace=True)
                              for i in lookup[b]])
        if len(set(y[indices])) < 2:
            continue
        values = {method: roc_auc_score(y[indices], score[indices])
                  for method, score in scores.items()}
        for method, value in values.items():
            draws[method].append(value)
        for key in contrasts:
            a, b = key.split("_minus_")
            contrasts[key].append(values[a] - values[b])
    def summary(values: list[float]) -> dict:
        return {"ci95": np.percentile(values, [2.5, 97.5]).tolist(),
                "valid_draws": len(values)}
    return {"methods": {key: summary(value) for key, value in draws.items()},
            "contrasts": {key: summary(value) for key, value in contrasts.items()},
            "unit": "original 10 km block", "iterations": iterations}


def mcnemar(records: list[dict], thresholds: dict, first: str, second: str) -> dict:
    y = np.asarray([r["endpoint"] == "agriculture" for r in records])
    a = np.asarray([r[first] >= thresholds[first]["threshold"] for r in records]) == y
    b = np.asarray([r[second] >= thresholds[second]["threshold"] for r in records]) == y
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    p_value = float(binomtest(min(a_only, b_only), a_only + b_only, .5).pvalue) if a_only + b_only else 1.0
    return {"first": first, "second": second, "first_only_correct": a_only,
            "second_only_correct": b_only, "exact_two_sided_p": p_value}


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=list(dict.fromkeys(
            key for record in records for key in record)))
        writer.writeheader(); writer.writerows(records)


def main() -> None:
    torch.set_num_threads(8)
    validation = predict("validation")
    thresholds = {method: choose_threshold(validation, method) for method in METHODS}
    manifest = {
        "seed": SEED,
        "selection": "Maximum validation recall under empirical nonagricultural-loss FPR <= 10%",
        "protocol_sha256": sha256(HERE / "PROTOCOL.md"),
        "thresholds": thresholds,
    }
    # Persist the operating points before loading/predicting the test records.
    (OUT / "operating_points.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_csv(OUT / "validation_predictions.csv", validation)
    test = predict("test")
    write_csv(OUT / "test_predictions.csv", test)
    groups = {
        "loss_endpoints": [r for r in test if r["endpoint"] != "stable_forest"],
        "loss_and_stable_forest": test,
        "agriculture_vs_mining": [r for r in test if r["endpoint"] in ("agriculture", "mining")],
        "agriculture_vs_secondary_forest": [r for r in test if r["endpoint"] in ("agriculture", "secondary_forest")],
        "para_loss_endpoints": [r for r in test if r["region"] == "brazil_para" and r["endpoint"] != "stable_forest"],
        "matogrosso_loss_endpoints": [r for r in test if r["region"] == "brazil_matogrosso" and r["endpoint"] != "stable_forest"],
    }
    results = {name: metrics(records, thresholds) for name, records in groups.items()}
    for name, records in groups.items():
        if min(results[name]["positive_n"], results[name]["negative_n"]) >= 5:
            results[name]["block_bootstrap"] = block_bootstrap(records)
    loss = groups["loss_endpoints"]
    state_score = np.asarray([r["region"] == "brazil_matogrosso" for r in loss])
    y = np.asarray([r["endpoint"] == "agriculture" for r in loss])
    geography = {"state_only_auroc": float(roc_auc_score(y, state_score)),
                 "class_by_state": {region: dict(Counter(r["endpoint"] for r in test if r["region"] == region))
                                    for region in sorted({r["region"] for r in test})}}
    report = {
        "protocol_sha256": sha256(HERE / "PROTOCOL.md"),
        "operating_points_sha256": sha256(OUT / "operating_points.json"),
        "results": results,
        "geography": geography,
        "mcnemar": [
            mcnemar(loss, thresholds, "joint_original", "spectral_linear_joint"),
            mcnemar(loss, thresholds, "endpoint_terrafm", "spectral_linear_endpoint"),
            mcnemar(loss, thresholds, "bitemporal_terrafm", "spectral_forest_bitemporal"),
        ],
        "event_diagnostics": {
            field: {endpoint: {
                "n": len(values), "finite_n": len(finite),
                "median": float(np.median(finite)) if finite else None,
                "q25": float(np.percentile(finite, 25)) if finite else None,
                "q75": float(np.percentile(finite, 75)) if finite else None}
                for endpoint in sorted({r["endpoint"] for r in test})
                if (values := [float(r[field]) for r in test if r["endpoint"] == endpoint])
                if (finite := [value for value in values if np.isfinite(value)]) or values}
            for field in ["mapped_prior_agriculture_fraction_2020",
                          "mapped_agriculture_transition_fraction",
                          "endpoint_top9_mean_event_fraction",
                          "joint_top9_mean_event_fraction",
                          "endpoint_score_inside_event", "endpoint_score_outside_event"]
        },
    }
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps({
        "loss_endpoint_aurocs": {method: results["loss_endpoints"]["methods"][method]["auroc"]
                                 for method in METHODS},
        "para_aurocs": {method: results["para_loss_endpoints"]["methods"][method]["auroc"]
                        for method in METHODS},
        "state_only_auroc": geography["state_only_auroc"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
