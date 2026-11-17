"""Exploratory matched spectral post-classification and output-localization audit."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import evaluate as ev
import models

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results/reviewer_revision"
STABLE = HERE.parent / "data/processed/reviewer_revision/stable_agriculture/features"
METHOD = "spectral_postclassification"


def predict(split):
    spectral_head = ev.load_linear(OUT / "spectral_landuse_converged.pt", 14)
    token_head = ev.load_linear(models.SOURCE / "tokens_landuse_head.pt", 768)
    path = OUT / ("stable_agriculture_reference.csv" if split == "stable" else f"{split}_predictions.csv")
    with path.open() as f:
        records = list(csv.DictReader(f))
    output = []
    for r in records:
        wid = r["window_id"]
        if split == "stable":
            with np.load(STABLE / f"{wid}.npz") as z:
                s20, s24 = z["spectral_2020"], z["spectral_2024"]
        else:
            with np.load(models.PAIRED / f"{wid}.npz") as z:
                s20, t20 = z["spectral"], z["tokens"]
            with np.load(models.DATA / "features" / f"{wid}.npz") as z:
                s24, t24 = z["spectral"], z["tokens"]
        p20 = ev.probability(*spectral_head, s20.astype(np.float32).reshape(14, -1).T)
        p24 = ev.probability(*spectral_head, s24.astype(np.float32).reshape(14, -1).T)
        record = {"window_id": wid, "region": r["region"],
                  "endpoint": r.get("endpoint", "stable_agriculture"),
                  "block_id": r.get("block_id", ""),
                  METHOD: ev.top9(p20[:, 0] * p24[:, 2])}
        if split != "stable":
            a = ev.probability(*token_head, t20.astype(np.float32).reshape(768, -1).T)[:, 0]
            b = ev.probability(*token_head, t24.astype(np.float32).reshape(768, -1).T)[:, 2]
            assert abs(ev.top9(a*b) - float(r["postclassification"])) < 1e-6
            record["postclassification"] = ev.top9(a*b)
            with np.load(models.DATA / "labels" / f"{wid}.npz") as z:
                event = (z["before"] == 1) & np.isin(z["after"], [3, 4])
            fractions = F.interpolate(torch.from_numpy(event.astype(np.float32))[None, None],
                                      size=(14, 14), mode="area").numpy().reshape(-1)
            top = np.argpartition(a*b, -9)[-9:]
            record["postclassification_top9_event_fraction"] = float(fractions[top].mean())
        output.append(record)
    return output


def group(records, threshold):
    y = np.array([r["endpoint"] == "agriculture" for r in records])
    s = np.array([r[METHOD] for r in records]); p = s >= threshold
    tp, fn, fp, tn = [int(v.sum()) for v in (y&p, y&~p, ~y&p, ~y&~p)]
    return {"n": len(y), "auroc": float(roc_auc_score(y, s)) if y.any() and (~y).any() else None,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp/int(y.sum()) if y.any() else None,
            "fpr": fp/int((~y).sum()), "recall_ci95_wilson": ev.wilson(tp, int(y.sum())),
            "fpr_ci95_wilson": ev.wilson(fp, int((~y).sum()))}


def bootstrap(records):
    blocks = sorted({r["block_id"] for r in records})
    lookup = {b: [i for i,r in enumerate(records) if r["block_id"] == b] for b in blocks}
    y = np.array([r["endpoint"] == "agriculture" for r in records])
    s = np.array([r[METHOD] for r in records])
    t = np.array([r["postclassification"] for r in records])
    rng = np.random.default_rng(ev.SEED); draws, contrasts = [], []
    for _ in range(4000):
        ix = np.array([i for b in rng.choice(blocks, len(blocks), replace=True) for i in lookup[b]])
        if len(set(y[ix])) < 2: continue
        auc = roc_auc_score(y[ix], s[ix]); draws.append(auc)
        contrasts.append(roc_auc_score(y[ix], t[ix]) - auc)
    return {"ci95": np.percentile(draws, [2.5,97.5]).tolist(),
            "terrafm_minus_spectral_ci95": np.percentile(contrasts, [2.5,97.5]).tolist(),
            "valid_draws": len(draws)}


def main():
    torch.set_num_threads(4)
    validation = predict("validation")
    operating = ev.choose_threshold(validation, METHOD)
    test = predict("test"); stable = predict("stable")
    loss = [r for r in test if r["endpoint"] != "stable_forest"]
    groups = {"loss_endpoints": loss,
              "para_loss_endpoints": [r for r in loss if r["region"] == "brazil_para"],
              "stable_agriculture": stable}
    report = {"status": "exploratory; specified after prior extension results were inspected",
              "plan_sha256": models.sha256(HERE / "CRITICAL_AUDIT_PLAN.md"),
              "operating_point": operating,
              "results": {k: group(v, operating["threshold"]) for k,v in groups.items()},
              "bootstrap": bootstrap(loss)}
    overlap = [r["postclassification_top9_event_fraction"] for r in loss if r["endpoint"] == "agriculture"]
    report["postclassification_localization"] = {"n": len(overlap),
        "median": float(np.median(overlap)), "q25": float(np.percentile(overlap,25)),
        "q75": float(np.percentile(overlap,75))}
    for split, records in [("validation",validation),("test",test),("stable",stable)]:
        ev.write_csv(OUT / f"critical_audit_{split}_predictions.csv", records)
    (OUT / "critical_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
