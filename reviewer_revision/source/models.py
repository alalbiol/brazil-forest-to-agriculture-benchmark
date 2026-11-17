"""Fit reviewer-requested baselines using training/validation data only."""
from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
SOURCE = BENCH / "results/last_effort"
OUT = BENCH / "results/reviewer_revision"
DATA = BENCH / "data/processed/last_effort"
PAIRED = BENCH / "data/processed/reviewer_revision/features_2020"
SEED = 20261015
PRIMARY, SECONDARY, PASTURE, CROP, MINING = 1, 2, 3, 4, 10
CLASSES = ["primary_forest", "secondary_forest", "agriculture", "mining"]


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(split: str) -> list[dict[str, str]]:
    with (SOURCE / "reference.csv").open() as stream:
        return [row for row in csv.DictReader(stream)
                if row["reference_label"] == "accepted" and row["split"] == split]


def area_fraction(mask: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(tensor, size=(14, 14), mode="area")[0, 0].numpy()


def transition_labels(window_id: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(DATA / "labels" / f"{window_id}.npz") as saved:
        before, after = saved["before"], saved["after"]
    primary = area_fraction(before == PRIMARY)
    after_fractions = np.stack([
        area_fraction(after == PRIMARY), area_fraction(after == SECONDARY),
        area_fraction(np.isin(after, [PASTURE, CROP])), area_fraction(after == MINING)])
    after_label = after_fractions.argmax(axis=0)
    valid = (primary >= .8) & (after_fractions.max(axis=0) >= .8)
    target = (after_label == 2).astype(np.int64)
    return target.reshape(-1), valid.reshape(-1)


def indices(x: np.ndarray) -> np.ndarray:
    # Historical name retained for exact reproduction of the fitted forest.
    # x contains standardized bands: these are bounded band contrasts, NOT
    # physical-reflectance NDVI/NDMI/NBR. Do not change without refitting.
    # Bands: VV, VH, B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12.
    def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a - b) / (np.abs(a) + np.abs(b) + 1e-6)
    red, nir, swir1, swir2 = x[:, 5], x[:, 9], x[:, 12], x[:, 13]
    return np.column_stack([ratio(nir, red), ratio(nir, swir1), ratio(nir, swir2)])


def chip_arrays(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wid = row["window_id"]
    with np.load(DATA / "features" / f"{wid}.npz") as current:
        t24 = current["tokens"].astype(np.float32).reshape(768, -1).T
        s24 = current["spectral"].astype(np.float32).reshape(14, -1).T
        land_labels = current["token_labels"].reshape(-1).astype(np.int64)
    with np.load(PAIRED / f"{wid}.npz") as prior:
        t20 = prior["tokens"].astype(np.float32).reshape(768, -1).T
        s20 = prior["spectral"].astype(np.float32).reshape(14, -1).T
    temporal = np.concatenate([t20, t24, t24 - t20, np.abs(t24 - t20)], axis=1)
    spectral = np.concatenate([s20, s24, s24 - s20, np.abs(s24 - s20),
                               indices(s20), indices(s24), indices(s24) - indices(s20)], axis=1)
    target, valid = transition_labels(wid)
    return temporal, spectral, np.column_stack([target, valid, land_labels])


def load_split(split: str) -> dict[str, np.ndarray]:
    terra, spectral, target, land = [], [], [], []
    chip = []
    for index, row in enumerate(rows(split)):
        tx, sx, labels = chip_arrays(row)
        valid = labels[:, 1].astype(bool)
        terra.append(tx[valid]); spectral.append(sx[valid]); target.append(labels[valid, 0])
        land_valid = labels[:, 2] >= 0
        land.append((sx[land_valid, 14:28], labels[land_valid, 2]))
        chip.extend([index] * int(valid.sum()))
    return {
        "terra": np.concatenate(terra), "spectral": np.concatenate(spectral),
        "target": np.concatenate(target).astype(np.int64), "chip": np.asarray(chip),
        "land_x": np.concatenate([pair[0] for pair in land]),
        "land_y": np.concatenate([pair[1] for pair in land]).astype(np.int64),
    }


def fit_spectral(train: dict[str, np.ndarray], validation: dict[str, np.ndarray]) -> dict:
    torch.manual_seed(SEED)
    x = torch.from_numpy(train["land_x"])
    y = torch.from_numpy(train["land_y"])
    vx = torch.from_numpy(validation["land_x"])
    vy = torch.from_numpy(validation["land_y"])
    mean, std = x.mean(0), x.std(0).clamp_min(.05)
    x, vx = (x - mean) / std, (vx - mean) / std
    counts = torch.bincount(y, minlength=4)
    weights = counts.float().sum().div(counts.float()).sqrt()
    weights /= weights.mean()
    model = torch.nn.Linear(14, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best, stale, history = float("inf"), 0, []
    path = OUT / "spectral_landuse_converged.pt"
    for epoch in range(1, 501):
        model.train()
        order = torch.randperm(len(y))
        total = 0.0
        for batch in order.split(4096):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x[batch]), y[batch], weight=weights)
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(batch)
        model.eval()
        with torch.no_grad():
            value = float(F.cross_entropy(model(vx), vy, weight=weights,
                                          reduction="sum") / weights[vy].sum())
        history.append({"epoch": epoch, "training_loss": total / len(y),
                        "validation_loss": value})
        if value < best - 1e-5:
            best, stale = value, 0
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                        "classes": CLASSES, "epoch": epoch, "validation_loss": best,
                        "seed": SEED}, path)
        else:
            stale += 1
        if epoch >= 100 and stale >= 30:
            break
    return {"best_validation_loss": best, "epochs_run": len(history),
            "selected_epoch": torch.load(path, weights_only=False)["epoch"],
            "history": history, "checkpoint_sha256": sha256(path)}


def fit_logistic(train: dict[str, np.ndarray], validation: dict[str, np.ndarray]) -> dict:
    candidates = []
    for c_value in (.01, .1, 1.0, 10.0):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, class_weight="balanced", solver="lbfgs",
                               max_iter=300, random_state=SEED, n_jobs=8),
        )
        model.fit(train["terra"], train["target"])
        probability = model.predict_proba(validation["terra"])[:, 1]
        candidates.append((log_loss(validation["target"], probability), c_value, model))
        print("bitemporal C", c_value, "validation log loss", candidates[-1][0], flush=True)
    candidates.sort(key=lambda item: (round(item[0], 12), item[1]))
    loss, c_value, model = candidates[0]
    with (OUT / "bitemporal_terrafm.pkl").open("wb") as stream:
        pickle.dump(model, stream)
    return {"selected_C": c_value, "validation_log_loss": loss,
            "candidates": [{"C": c, "validation_log_loss": l} for l, c, _ in candidates],
            "checkpoint_sha256": sha256(OUT / "bitemporal_terrafm.pkl")}


def fit_forest(train: dict[str, np.ndarray], validation: dict[str, np.ndarray]) -> dict:
    candidates = []
    for depth in (6, 12, None):
        for leaf in (2, 10):
            model = RandomForestClassifier(
                n_estimators=500, max_depth=depth, min_samples_leaf=leaf,
                class_weight="balanced", random_state=SEED, n_jobs=8,
                max_features="sqrt", oob_score=False,
            )
            model.fit(train["spectral"], train["target"])
            probability = model.predict_proba(validation["spectral"])[:, 1]
            loss = log_loss(validation["target"], probability)
            candidates.append((loss, depth, leaf, model))
            print("forest", depth, leaf, "validation log loss", loss, flush=True)
    rank_depth = {6: 0, 12: 1, None: 2}
    candidates.sort(key=lambda item: (round(item[0], 12), rank_depth[item[1]], -item[2]))
    loss, depth, leaf, model = candidates[0]
    with (OUT / "bitemporal_spectral_forest.pkl").open("wb") as stream:
        pickle.dump(model, stream)
    return {"selected_max_depth": depth, "selected_min_samples_leaf": leaf,
            "validation_log_loss": loss,
            "candidates": [{"max_depth": d, "min_samples_leaf": l,
                            "validation_log_loss": v} for v, d, l, _ in candidates],
            "feature_importances": model.feature_importances_.tolist(),
            "checkpoint_sha256": sha256(OUT / "bitemporal_spectral_forest.pkl")}


def main() -> None:
    np.random.seed(SEED)
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_split("train")
    validation = load_split("validation")
    print("training transitions", Counter(train["target"]), train["terra"].shape, flush=True)
    print("validation transitions", Counter(validation["target"]), validation["terra"].shape, flush=True)
    report = {
        "seed": SEED,
        "protocol_sha256": sha256(HERE / "PROTOCOL.md"),
        "source_reference_sha256": sha256(SOURCE / "reference.csv"),
        "training_transition_counts": dict(Counter(map(int, train["target"]))),
        "validation_transition_counts": dict(Counter(map(int, validation["target"]))),
        "spectral_linear": fit_spectral(train, validation),
        "bitemporal_terrafm": fit_logistic(train, validation),
        "bitemporal_spectral_forest": fit_forest(train, validation),
    }
    (OUT / "model_selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUT / "model_selection.json", flush=True)


if __name__ == "__main__":
    main()
