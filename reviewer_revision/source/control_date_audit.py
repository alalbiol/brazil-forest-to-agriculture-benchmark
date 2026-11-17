"""Audit the 2024 land-use head on 2020 validation and farm-control imagery.

Local mode builds control label grids from the original TerraClass vectors.
Bundle mode uses the supplied grids, features, and fitted head; neither mode
refits a model or changes the published test decisions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BENCH = Path(__file__).resolve().parent.parent
CLASSES = ("primary_forest", "secondary_forest", "agriculture", "mining")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def patch_labels(mapped: np.ndarray) -> np.ndarray:
    masks = np.stack((mapped == 1, mapped == 2,
                      np.isin(mapped, (3, 4)), mapped == 10)).astype(np.float32)
    fractions = F.interpolate(torch.from_numpy(masks)[None], size=(14, 14),
                              mode="area")[0].numpy()
    labels = fractions.argmax(axis=0).astype(np.int8)
    labels[fractions.max(axis=0) < 0.8] = -1
    return labels.ravel()


def probabilities(model: torch.nn.Module, state: dict, tokens: np.ndarray) -> np.ndarray:
    values = torch.from_numpy(tokens.astype(np.float32).reshape(768, -1).T)
    with torch.no_grad():
        return model((values - state["mean"]) / state["std"]).softmax(-1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path,
                        help="Extracted reproduction package, including control label grids")
    args = parser.parse_args()
    bundle = args.bundle.resolve() if args.bundle else None
    validation_ref = (bundle / "published/dataset_c/reference.csv" if bundle else
                      BENCH / "results/last_effort/reference.csv")
    control_ref = (bundle / "reviewer_revision/results/stable_agriculture_reference.csv"
                   if bundle else BENCH / "results/reviewer_revision/stable_agriculture_reference.csv")
    validation_labels = (bundle / "cache/labels" if bundle else
                         BENCH / "data/processed/last_effort/labels")
    validation_features = (bundle / "cache/features_2020" if bundle else
                           BENCH / "data/processed/reviewer_revision/features_2020")
    control_labels = (bundle / "cache/stable_agriculture/labels" if bundle else
                      BENCH / "data/processed/reviewer_revision/stable_agriculture/labels")
    control_features = (bundle / "cache/stable_agriculture/features" if bundle else
                        BENCH / "data/processed/reviewer_revision/stable_agriculture/features")
    head = (bundle / "published/dataset_c/tokens_landuse_head.pt" if bundle else
            BENCH / "results/last_effort/tokens_landuse_head.pt")
    output = (bundle / "rerun/control_date_audit.json" if bundle else
              BENCH / "results/reviewer_revision/control_date_audit.json")

    state = torch.load(head, map_location="cpu", weights_only=False)
    model = torch.nn.Linear(768, 4)
    model.load_state_dict(state["state_dict"])
    model.eval()
    validation = [r for r in read_csv(validation_ref)
                  if r["reference_label"] == "accepted" and r["split"] == "validation"]
    controls = read_csv(control_ref)
    assert len(validation) == 136 and len(controls) == 100
    matrices = {part: np.zeros((4, 4), dtype=np.int64)
                for part in ("validation", "established_agriculture")}
    farm_forest_probabilities = []
    farm_chip_top9 = []

    if bundle is None:
        # These imports need local TerraClass vectors and are never needed for
        # reproduction from the distributed cache.
        import sys
        sys.path.insert(0, str(BENCH))
        sys.path.insert(0, str(BENCH / "last_effort"))
        from build_reference import window_bbox
        from fast_maps import grid
        control_labels.mkdir(parents=True, exist_ok=True)

    for part, rows in (("validation", validation),
                       ("established_agriculture", controls)):
        for row in rows:
            wid = row["window_id"]
            if part == "validation":
                with np.load(validation_labels / f"{wid}.npz") as saved:
                    mapped = saved["before"].copy()
                with np.load(validation_features / f"{wid}.npz") as saved:
                    tokens = saved["tokens"].copy()
            else:
                label_path = control_labels / f"{wid}.npz"
                if bundle is None:
                    bbox = window_bbox(float(row["lon"]), float(row["lat"]), 440)
                    mapped, _ = grid(row["region"], 2020, bbox, 44)
                    fraction = float(np.isin(mapped, (3, 4)).mean())
                    assert abs(fraction - float(row["agriculture_fraction_2020"])) < 1e-6, wid
                    np.savez_compressed(label_path, before=mapped)
                else:
                    with np.load(label_path) as saved:
                        mapped = saved["before"].copy()
                with np.load(control_features / f"{wid}.npz") as saved:
                    tokens = saved["tokens_2020"].copy()
            labels = patch_labels(mapped)
            probabilities_2020 = probabilities(model, state, tokens)
            predictions = probabilities_2020.argmax(axis=1)
            for true_class in range(4):
                selected = labels == true_class
                matrices[part][true_class] += np.bincount(predictions[selected], minlength=4)
            if part == "established_agriculture":
                farm = labels == 2
                farm_forest_probabilities.extend(probabilities_2020[farm, 0].tolist())
                farm_chip_top9.append(float(np.sort(probabilities_2020[:, 0])[-9:].mean()))

    def describe(matrix: np.ndarray) -> dict:
        return {"classes": CLASSES, "matrix_rows_reference_columns_prediction": matrix.tolist(),
                "eligible_class_counts": matrix.sum(axis=1).tolist(),
                "eligible_total": int(matrix.sum()),
                "agreement": float(np.trace(matrix) / matrix.sum())}

    farm_probs = np.asarray(farm_forest_probabilities)
    top9 = np.asarray(farm_chip_top9)
    result = {"validation_2020": describe(matrices["validation"]),
              "established_agriculture_2020": describe(matrices["established_agriculture"]),
              "control_mapped_agriculture_forest_probability": {
                  "n": int(farm_probs.size),
                  "median": float(np.median(farm_probs)),
                  "p90": float(np.quantile(farm_probs, 0.9)),
                  "p99": float(np.quantile(farm_probs, 0.99)),
                  "fraction_at_least_0_5": float(np.mean(farm_probs >= 0.5))},
              "control_chip_top9_2020_forest_probability": {
                  "n": int(top9.size), "median": float(np.median(top9)),
                  "p90": float(np.quantile(top9, 0.9)),
                  "max": float(np.max(top9))}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
