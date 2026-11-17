---
name: brazil-forest-to-agriculture-benchmark
authors:
  - Alberto Albiol
  - Miquel Obrador-Reina
  - Juan B. Pedro
  - Albert Fernández
license: CC-BY-SA-4.0
source: https://github.com/alalbiol/brazil-forest-to-agriculture-benchmark
---

# Forest-to-agriculture screening benchmark: Pará and Mato Grosso, Brazil

**Reference labels, cached Sentinel-1/2 TerraFM features, fitted models and the code to reproduce the reported results.**

This repository accompanies the manuscript *Temporal Evidence in Forest-to-Agriculture Screening: A Benchmark in Pará and Mato Grosso, Brazil*, by Alberto Albiol, Miquel Obrador-Reina, Juan B. Pedro, and Albert Fernández.

**Publication status:** the manuscript is under revision. It will be linked here, or its arXiv version, after acceptance.

**Released asset:** [GitHub v1.0.0](https://github.com/alalbiol/brazil-forest-to-agriculture-benchmark/releases/tag/v1.0.0) is distributed as the EOTDL dataset [`brazil-forest-to-agriculture-benchmark`](https://www.eotdl.com/datasets/brazil-forest-to-agriculture-benchmark). The EOTDL version number is recorded in `provenance/releases/` once the upload has been verified.

**Applying the model to new imagery:** the backbone checkpoint, inference code and Earth Engine downloader are a separate asset: [forest-to-agriculture-inference](https://github.com/alalbiol/forest-to-agriculture-inference).

## GitHub and EOTDL

| | GitHub | EOTDL snapshot |
|---|---|---|
| Code, documentation, reference tables, predictions, thresholds, small fitted models | ✓ | ✓ |
| Cached TerraFM features, Sentinel composites and TerraClass label grids (`cache/`, 2,610 files) | | ✓ |
| Two-date spectral/radar random forest (215 MB) and IBGE biome archive | | ✓ |
| Size | ~9 MB | ~730 MB |

[`manifest.json`](manifest.json) lists every released data file with its SHA-256 hash; `location: eotdl` marks those stored only in the snapshot. The EOTDL snapshot is self-contained: it holds the full Git tree of the tagged release.

## What the study tests

EU deforestation screening has to show that the land was forest before the 2020 cutoff *and* became agriculture afterwards. A map of current agriculture alone does not show when the agriculture arrived. We built chips (440 m, 14 × 14 patches) from TerraClass 2020/2024 maps, dated MapBiomas Alerta polygons and annual Sentinel-1/2 composites. We then compared a 2024-only agriculture score with a temporal score that also requires primary forest in 2020. Both use a linear land-use classifier on frozen TerraFM features.

| Test | 2024-only TerraFM | Temporal TerraFM |
|---|---:|---:|
| Forest-loss test, AUROC (84 agriculture vs 29 mining/secondary forest, 10-km geographic blocks) | 0.964 | 0.963 |
| Locations mapped as agriculture in both 2020 and 2024 flagged as conversions | 97 / 100 | 3 / 100 |

A two-date spectral/radar random forest and an exploratory temporal spectral method are included for comparison. The results do **not** establish a TerraFM ranking advantage over conventional temporal methods. References derive from TerraClass and MapBiomas maps, not field observations. The sample is pasture-dominated and mostly from Pará, so conclusions are limited to these cases. Scores are uncalibrated rankings, not probabilities of conversion or legal compliance verdicts.

## Quick start: check the reported metrics (GitHub only)

Use Python 3.11 (tested with 3.11.15):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt   # CPU users: install torch from the PyTorch CPU index first

python scripts/reproduce.py --quick
```

This checks the hashes of the files in Git. It then recomputes AUROC and confusion counts for nine methods on the forest-loss test, plus the flags on the 100 established-agriculture controls. All of these come from the saved predictions and frozen thresholds, and must match the saved evaluation. It takes a few seconds and writes `rerun/quick_check.json`.

## Full reproduction (EOTDL data)

Download the snapshot with the [EOTDL client](https://www.eotdl.com/docs/getting-started/install) (Python 3.12+, kept in a separate environment):

```bash
eotdl auth login
eotdl datasets get brazil-forest-to-agriculture-benchmark --path ./downloads
```

Either work directly inside the downloaded folder, or copy its large files into this checkout:

```bash
python scripts/fetch_data.py --from ./downloads/<folder containing manifest.json>
python scripts/reproduce.py
```

The full run takes about three minutes on a CPU; no GPU, Earth Engine account or backbone checkpoint is needed. It:

1. checks all data hashes;
2. re-scores every validation, test and control chip from the cached features and fitted models, and checks the frozen thresholds, confusion counts and paired geographic-block bootstrap;
3. re-runs the post hoc checks (patch agreement, control-date audit, biome boundary), the geographic and threshold-stability diagnostics, and the average-precision and threshold-cap sensitivity tables.

Outputs are compared with the saved results and the records in `provenance/verification/`. See [REPRODUCING.md](REPRODUCING.md) for each step and for repeating the two original classifier fits.

## Repository guide

```text
published/development/   Datasets A/B: earlier screening and endpoint inventories (development evidence)
published/dataset_c/     Main dataset: candidates, references, splits, predictions, thresholds, two linear heads
reviewer_revision/       Comparator models: protocol, code, predictions, operating points, fitted models
cache/                   (EOTDL) TerraFM features, 2020 paired features, label grids, composites, controls
diagnostics/             Average precision, threshold-cap sensitivity and paired-correctness tables
provenance/              Source URLs/hashes, checkpoint metadata, training repeat, verification records
sources/                 (EOTDL) IBGE 2019 biome map for the biome-boundary check
scripts/                 Reproduction, diagnostics, data fetch, manifest and snapshot tools
offline.py               Original experiment: verify scores, or repeat the two fits
verify_revision.py       Comparator models: verify scores, thresholds, flags and bootstrap
additional_diagnostics.py   Average precision and threshold-cap sensitivity
```

See [DATA.md](DATA.md) for the file layout and column definitions, and [PROTOCOL.md](PROTOCOL.md) for the protocol written before the main experiment.

## Licence and attribution

- **Code** (Python scripts and software documentation): [MIT](LICENSE).
- **Data** (reference tables, label grids, cached features, predictions and fitted models): [CC BY-SA 4.0](LICENSE-DATA), attributed to the four authors above and this repository. The share-alike licence follows the terms of the TerraClass and MapBiomas Alerta sources from which the references derive.
- Sentinel-derived features and composites contain modified Copernicus Sentinel data (2020, 2024).

Data providers and citations are listed in [ATTRIBUTION.md](ATTRIBUTION.md). When citing these materials, identify the repository and release, or the EOTDL dataset version.

## Acknowledgements

This work was carried out within **AI4DIS**, “Industrialization of AI Methods for Decision Intelligence in Climate and Security Applications”, funded by the European Space Agency under Contract No. **4000147555/25/I-DT**. The Universitat Politècnica de València participated as a subcontractor to EarthPulse S.L., the prime contractor. The views expressed are those of the authors and do not necessarily reflect the official position of the European Space Agency.

We acknowledge INPE/Embrapa (TerraClass), MapBiomas Alerta, the Copernicus programme, the TerraFM and BigEarthNet teams, and IBGE.

## EOTDL distribution

See [EOTDL.md](EOTDL.md) for building and publishing the snapshot.
