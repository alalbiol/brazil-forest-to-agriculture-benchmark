# EOTDL snapshot

This is an EOTDL **dataset entry** named `brazil-forest-to-agriculture-benchmark`.
The root README carries its EOTDL YAML metadata and description. The dataset
licence is CC BY-SA 4.0; code and software documentation are MIT.

GitHub holds the code and every file small enough for Git. The EOTDL snapshot
holds the complete tagged Git tree **plus** the files marked `location: eotdl` in
`manifest.json`: cached features, composites and label grids (`cache/`), the
215 MB random forest and the IBGE biome archive. EOTDL ingests a folder of files;
no ZIP wrapper is needed.

## GitHub and EOTDL version mapping

GitHub release `v1.0.0` identifies the code. The EOTDL version number is assigned
on upload and **is not derived from the Git tag**. After upload, record it in
`provenance/releases/v1.0.0.json` and the README, as was done for
`iberian-pines-ifn-validation`.

## Build the snapshot

From a clean checkout of the tag, with the EOTDL-only files present:

```bash
git checkout v1.0.0
python scripts/fetch_data.py --verify-only
python scripts/reproduce.py
python scripts/prepare_snapshot.py --tag v1.0.0
```

The snapshot is written to `dist/v1.0.0/brazil-forest-to-agriculture-benchmark/`. It
contains independent copies, not links, and existing output folders are never
overwritten. `release-manifest.json` records the source commit, tag and per-file
SHA-256. Check the copy independently:

```bash
cd dist/v1.0.0/brazil-forest-to-agriculture-benchmark
python scripts/reproduce.py
rm -rf rerun
```

Remove the generated `rerun/` folder (or rebuild into a fresh folder) before
ingesting, so that only the manifest-listed payload is uploaded.

## Upload

Use a separate **Python 3.12+** environment for the EOTDL client, so the tested
analysis dependencies are not changed:

```bash
python3.12 -m venv /tmp/eotdl-uploader
source /tmp/eotdl-uploader/bin/activate
python -m pip install eotdl
eotdl auth login
eotdl datasets ingest -p ./dist/v1.0.0/brazil-forest-to-agriculture-benchmark
```

Run the last command only when ready to publish. Push the GitHub repository and
release first, so the `source` link in the metadata resolves. After ingestion:

1. Download the published manifest from EOTDL and check every hash against
   `release-manifest.json`.
2. Add `provenance/releases/v1.0.0.json` with the dataset ID, EOTDL version, commit,
   tag, manifest hash, payload file count and date, and link it from the README.
3. Run `python scripts/update_github_about.py` (after `gh auth login`) to set the
   GitHub description, website and topics.
4. Add a row to the AI4DIS project README's EOTDL resources table.

## Download

```bash
eotdl auth login
eotdl datasets get brazil-forest-to-agriculture-benchmark --version <N> --path ./downloads
```

Use the folder that contains `manifest.json` directly, or install its large files into
a Git checkout with `scripts/fetch_data.py --from <folder>`.

## Official references

- [Dataset ingestion and README metadata](https://www.eotdl.com/docs/datasets/ingest)
- [Authentication](https://www.eotdl.com/docs/getting-started/authenticate)
- [Client installation](https://www.eotdl.com/docs/getting-started/install)
