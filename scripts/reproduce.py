"""Reproduce the paper's results.

Quick check (GitHub checkout only, seconds):
    python scripts/reproduce.py --quick
Recomputes the headline loss-test AUROC and confusion counts and the
established-agriculture flags from the saved prediction tables and frozen
thresholds, and checks them against the saved evaluation files.

Full check (needs the EOTDL-only files, see scripts/fetch_data.py; minutes, CPU):
    python scripts/reproduce.py
Re-scores every chip from the cached TerraFM features and the fitted models and
re-runs every reported check and diagnostic, writing results under rerun/.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'reviewer_revision/results'
RERUN = ROOT / 'rerun'
RECORDS = ROOT / 'provenance/verification'
METHODS = ['joint_original', 'endpoint_terrafm', 'postclassification', 'bitemporal_terrafm',
           'spectral_linear_endpoint', 'spectral_linear_joint', 'spectral_forest_bitemporal',
           'change_original', 'optical_index']
FULL_STEPS = [
    ('original experiment', ['offline.py', 'verify']),
    ('revised methods', ['verify_revision.py', '--bundle', '.']),
    ('added checks', ['reviewer_revision/source/additional_checks.py', '--bundle', '.']),
    ('control-date audit', ['reviewer_revision/source/control_date_audit.py', '--bundle', '.']),
    ('biome check', ['reviewer_revision/source/check_biome.py', '--bundle', '.']),
    ('geographic and threshold-stability diagnostics', ['scripts/run_diagnostics.py']),
    ('average precision and threshold-cap sensitivity', ['additional_diagnostics.py', '--bundle', '.',
                                                          '--output', 'rerun/diagnostics']),
]
# Saved tables in diagnostics/ that the supplementary run must reproduce byte for byte.
TABLES = ['average_precision.csv', 'threshold_sensitivity.csv', 'paired_disagreements.csv']
# Fresh outputs that must match the committed records exactly (apart from run-specific fields).
RECORDED = ['verification.json', 'revision_verification.json', 'additional_checks.json',
            'control_date_audit.json', 'biome_check.json']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def check_manifest(locations):
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    missing, changed = [], []
    for item in manifest['files']:
        if item['location'] not in locations:
            continue
        path = ROOT / item['path']
        if not path.is_file():
            missing.append(item['path'])
        elif path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            changed.append(item['path'])
    if changed:
        raise SystemExit(f'Changed released files: {changed[:5]} ({len(changed)} total)')
    if missing:
        raise SystemExit(f'{len(missing)} released files are missing, e.g. {missing[0]}. '
                         'Download the EOTDL snapshot and run scripts/fetch_data.py.')
    return sum(item['location'] in locations for item in manifest['files'])


def quick():
    verified = check_manifest({'git'})
    thresholds = json.loads((RESULTS / 'operating_points.json').read_text())['thresholds']
    saved = json.loads((RESULTS / 'evaluation.json').read_text())['results']['loss_endpoints']
    stable_saved = json.loads((RESULTS / 'stable_agriculture_evaluation.json').read_text())
    loss = [r for r in read(RESULTS / 'test_predictions.csv') if r['endpoint'] != 'stable_forest']
    controls = read(RESULTS / 'stable_agriculture_predictions.csv')
    y = np.array([r['endpoint'] == 'agriculture' for r in loss])
    assert (len(loss), int(y.sum()), len(controls)) == (113, 84, 100)
    table = {}
    for method in METHODS:
        score = np.array([float(r[method]) for r in loss])
        flag = score >= thresholds[method]['threshold']
        counts = {'tp': int((y & flag).sum()), 'fn': int((y & ~flag).sum()),
                  'fp': int((~y & flag).sum()), 'tn': int((~y & ~flag).sum())}
        auroc = float(roc_auc_score(y, score))
        reference = saved['methods'][method]
        assert all(counts[k] == reference[k] for k in counts), (method, counts)
        assert abs(auroc - reference['auroc']) < 1e-9, (method, auroc)
        stable_flags = sum(float(r[method]) >= thresholds[method]['threshold'] for r in controls)
        assert stable_flags == stable_saved['groups']['all']['methods'][method]['false_positives']
        table[method] = {'auroc': round(auroc, 6), **counts, 'stable_agriculture_flags': stable_flags}
    report = {'status': 'passed', 'mode': 'quick', 'verified_git_files': verified,
              'loss_test': {'agriculture': 84, 'controls': 29}, 'methods': table}
    RERUN.mkdir(exist_ok=True)
    (RERUN / 'quick_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{"method":28s} AUROC   TP FN FP TN  flagged controls/100')
    for method, v in table.items():
        print(f'{method:28s} {v["auroc"]:.3f} {v["tp"]:4d}{v["fn"]:3d}{v["fp"]:3d}{v["tn"]:3d}'
              f'  {v["stable_agriculture_flags"]:3d}')
    print(f'PASS: saved predictions reproduce the reported metrics. Report: {RERUN / "quick_check.json"}')


def comparable(path):
    """Drop run-specific fields: residual float errors (~1e-16, below the scripts'
    own 1e-6/2e-6 tolerances) and the manifest identity."""
    data = json.loads(path.read_text())
    for key in ('manifest_sha256', 'maximum_score_errors', 'verified_files'):
        data.pop(key, None)
    return data


def full():
    check_manifest({'git', 'eotdl'})
    for label, command in FULL_STEPS:
        print(f'== {label}: python {" ".join(command)}', flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    for name in RECORDED:
        if comparable(RERUN / name) != comparable(RECORDS / name):
            raise SystemExit(f'rerun/{name} differs from provenance/verification/{name}')
    for name in TABLES:
        if (RERUN / 'diagnostics/generated' / name).read_bytes() != (ROOT / 'diagnostics' / name).read_bytes():
            raise SystemExit(f'rerun/diagnostics/generated/{name} differs from diagnostics/{name}')
    print('PASS: all scores, checks and diagnostics reproduced; recorded outputs match.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='saved-prediction check only')
    args = parser.parse_args()
    quick() if args.quick else full()


if __name__ == '__main__':
    main()
