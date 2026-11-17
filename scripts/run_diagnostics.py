"""Re-run the geographic and threshold-stability diagnostics from the saved predictions.

    python scripts/run_diagnostics.py
reviewer_revision/source/diagnostics.py writes next to its inputs, so this wrapper
copies the four inputs into rerun/diagnostics_inputs/, runs it there, and compares
the result with the saved reviewer_revision/results/diagnostics.json. Saved results
are never modified.
"""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'reviewer_revision/results'
WORK = ROOT / 'rerun/diagnostics_inputs'
INPUTS = ['validation_predictions.csv', 'test_predictions.csv',
          'stable_agriculture_predictions.csv', 'operating_points.json']


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.copyfile(RESULTS / name, WORK / name)
    sys.path.insert(0, str(ROOT / 'reviewer_revision/source'))
    import diagnostics
    diagnostics.OUT = WORK
    diagnostics.SOURCE = ROOT / 'published/dataset_c'
    diagnostics.main()
    fresh = json.loads((WORK / 'diagnostics.json').read_text())
    saved = json.loads((RESULTS / 'diagnostics.json').read_text())
    shutil.copyfile(WORK / 'diagnostics.json', ROOT / 'rerun/diagnostics.json')
    if fresh != saved:
        raise SystemExit('rerun/diagnostics.json differs from reviewer_revision/results/diagnostics.json')
    print('PASS: geographic and threshold-stability diagnostics reproduced exactly.')


if __name__ == '__main__':
    main()
