"""Write manifest.json: sizes and SHA-256 hashes of every released data file.

Files marked "eotdl" are too large for Git and are distributed only in the
EOTDL snapshot. offline.py and verify_revision.py check every listed file, so
re-run this only after a deliberate change to released data, then re-run the
verification and commit the new records.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORIES = ['cache', 'diagnostics', 'provenance', 'published',
                    'reviewer_revision', 'source', 'sources']
# Release records and the verification outputs that cite this manifest's hash.
EXCLUDED = ['provenance/releases', 'provenance/verification']
EOTDL_ONLY_PREFIXES = ['cache/']
EOTDL_ONLY_FILES = ['reviewer_revision/results/bitemporal_spectral_forest.pkl',
                    'sources/Biomas_250mil.zip']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def is_eotdl_only(relative):
    return relative in EOTDL_ONLY_FILES or any(relative.startswith(p) for p in EOTDL_ONLY_PREFIXES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    entries = []
    for directory in DATA_DIRECTORIES:
        for path in sorted((ROOT / directory).rglob('*')):
            relative = path.relative_to(ROOT).as_posix()
            if (not path.is_file() or '__pycache__' in path.parts
                    or any(relative.startswith(e + '/') for e in EXCLUDED)):
                continue
            if path.is_symlink():
                raise ValueError(f'Symlinks are not released: {relative}')
            entries.append({'path': relative, 'bytes': path.stat().st_size,
                            'sha256': sha256(path),
                            'location': 'eotdl' if is_eotdl_only(relative) else 'git'})
    missing = [f for f in EOTDL_ONLY_FILES if f not in {e['path'] for e in entries}]
    if missing:
        raise FileNotFoundError(f'Large files absent; run scripts/fetch_data.py first: {missing}')
    manifest = {'created': '2026-10-01', 'revised': '2026-11-17',
                'note': 'Released data files. location=eotdl files are not stored in Git.',
                'files': entries}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    large = [e for e in entries if e['location'] == 'eotdl']
    print(f'{len(entries)} files; {len(large)} EOTDL-only '
          f'({sum(e["bytes"] for e in large) / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
