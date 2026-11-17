"""Install the EOTDL-only files (cached features, imagery, labels, large model) into this checkout.

Download the EOTDL snapshot first (see README.md), then:
    python scripts/fetch_data.py --from /path/to/downloaded/brazil-forest-to-agriculture-benchmark
Every file is checked against manifest.json before and after copying. Files already
present with the correct hash are skipped. Use --verify-only to check a checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def matches(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and sha256(path) == item['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--from', dest='source', type=Path, help='downloaded EOTDL snapshot folder')
    source.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    items = [i for i in json.loads((ROOT / 'manifest.json').read_text())['files']
             if i['location'] == 'eotdl']
    if args.source and not (args.source / 'manifest.json').is_file():
        raise SystemExit(f'{args.source} does not look like the snapshot root (no manifest.json).')
    present = copied = 0
    missing = []
    for item in items:
        target = ROOT / item['path']
        if matches(target, item):
            present += 1
            continue
        if args.verify_only:
            missing.append(item['path'])
            continue
        origin = args.source / item['path']
        if not matches(origin, item):
            raise SystemExit(f'Missing or changed in the download: {item["path"]}')
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.part')
        shutil.copyfile(origin, partial)
        if not matches(partial, item):
            partial.unlink()
            raise SystemExit(f'Copy failed verification: {item["path"]}')
        partial.replace(target)
        copied += 1
    if missing:
        raise SystemExit(f'{len(missing)} of {len(items)} EOTDL-only files missing or changed, '
                         f'e.g. {missing[0]}')
    total = sum(i['bytes'] for i in items) / 1e6
    print(f'{len(items)} EOTDL-only files verified ({total:.0f} MB): '
          f'{present} already present, {copied} copied.')


if __name__ == '__main__':
    main()
