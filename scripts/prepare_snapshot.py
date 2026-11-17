"""Build the EOTDL upload folder from a clean, tagged checkout plus the EOTDL-only files.

    python scripts/prepare_snapshot.py --tag v1.0.0
Writes dist/<tag>/brazil-forest-to-agriculture-benchmark/ with independent copies of
every Git-tracked file and every manifest file marked location=eotdl, plus
release-manifest.json (source commit, tag and per-file SHA-256). Never uploads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NAME = 'brazil-forest-to-agriculture-benchmark'
LICENSE_ID = 'CC-BY-SA-4.0'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / 'dist' / args.tag / NAME
    if git('status', '--porcelain', '--untracked-files=no'):
        raise SystemExit('Commit source changes before building a release snapshot.')
    commit = git('rev-parse', 'HEAD')
    if commit != git('rev-parse', f'{args.tag}^{{commit}}'):
        raise SystemExit(f'Check out {args.tag} to build its snapshot.')
    readme = (ROOT / 'README.md').read_text()
    metadata = readme.split('---', 2)[1] if readme.startswith('---\n') else ''
    if not re.search(rf'^name: {NAME}$', metadata, re.M) or f'license: {LICENSE_ID}' not in metadata:
        raise SystemExit('README.md must start with EOTDL metadata naming this dataset and licence.')
    sources = [p for p in git('ls-files').splitlines() if p != '.gitignore']
    data = json.loads((ROOT / 'manifest.json').read_text())['files']
    for item in data:
        if item['location'] == 'eotdl':
            sources.append(item['path'])
    expected = {item['path']: item['sha256'] for item in data}
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for relative in sorted(set(sources)):
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise SystemExit(f'Expected a regular file: {relative}')
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy rather than link, so the snapshot cannot change through the working tree.
        shutil.copyfile(source, target)
        digest = sha256(target)
        if relative in expected and digest != expected[relative]:
            raise SystemExit(f'Hash differs from manifest.json: {relative}')
        entries.append({'path': relative, 'bytes': target.stat().st_size, 'sha256': digest})
    manifest = {'name': NAME, 'release': args.tag, 'github_commit': commit,
                'github_release': f'https://github.com/alalbiol/{NAME}/releases/tag/{args.tag}',
                'eotdl_dataset': f'https://www.eotdl.com/datasets/{NAME}', 'eotdl_version': None,
                'files': entries,
                'note': 'Lists every payload file except itself. The EOTDL version is assigned on upload.'}
    (output / 'release-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'Snapshot ready: {output}; {len(entries)} files; '
          f'{sum(e["bytes"] for e in entries) / 1e6:.0f} MB. Nothing uploaded.')


if __name__ == '__main__':
    main()
