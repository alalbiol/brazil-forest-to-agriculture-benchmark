"""Release consistency checks: python -m unittest discover -s tests"""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'manifest.json').read_text())['files']


def in_git_checkout():
    result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT,
                            capture_output=True, text=True)
    return result.returncode == 0 and Path(result.stdout.strip()).resolve() == ROOT


class ReleaseTest(unittest.TestCase):
    @unittest.skipUnless(in_git_checkout(), 'not a Git checkout (e.g. the EOTDL snapshot)')
    def test_eotdl_only_files_stay_out_of_git(self):
        paths = [i['path'] for i in MANIFEST if i['location'] == 'eotdl']
        ignored = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'], cwd=ROOT,
                                 input='\n'.join(paths), capture_output=True, text=True).stdout.split()
        self.assertEqual(set(paths), set(ignored))

    @unittest.skipUnless(in_git_checkout(), 'not a Git checkout (e.g. the EOTDL snapshot)')
    def test_git_files_are_tracked(self):
        tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
        untracked = [i['path'] for i in MANIFEST if i['location'] == 'git' and i['path'] not in tracked]
        self.assertEqual(untracked, [])

    def test_git_files_are_small(self):
        large = [i['path'] for i in MANIFEST if i['location'] == 'git' and i['bytes'] > 20e6]
        self.assertEqual(large, [])

    def test_no_internal_partner_material(self):
        self.assertFalse(any('esa_report' in i['path'] or i['path'].startswith('source/')
                             for i in MANIFEST))

    def test_quick_reproduction(self):
        subprocess.run([sys.executable, 'scripts/reproduce.py', '--quick'], cwd=ROOT,
                       check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
