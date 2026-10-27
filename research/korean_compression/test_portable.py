"""CPU-only portability/configuration tests; no model assets or GPU needed."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import project


class PortableTests(unittest.TestCase):
    def test_examples_have_no_machine_roots(self):
        d=json.loads((project.ROOT/'configs/korean_compression/settings.example.json').read_text())
        self.assertEqual(d['microbatch']*d['accumulation'],64)
        for k in ['backbone_dir','followup_dir','assets_dir','evaluation_data','metrics_dir','train_manifest','valid_manifest']:
            self.assertFalse(Path(d[k]).is_absolute())

    def test_python_entrypoints_parse(self):
        for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())

    def test_no_cluster_guard_literal_in_runners(self):
        for name in ['pocket_small_b6k.py','pocket_small_followup.py']:
            text=Path(__file__).with_name(name).read_text()
            self.assertNotIn("os.environ['SLURM_JOB_ID']",text)
            self.assertNotIn('GPU-',text)

    def test_assets_and_settings_are_ignored(self):
        for name in ['.local/settings.json','.local/artifacts/model.pt','runs/backbone/args.yaml']:
            self.assertEqual(subprocess.run(['git','check-ignore',name],cwd=project.ROOT,capture_output=True).returncode,0)

    def test_streaming_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('abc')
            self.assertEqual(project.sha(p),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')


if __name__=='__main__':unittest.main()
