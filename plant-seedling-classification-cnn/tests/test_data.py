"""Behavioral checks for validation, split independence and data preparation."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import load_arrays, split_indices, duplicate_groups


class DataTests(unittest.TestCase):
    def test_historical_split_sizes_and_independence(self):
        # Balanced synthetic labels; no private or external data needed.
        labels = np.array([f'species_{i % 12:02}' for i in range(4750)])
        parts = split_indices(labels)
        self.assertEqual([len(p) for p in parts], [3847, 428, 475])
        sets = [set(p) for p in parts]
        self.assertEqual(len(set.union(*sets)), 4750)
        self.assertTrue(all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i)))
        for first, second in zip(parts, split_indices(labels)):
            np.testing.assert_array_equal(first, second)
        self.assertTrue(all(len(np.unique(labels[p])) == 12 for p in parts))

    def test_validation_rejects_misaligned_and_normalized_data(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            np.save(p / 'x.npy', np.zeros((24, 4, 4, 3), dtype=np.uint8))
            pd.DataFrame({'Label': [f'class_{i % 12}' for i in range(24)]}).to_csv(p / 'y.csv', index=False)
            self.assertEqual(len(load_arrays(p / 'x.npy', p / 'y.csv')[0]), 24)
            pd.DataFrame({'Label': ['a']}).to_csv(p / 'y.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'counts'):
                load_arrays(p / 'x.npy', p / 'y.csv')
            pd.DataFrame({'Label': [f'class_{i % 12}' for i in range(24)]}).to_csv(p / 'y.csv', index=False)
            np.save(p / 'x.npy', np.zeros((24, 4, 4, 3), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'uint8'):
                load_arrays(p / 'x.npy', p / 'y.csv')

    def test_duplicate_detection(self):
        x = np.array([np.zeros((2, 2, 3)), np.ones((2, 2, 3)), np.zeros((2, 2, 3))])
        self.assertEqual(duplicate_groups(x), [[0, 2]])

    def test_folder_preparation_keeps_rgb_alignment_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for i in range(12):
                folder = p / 'input' / f'class_{i:02}'
                folder.mkdir(parents=True)
                Image.new('RGB', (7, 5), (i, 30, 240)).save(folder / 'example.png')
            command = [sys.executable, str(ROOT / 'scripts/prepare_data.py'),
                       '--input', str(p / 'input'), '--output', str(p / 'output')]
            subprocess.run(command, check=True, capture_output=True)
            x, labels = load_arrays(p / 'output/images_plant.npy', p / 'output/Labels_plant.csv')
            self.assertEqual(x.shape, (12, 128, 128, 3))
            for i in range(12):
                self.assertEqual(labels[i], f'class_{i:02}')
                np.testing.assert_array_equal(x[i, 0, 0], [i, 30, 240])
            self.assertEqual(json.loads((p / 'output/preparation.json').read_text())['color_order'], 'rgb')
            self.assertEqual(len(pd.read_csv(p / 'output/source_manifest.csv')), 12)
            # Existing output is deliberately protected.
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
