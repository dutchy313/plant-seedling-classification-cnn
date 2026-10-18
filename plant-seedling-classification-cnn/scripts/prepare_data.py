"""Build reproducible RGB arrays from one labelled class-folder dataset subset."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='Directory containing exactly 12 species subdirectories.')
    parser.add_argument('--output', type=Path, default=Path('data/raw'))
    args = parser.parse_args()
    folders = sorted(p for p in args.input.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if len(folders) != 12:
        parser.error('Choose one labelled subset containing exactly 12 class directories.')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty.')
    files = [(p, folder.name) for folder in folders for p in sorted(folder.rglob('*'))
             if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}]
    if set(label for _, label in files) != {f.name for f in folders}:
        parser.error('Every species directory must contain at least one supported image.')
    args.output.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(args.output / 'images_plant.npy', mode='w+',
                                     dtype=np.uint8, shape=(len(files), 128, 128, 3))
    with (args.output / 'Labels_plant.csv').open('w', newline='') as labels_file, \
         (args.output / 'source_manifest.csv').open('w', newline='') as manifest_file:
        labels_writer = csv.writer(labels_file); labels_writer.writerow(['Label'])
        manifest = csv.writer(manifest_file)
        manifest.writerow(['row_index', 'relative_source', 'label', 'source_sha256'])
        for i, (path, label) in enumerate(files):
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert('RGB')
                array[i] = np.asarray(rgb.resize((128, 128), Image.Resampling.BILINEAR))
            labels_writer.writerow([label])
            manifest.writerow([i, str(path.relative_to(args.input)), label,
                               hashlib.sha256(path.read_bytes()).hexdigest()])
    array.flush()
    (args.output / 'preparation.json').write_text(json.dumps({
        'color_order': 'rgb', 'dtype': 'uint8', 'count': len(files),
        'size': [128, 128], 'sorting': 'class name then source path',
        'warning': 'This is a new deterministic preparation, not a reconstruction of the original array order.'
    }, indent=2))
    print(f'Prepared {len(files)} RGB images in {args.output}')


if __name__ == '__main__':
    main()
