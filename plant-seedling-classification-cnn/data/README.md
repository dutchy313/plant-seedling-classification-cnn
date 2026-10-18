# Data access and preparation

Download a labelled plant subset from the [official dataset page](https://vision.eng.au.dk/plant-seedlings-dataset/). Record the exact version and subset you use. Do not combine differently processed releases or labelled training data with an unlabelled competition test set.

## Option A: original prepared arrays

Place `images_plant.npy` and `Labels_plant.csv` in `data/raw/`.

- Images: numeric, non-pickled uint8 array, shape `(N, H, W, 3)`.
- CSV: complete `Label` column; row i must label image i.
- Exactly 12 species; enough observations per class for stratification.
- Independently confirm BGR or RGB; the `.npy` format does not encode channel meaning.

The supplied notebook recorded 4,750 images of 128 × 128 pixels. The arrays themselves were not supplied for repository preparation. Source release, row order, plant identifiers, channel provenance and checksums are unknown. Do not claim exact reproduction without them.

## Option B: prepare a new subset

Extract one labelled subset into a directory with exactly 12 species folders, each containing its corresponding images. From the repository root:

```bash
python scripts/prepare_data.py --input path/to/labelled-subset --output data/raw
```

The output directory must be new or empty. This sorts species folders and source paths, applies EXIF orientation, converts to RGB and resizes to 128 × 128 using Pillow bilinear interpolation. It writes the two arrays/labels inputs, a source-file manifest with hashes, and preparation metadata. Use `--color-order rgb` during training. The original notebook's BGR conversion must be disabled for these RGB arrays.

This is a reproducible new preparation. Its ordering and resizing may differ from the original arrays, so scores and split membership can differ. Review class names, counts, near duplicates and plant identity before modelling. Folder names determine labels; this script cannot correct misfiled images.

## Rights and attribution

Dataset content is not included. Images displayed in the historical notebook retain applicable dataset terms. Cite the dataset paper and review the official CC BY-SA notice before distributing images, annotations, modified datasets or trained weights. A repository code license does not replace those terms.
