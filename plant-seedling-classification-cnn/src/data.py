"""Input validation, preprocessing and reproducible row-level splits."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_arrays(images_path, labels_path):
    images = np.load(images_path, allow_pickle=False)
    table = pd.read_csv(labels_path)
    if 'Label' not in table.columns or table['Label'].isna().any():
        raise ValueError('CSV must contain a complete Label column.')
    labels = table['Label'].astype(str).str.strip().to_numpy()
    if any(not label for label in labels):
        raise ValueError('Labels cannot be blank.')
    if images.ndim != 4 or images.shape[-1] != 3 or min(images.shape[1:3]) < 1:
        raise ValueError('Expected an (N, H, W, 3) image array.')
    if len(images) != len(labels) or len(images) == 0:
        raise ValueError('Image and label counts must match and be nonzero.')
    if images.dtype != np.uint8:
        raise ValueError('Expected uint8 pixels in [0, 255]; do not supply normalized arrays.')
    if len(np.unique(labels)) != 12:
        raise ValueError('This project expects exactly 12 species.')
    return images, labels


def split_indices(labels, seed=42):
    """Preserve the original approximate 81/9/10 split, without overlap."""
    ids = np.arange(len(labels))
    remaining, test = train_test_split(ids, test_size=0.1, random_state=seed,
                                       stratify=labels)
    train, validation = train_test_split(remaining, test_size=0.1, random_state=seed,
                                         stratify=labels[remaining])
    return train, validation, test


def preprocess(images, color_order, size=64):
    """The declared INPUT order is converted to RGB before normalization."""
    import cv2
    if color_order not in ('rgb', 'bgr'):
        raise ValueError('Declare the input color order as rgb or bgr.')
    result = np.empty((len(images), size, size, 3), dtype=np.float32)
    for i, image in enumerate(images):
        rgb = image[..., ::-1] if color_order == 'bgr' else image
        result[i] = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR) / 255.0
    return result


def duplicate_groups(images):
    """Exact byte-identical images only; cannot detect repeated physical plants."""
    groups = {}
    for i, image in enumerate(images):
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        groups.setdefault(digest, []).append(i)
    return [ids for ids in groups.values() if len(ids) > 1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
