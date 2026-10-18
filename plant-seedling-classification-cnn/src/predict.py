"""Predict a species from a trusted local model and an ordinary RGB image."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import tensorflow as tf
    from PIL import Image, ImageOps
    from data import preprocess
    meta = json.loads((args.run / 'metadata.json').read_text())
    model = tf.keras.models.load_model(args.run / 'model.keras', compile=False, safe_mode=True)
    with Image.open(args.image) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    x = preprocess(rgb[None, ...], 'rgb', meta['image_size'])
    scores = model.predict(x, verbose=0)[0]
    top = np.argsort(scores)[::-1][:3]
    print(json.dumps({'predictions': [{'species': meta['classes'][int(i)],
                                      'softmax_score': float(scores[i])} for i in top],
                      'note': 'Scores are uncalibrated; this is not a field-use recommendation.'},
                     indent=2))


if __name__ == '__main__':
    main()
