"""Train a baseline or revised CNN. Run with --help for the CLI."""
import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from data import load_arrays, split_indices, preprocess, duplicate_groups, sha256


def build_model(tf, variant, classes, seed):
    layers = tf.keras.layers
    blocks = [layers.Input(shape=(64, 64, 3))]
    if variant == 'augmented':
        blocks.append(layers.RandomRotation(20 / 360, fill_mode='nearest', seed=seed))
    filters = (128, 64, 32) if variant == 'baseline' else (64, 32)
    for count in filters:
        blocks.extend([layers.Conv2D(count, 3, padding='same', activation='relu'),
                       layers.MaxPooling2D(2, padding='same')])
    if variant == 'augmented':
        blocks.append(layers.BatchNormalization())
    blocks.extend([layers.Flatten(), layers.Dense(16, activation='relu'),
                   layers.Dropout(0.3), layers.Dense(classes, activation='softmax')])
    model = tf.keras.Sequential(blocks)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--color-order', choices=['rgb', 'bgr'], required=True,
                        help='Color order in the source array; verify before training.')
    parser.add_argument('--model', choices=['baseline', 'augmented'], default='augmented')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True,
                        help='New or empty directory; existing runs are never overwritten.')
    parser.add_argument('--allow-exact-duplicates', action='store_true',
                        help='Proceed despite exact duplicate images; recorded in run metadata.')
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('--output must be new or empty')

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix

    images, labels = load_arrays(args.images, args.labels)
    duplicates = duplicate_groups(images)
    if duplicates and not args.allow_exact_duplicates:
        raise ValueError('Exact duplicate images found. Resolve them, or explicitly accept '
                         'the leakage risk with --allow-exact-duplicates.')
    ids = split_indices(labels, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    memberships = {int(i): name for name, part in zip(['train', 'validation', 'test'], ids)
                   for i in part}
    pd.DataFrame({'row_index': np.arange(len(labels)), 'label': labels,
                  'split': [memberships[i] for i in range(len(labels))]}).to_csv(
                      args.output / 'split_manifest.csv', index=False)
    (args.output / 'duplicate_groups.json').write_text(json.dumps(duplicates, indent=2))
    encoder = LabelEncoder().fit(labels[ids[0]])
    y = encoder.transform(labels)
    x = preprocess(images, args.color_order)
    del images

    import tensorflow as tf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf.keras.backend.clear_session()
    model = build_model(tf, args.model, len(encoder.classes_), args.seed)
    callbacks = []
    if args.model == 'augmented':
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', patience=3, factor=0.5, min_lr=1e-5, mode='max'))
    # Use every training row each epoch; do not floor steps_per_epoch.
    history = model.fit(x[ids[0]], y[ids[0]], validation_data=(x[ids[1]], y[ids[1]]),
                        epochs=args.epochs, batch_size=32 if args.model == 'baseline' else 64,
                        shuffle=True, callbacks=callbacks, verbose=2)
    # Evaluate the pre-specified final epoch once. Do not tune using these results.
    scores = model.evaluate(x[ids[2]], y[ids[2]], verbose=0, return_dict=True)
    probabilities = model.predict(x[ids[2]], verbose=0)
    predicted = probabilities.argmax(axis=1)
    report = classification_report(y[ids[2]], predicted, labels=np.arange(12),
                                   target_names=encoder.classes_, output_dict=True,
                                   zero_division=0)
    (args.output / 'metrics.json').write_text(json.dumps(scores, indent=2))
    (args.output / 'classification_report.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame(report).T.to_csv(args.output / 'classification_report.csv')
    pd.DataFrame(history.history).to_csv(args.output / 'training_history.csv', index=False)
    pd.DataFrame({'row_index': ids[2], 'true_label': labels[ids[2]],
                  'predicted_label': encoder.inverse_transform(predicted),
                  'softmax_score': probabilities.max(axis=1)}).to_csv(
                      args.output / 'predictions.csv', index=False)
    matrix = confusion_matrix(y[ids[2]], predicted, labels=np.arange(12))
    pd.DataFrame(matrix, index=encoder.classes_, columns=encoder.classes_).to_csv(
        args.output / 'confusion_matrix.csv')
    fig, ax = plt.subplots(figsize=(11, 10))
    display = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(display, ax=ax)
    ax.set(xticks=range(12), yticks=range(12), xticklabels=encoder.classes_,
           yticklabels=encoder.classes_, xlabel='Predicted species', ylabel='True species')
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
    for i in range(12):
        for j in range(12):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    fig.tight_layout(); fig.savefig(args.output / 'confusion_matrix.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ['accuracy', 'loss']):
        ax.plot(history.history[metric], label='Train')
        ax.plot(history.history['val_' + metric], label='Validation')
        ax.set(xlabel='Epoch (zero-based)', ylabel=metric); ax.legend()
    fig.tight_layout(); fig.savefig(args.output / 'training_curves.png', dpi=160); plt.close(fig)
    model.save(args.output / 'model.keras')
    metadata = {'classes': encoder.classes_.tolist(), 'image_size': 64,
                'training_array_color_order': args.color_order, 'model_input_color_order': 'rgb',
                'normalization': 'uint8 / 255.0', 'variant': args.model, 'seed': args.seed,
                'epochs': args.epochs, 'split_counts': dict(zip(['train', 'validation', 'test'],
                                                             map(len, ids))),
                'exact_duplicate_groups': len(duplicates),
                'allow_exact_duplicates': args.allow_exact_duplicates,
                'images_sha256': sha256(args.images), 'labels_sha256': sha256(args.labels),
                'python': platform.python_version(), 'command': sys.argv,
                'devices': [str(device) for device in tf.config.list_physical_devices()],
                'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
