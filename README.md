# Plant Seedling Classification with CNNs

**Author: Oludotun Babayemi**  
An agricultural computer vision case study using TensorFlow/Keras to classify seedling images into 12 species and compare two convolutional neural networks.

The supplied notebook's saved test outputs show **63.58% accuracy for the baseline CNN** and **70.95% for the second CNN**, an increase of **7.37 percentage points** on 475 test images. These are historical notebook results, not results of a new execution of this repository.

## Project overview

Identifying crops and weeds at the seedling stage is a useful computer vision research problem. This project demonstrates image exploration, preprocessing, stratified splitting, CNN training, learning-rate scheduling and class-level evaluation. It is a portfolio research prototype; field performance and agricultural decision safety have not been established.

## Explore the evidence

- [Reviewed notebook with historical outputs](notebooks/plant_seedling_classification.ipynb)
- [Recorded model comparison](reports/model_comparison.csv)
- [Species-level metrics](reports/classification_reports.csv)
- [Results provenance and interpretation](reports/RESULTS.md)
- [Model card](MODEL_CARD.md)
- [Review changes and validation](CHANGELOG.md)

![Second CNN confusion matrix from the supplied notebook](reports/figures/augmented_confusion_matrix.png)

The largest observed confusions include Black-grass and Common wheat being classified as Loose Silky-bent. Overall accuracy obscures these weaknesses.

## Dataset

The [Plant Seedlings Dataset](https://vision.eng.au.dk/plant-seedlings-dataset/) was recorded through a collaboration between Aarhus University and the University of Southern Denmark. The supplied notebook uses a prepared subset containing **4,750 images**, each **128 × 128 × 3**, and 12 species:

Black-grass; Charlock; Cleavers; Common Chickweed; Common wheat; Fat Hen; Loose Silky-bent; Maize; Scentless Mayweed; Shepherds Purse; Small-flowered Cranesbill; Sugar beet.

Raw data and trained models are excluded. See [data instructions](data/README.md) for using existing arrays or preparing new arrays from labelled image folders. The original array-construction process, row ordering and source version are unavailable, so the preparation script does not promise to reproduce the historical scores.

## Historical results

| Metric | Baseline CNN | Second CNN |
|---|---:|---:|
| Test accuracy | 63.58% | 70.95% |
| Test loss | 1.1147 | 0.8945 |
| Macro F1, rounded report | 0.52 | 0.64 |
| Weighted F1, rounded report | 0.59 | 0.69 |
| Test images | 475 | 475 |

The actual image-level split was approximately **81% train / 9% validation / 10% test**: 3,847 / 428 / 475 images. This corrects the original 80/10/10 description without changing the historical split.

The baseline uses three convolution/pooling blocks (128, 64, 32 filters), a 16-unit dense layer and dropout. The second CNN uses two blocks (64, 32 filters), batch normalization, dropout, random rotations and learning-rate reduction. Several factors change together, so the score difference cannot be attributed to augmentation alone.

## Install

Suggested environment: Python 3.11 and a fresh virtual environment. GPU acceleration is optional; training on a CPU may take longer.

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependency ranges are proposed compatibility bounds, **not a tested lockfile**. TensorFlow training was not run during packaging. After a successful run, record the resolved environment with `python -m pip freeze > requirements-lock.txt` and review it before committing.

## Run the reviewed notebook

```bash
jupyter lab notebooks/plant_seedling_classification.ipynb
```

The first setup cells use `data/raw/images_plant.npy` and `data/raw/Labels_plant.csv` relative to the project root. Confirm the source array is BGR before running the historical BGR-to-RGB conversion. Arrays produced by the new preparation script are RGB and need that conversion disabled. Saved plots and outputs remain evidence from the original execution, not from the edited setup cells.

## Train the reusable implementation

The reusable implementation preserves the approximate historical split proportions and architectures, but modernizes augmentation and uses every training batch. It is a **revised experiment**, not a promise of identical historical metrics.

For RGB arrays created by the preparation script:

```bash
python src/train.py --images data/raw/images_plant.npy --labels data/raw/Labels_plant.csv --color-order rgb --model baseline --output artifacts/baseline
python src/train.py --images data/raw/images_plant.npy --labels data/raw/Labels_plant.csv --color-order rgb --model augmented --output artifacts/augmented
```

For verified BGR arrays, use `--color-order bgr` instead. Each output directory must be new or empty. Runs save the model, class mapping, environment, dataset hashes, split manifest, exact-duplicate check, predictions, classification report, confusion matrix and training curves.

The final epoch is pre-specified. Validation drives learning-rate scheduling; do not use the test results to tune models. Comparing these two pre-specified experiments is exploratory and does not replace independent validation of the chosen model.

## Predict an image

After training:

```bash
python src/predict.py --run artifacts/augmented --image path/to/seedling.png
```

Prediction prints three species and their uncalibrated softmax scores. It is not an agronomic recommendation. Load only trusted local model files.

## Validation and limitations

- Packaging checks cover Python syntax, notebook structure, historical metric extraction, split behavior, input validation, deterministic data preparation and CLI help. Full model training and inference remain untested because TensorFlow and the dataset were not available in the packaging environment.
- The supplied notebook did not record a complete dependency environment. Its Python metadata alone cannot establish the environment used for all saved outputs.
- The original augmented training loop floors `steps_per_epoch` and disables shuffling. Logs show alternating very short epochs. The revised script removes this scheduling pattern; its results must be measured afresh.
- Image-level stratification does not establish plant-level independence. Investigate repeated physical plants, near duplicates and growth-stage observations across splits. The script checks only exact byte-identical input images and stops on duplicates by default.
- The source array's channel order and preparation history require verification.
- Species-level recall is uneven. Controlled-background performance does not establish performance on Nigerian farms or other field settings.
- No calibrated rejection threshold, external validation, field trial, latency benchmark or edge-device evaluation is included.

## Future work

Priorities are plant/group-aware validation where identifiers are available, external field images, transfer-learning comparisons, calibration and uncertainty estimation, and error review with agronomists. Mobile deployment, herbicide application and autonomous weeding require separate validation.

## Attribution and licensing

Dataset reference: Giselsson, T. M., Dyrmann, M., Jørgensen, R. N., Jensen, P. K., and Midtiby, H. S. (2017). [A Public Image Database for Benchmark of Plant Seedling Classification Algorithms](https://arxiv.org/abs/1711.05458).

The dataset provider publishes images and annotations under CC BY-SA and requests full citation, including for trained models. See [NOTICE.md](NOTICE.md) for scope and third-party content. The MIT license covers newly supplied standalone code and original repository documentation only where the author holds the rights; it does not relicense dataset content or third-party teaching materials.

## Author

**Oludotun Babayemi** — monitoring, evaluation and learning specialist working across agriculture, data systems and applied AI. This project explores agricultural computer vision as part of an applied machine-learning portfolio.
