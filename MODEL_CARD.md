# Model card

## Identity and status

Project: Plant Seedling Classification with CNNs. Author: Oludotun Babayemi.
Two educational CNN experiments classify 12 species. The packaged historical notebook contains saved results but no trained weights. The standalone training code is a revised implementation and has not been trained during packaging.

## Intended use

Learning, portfolio review, comparative model development and investigation of classification errors on the plant seedlings benchmark. Input to the standalone training pipeline is uint8 three-channel images with declared RGB/BGR order. Models receive RGB, resized to 64 × 64 and normalized by 255. Output is a 12-class softmax distribution.

## Evaluation

Historical image-level stratified split: 3,847 train, 428 validation, 475 test images. Recorded test accuracy: baseline 0.6358; second model 0.7095. Macro F1: 0.52 and 0.64, respectively, as rounded in saved reports. See `reports/RESULTS.md` for class mapping and provenance. These outputs were inspected, not independently reproduced.

## Data and generalization

The benchmark originates from research-station imagery. The prepared subset's lineage and plant/group identifiers are missing. Exact and near-duplicate contamination and repeated-plant overlap have not been ruled out for the historical run. Generalization to natural backgrounds, Nigeria, different cameras, soil conditions, species outside the label set and growth stages requires testing.

## Error profile

The second model has low recall for Black-grass (0.15) and Common wheat (0.09) in the saved rounded report; their dominant confusion is with Loose Silky-bent. Common Chickweed and Small-flowered Cranesbill perform substantially better than the original narrative suggested. A high average score does not establish adequate crop/weed discrimination for intervention decisions.

## Unsuitable uses

Autonomous spraying, weeding, crop removal, pesticide recommendations or farmer-facing identification without independent validation and appropriate agronomic review. Softmax scores are not calibrated confidence estimates. Unknown-species rejection, human-review thresholds and fallback workflows are not implemented.

## Training changes in the reusable script

New implementation uses Keras RandomRotation (up to 20 degrees), full training epochs, shuffled training arrays and integer labels with sparse categorical cross-entropy. Architectural intent and split proportions are retained. Rotation interpolation and random draws differ from the historical ImageDataGenerator. Final-epoch results must be reported as a new experiment. No best-checkpoint or early-stopping claims are made.

## Reproducibility and stewardship

Each new run saves package versions, data hashes, split membership, class order and preprocessing metadata. Fixed seeds and deterministic operations improve repeatability but do not guarantee identical results across environments. The pipeline blocks exact duplicates unless explicitly overridden; repeated physical plants still require manual/group-aware review. Author review of data rights and any course-origin material remains necessary before public release.
