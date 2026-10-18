# Historical results and provenance

All numbers here come from saved outputs in the supplied notebook. Training was not repeated. Original cell indexes below are zero-based; the reviewed notebook retains their order.

## Evidence sources

| Evidence | Original cell |
|---|---:|
| Array dimensions | 15 |
| Split dimensions | 36 |
| Baseline training | 49 |
| Baseline evaluation | 53 |
| Baseline labelled confusion matrix | 56 |
| Baseline numeric classification report | 58 |
| Second-model training | 68 |
| Second-model evaluation | 72 |
| Second-model labelled confusion matrix | 75 |
| Second-model numeric classification report | 77 |

`provenance.json` records the source notebook SHA-256. Report text files and figures are extracted from saved outputs. The CSV values retain the precision shown in those outputs; they are not recalculated from predictions, which were unavailable.

## Comparison

| Measure | Baseline | Second model |
|---|---:|---:|
| Test accuracy | 0.6358 | 0.7095 |
| Test loss | 1.1147 | 0.8945 |
| Macro F1 (rounded) | 0.52 | 0.64 |
| Weighted F1 (rounded) | 0.59 | 0.69 |
| Test support | 475 | 475 |

The difference is 0.0737, or 7.37 percentage points, using displayed accuracy. It is descriptive: no paired uncertainty interval or significance test was computed. The test set was evaluated for both models, so this is exploratory model comparison rather than an independent final validation of a selected winner.

## Corrected class mapping

The original narrative assigned several numeric IDs to the wrong species. The class order is consistent with LabelBinarizer alphabetical encoding and was visually checked against the labelled saved second-model confusion matrix. Use the actual saved labels, including `Common wheat` capitalization.

| ID | Species | Test support | Baseline F1 | Second-model F1 | Second-model recall |
|---|---|---:|---:|---:|---:|
| 0 | Black-grass | 26 | 0.00 | 0.24 | 0.15 |
| 1 | Charlock | 39 | 0.63 | 0.87 | 0.92 |
| 2 | Cleavers | 29 | 0.23 | 0.72 | 0.76 |
| 3 | Common Chickweed | 61 | 0.83 | 0.87 | 0.89 |
| 4 | Common wheat | 22 | 0.09 | 0.13 | 0.09 |
| 5 | Fat Hen | 48 | 0.67 | 0.56 | 0.46 |
| 6 | Loose Silky-bent | 65 | 0.74 | 0.68 | 0.88 |
| 7 | Maize | 22 | 0.48 | 0.73 | 0.73 |
| 8 | Scentless Mayweed | 52 | 0.61 | 0.72 | 0.83 |
| 9 | Shepherds Purse | 23 | 0.48 | 0.56 | 0.43 |
| 10 | Small-flowered Cranesbill | 50 | 0.77 | 0.88 | 0.90 |
| 11 | Sugar beet | 38 | 0.68 | 0.74 | 0.68 |

The overall gain does not apply to every species: Fat Hen and Loose Silky-bent have lower F1 in the second experiment. Black-grass and Common wheat remain particularly weak. Common wheat is a crop, not an example of a weed for automatic spraying.

## Training-exposure caveat

The historical augmented model sets `steps_per_epoch = 3847 // 64` (60), although the generator contains 61 batches including a partial batch. Its saved logs alternate between multi-second and near-zero-duration epochs. This is consistent with uneven generator consumption; exact exposure cannot be established from the logs alone. Do not describe all 30 epochs as confirmed full passes through the training set. The revised implementation uses array-based full epochs, shuffling and a Keras rotation layer. Any results from it are new evidence.

## Additional limitations

The dataset arrays, preparation lineage, trained weights, raw prediction vectors and complete runtime versions were not supplied. Channel order is not independently verified. Image-level splitting does not establish independence across physical plants or growing stages. No field evaluation, calibrated confidence threshold, edge benchmark or deployment trial is available. The original notebook's deployment claims have been replaced by future research recommendations.

Dataset source and citation: [official dataset](https://vision.eng.au.dk/plant-seedlings-dataset/) and [Giselsson et al. (2017)](https://arxiv.org/abs/1711.05458). See `../NOTICE.md` for attribution and rights.
