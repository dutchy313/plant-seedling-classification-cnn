# Changes for repository preparation

## 0.1.0 — 2026-10-18

### Reviewed historical notebook

- Preserved saved training/evaluation outputs and model/training logic.
- Replaced personal Colab/Drive setup with relative paths and optional environment variables.
- Removed unused imports and an unused plotting helper with an incorrect class-count setting.
- Removed blanket warning suppression and standardized the callback import.
- Made the source RGB/BGR assumption explicit and configurable; default BGR preserves historical intent.
- Corrected split proportions, species-level interpretation and unsupported deployment recommendations.
- Removed Colab metadata and the private notebook link; added provenance and attribution.

### New reusable implementation

- Added input validation, deterministic folder-to-array preparation and explicit color-order handling.
- Added training and prediction commands with modern Keras augmentation.
- Retained approximate 81/9/10 split and baseline/second-model architectures, with dynamic class mapping.
- Replaced the original floored generator schedule with complete shuffled epochs; results must be measured afresh.
- Added exact-duplicate detection, dataset hashes, row split manifests, environment records and named metrics.
- Added model saving, matched inference preprocessing and protection against overwriting completed runs.

### Documentation and evidence

- Added README, model card, data guide, upload guide, license-scope notice and MIT license for new contributions.
- Exported saved metrics, logs and four figures; no raw images or weights added outside notebook outputs.
- Added lightweight automated checks and behavioral tests for data integrity.

### Validation limits

Python source parsing, notebook structure/code checks, metric consistency, links, CLI help and synthetic data tests were performed during packaging. Training and trained-model inference were not executed; TensorFlow and original dataset inputs were unavailable. Dependency ranges are proposed rather than a frozen historical environment. Publication rights for any teaching-provider starter content need author verification.
