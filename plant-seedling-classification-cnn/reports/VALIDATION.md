# Packaging validation

Checked 2026-10-18.

- Four behavioral tests passed: split sizes/independence/repeatability, invalid input rejection, exact duplicate detection, and folder preparation with RGB/label/provenance alignment and overwrite protection.
- All Python modules and notebook code cells parsed successfully.
- Notebook JSON/cell structure, internal Markdown file links and historical metric consistency checked. Full nbformat schema validation was unavailable locally; the GitHub Actions workflow installs nbformat and performs it.
- Training, prediction and data-preparation CLI help completed successfully.
- Saved shape outputs, test evaluations and numeric reports compared byte-for-byte with source text. Historical model definitions and fitting calls compared as Python syntax trees; unchanged.
- Credential-pattern scan: no matches. This is a limited pattern check, not a comprehensive security audit.
- TensorFlow training, trained-model inference and OpenCV preprocessing were not executed: required dependencies, dataset arrays and trained weights were not available.
- Dependency resolution/install and GitHub Actions have not been run in this environment. No tested lockfile is claimed.
