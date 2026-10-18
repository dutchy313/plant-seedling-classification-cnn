# Contributing

Use an issue to describe a reproducible bug or proposed experiment. For changes, create a branch and submit a pull request explaining behavior, validation and any effect on historical comparability.

Run `python -m unittest discover -s tests -v` and `python scripts/check_repository.py` from the repository root. Full training requires local data and TensorFlow. Do not commit raw datasets, credentials, model weights or private paths. Include dataset version, preprocessing, split design and environment when reporting new results. Preserve historical evidence; place new results in a clearly named experiment directory after reviewing its contents for publication.
