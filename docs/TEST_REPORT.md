# Numerical test and verification report

**Status:** Automated verification suite authored. Execution results are **not yet verified**.

## What is covered

The `tests/test_matching_methods.py` suite checks known-answer global statistics; 3 × 3 spatial-vector dimensionality and ordering; invalid-image handling; exact synthetic matches; missing-query and missing-gallery cases; zero-variance scaling; empty galleries; and two simple geometry-mask cases.

Tests use synthetic matrices and do not need the course image dataset.

## Reproducing test results

From the repository root, after installing dependencies:

```bash
python -m pytest tests -v --junitxml=pytest-results.xml
```

The same tests run on GitHub Actions in `.github/workflows/verify.yml`. The Actions artifact includes generated `test_report.txt`, `pytest-results.xml` and pdoc HTML files. Enter the total passed/failed counts, runner/environment, commit SHA and execution date in your final assessment report only **after examining a real run**.

## Scope of evidence

Passing these tests confirms bounded cases of numerical code behaviour; it does not validate depth-camera units, tree segmentation quality, field accuracy, or reproducibility of every exploratory script. The saved image-data results in `results/` are separate experimental evidence and should not be confused with unit-test results.

## Known follow-up risks

- The original experiment files do not all have docstrings or unit tests.
- `30_compare_matching_methods.py` computes preprocessing on gallery observations in a transductive comparison; it is not a trained generalisation benchmark.
- Course dataset permission and software licensing must still be confirmed.
