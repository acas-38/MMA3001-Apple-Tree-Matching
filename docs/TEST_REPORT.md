# Automated testing and matching verification report

**Verified:** 25 October 2026. The GitHub Actions workflow ran successfully, including installation, Python syntax checking, 10 pytest tests, script 31 matching verification, pdoc HTML generation and upload of the workflow artifact.

[Open successful GitHub Actions run](https://github.com/acas-38/MMA3001-Apple-Tree-Matching/actions/runs/40170257779) (workflow: Numerical tests and HTML documentation).

## Verification vs validation

The **10 passing unit tests** use synthetic numerical inputs and test features, nearest-neighbour rankings, missing data, zero variance, and synthetic geometry. Passing these checks verifies specified software behaviours; it does **not** prove correct field identification or physical reconstruction.

The `31_verify_matching.py` script rechecks `matching_summary.csv` against `per_tree_predictions.csv` using the 113-identitity gallery manifest. The script also reports exploratory statistical uncertainty.

## Reproducing the checks

```bash
python -m pip install -r requirements.txt
python -m pytest tests -v
python code/31_verify_matching.py --self-test
python code/31_verify_matching.py
```

Run script `30_compare_matching_methods.py` first to regenerate its CSV inputs if needed.

## Observed quantitative results (113 paired identities)

| Feature approach | Top-1 accuracy | Top-3 recall | 95% Wilson interval for top-1 |
| --- | ---: | ---: | ---: |
| Global depth statistics | 1/113 (0.885%) | 3/113 (2.655%) | 0.156%–4.843% |
| Spatial 3×3 statistics | 1/113 (0.885%) | 6/113 (5.310%) | 0.156%–4.843% |
| Exploratory geometric features | 2/113 (1.770%) | 6/113 (5.310%) | 0.487%–6.224% |

The exact paired McNemar tests in the saved output do not show a statistically established difference among these methods. The tests are exploratory, with few successful matches and no multiple-comparison adjustment. Do not interpret a nonsignificant result as proof of equivalent performance. The gallery is fixed at 113 candidates, so random uniform top-1 ranking has expected accuracy 1/113, approximately 0.885%.

See the actual saved [verification summary](../results/matching_verification/verification_summary.csv), [paired comparisons](../results/matching_verification/paired_method_tests.csv), and [verification notes](../results/matching_verification/verification_notes.txt).

## Generated HTML source documentation

The GitHub Actions artifact named `numerical-tests-and-html-docs` contains the HTML pages for scripts 27–31 and the complete text and XML test reports. On GitHub, open a successful workflow run, return to **Summary**, scroll to **Artifacts**, and download and extract the archive. Open `docs/api/index.html` in a browser. These generated HTML files are **not checked into the Git repository**; they are downloadable as the successful workflow artifact.

## Limitations and next checks

- The depth PNG grayscale values are not calibrated physical distances in the present method.
- The segmentation review comprises 25 manually evaluated winter images; the ratings are not pixelwise ground-truth labels.
- The branch/junction prototypes and earlier numbered scripts are not all covered by the 10 current unit tests.
- Matching accuracy remains too low for reliable orchard deployment.
- Dataset redistribution permission must be confirmed with the course teaching team.
