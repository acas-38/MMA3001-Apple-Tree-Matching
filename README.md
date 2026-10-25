# Cross-Season Apple Tree Matching — MMA3001

**MMA3001: Numerical Methods and Machine Learning**

This individual engineering project investigates whether photographs and depth-derived features can identify the same apple tree across summer and winter observations. Seasonal changes in foliage obscure branches and substantially change image appearance. Reliable cross-season identification could help an orchard robot associate repeated observations with the same physical tree, although this project is an exploratory image-based study rather than a deployable navigation system.

## Research question and scope

**Question:** Can a summer depth image be matched to the correct winter depth image from a gallery of candidate apple trees, and do spatial or approximate geometric features improve on a simple global image-statistics baseline?

The repository contains exploratory work on depth-image segmentation, trunk/branch detection, skeletonisation, junction mapping and cross-season nearest-neighbour matching. The primary quantitative comparison uses three feature-extraction methods on the **same 113-tree gallery**. The project does **not** establish robust tree identification, metric 3-D skeleton reconstruction or field-ready performance.

## Dataset and inputs

The `Data/` directory contains course-provided PNG observations, with names such as `R01N01_Summer_D-12-40-50.png`. The leading `RxxNxx` token is used as the tree identity. The filename also indicates season and image modality:

| Token | Description |
| --- | --- |
| `Summer_RGB` / `Winter_RGB` | Colour observations |
| `Summer_D` / `Winter_D` | Depth-labelled image files used for most experiments |
| `Summer_GT` | Supplied ground-truth-labelled images; **not** used as an oracle mask by the final comparison |

**Important input limitation:** Code currently reads depth PNGs as grayscale image intensity. The physical depth calibration and units have not been verified here; thresholds such as 60, 80 and 100 are therefore **image-intensity thresholds**, not distances in millimetres or metres. Filenames supply matching identity labels, but they do not independently demonstrate consistent camera pose. Missing or unsuitable feature vectors are counted as failed matches in the controlled comparison.

## Repository organisation

```text
MMA3001-Apple-Tree-Matching/
├── Data/                     # Course-provided images
├── code/
│   ├── 01_inspect_data.py      # Dataset exploration
│   ├── 02_... to 26_...        # Exploratory feature and branch algorithms
│   ├── 27_batch_tree_pipeline.py
│   ├── 28_segmentation_sensitivity.py
│   ├── 29_visual_segmentation_validation.py
│   └── 30_compare_matching_methods.py
├── results/
│   ├── batch_junctions/       # Branch/junction experiment outputs
│   ├── segmentation_sensitivity/
│   ├── visual_validation/
│   └── matching_comparison/
├── requirements.txt
└── README.md
```

The numbered scripts document **development stages**, not a single end-to-end pipeline. In particular, experimental junction-count outputs should not be interpreted as validated tree geometry. The key reproducible analyses are described below.

## Installation

Use Python in a virtual environment. From the repository root:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The scripts use relative paths resolved from their own location, so keep the `Data/`, `code/` and `results/` directories in their existing positions. Some early exploratory scripts display plots interactively.

## Running the key experiments

Run from the repository root:

| Command | Purpose | Output |
| --- | --- | --- |
| `python code/27_batch_tree_pipeline.py` | Automatic branch/junction attempt on five configured examples | `results/batch_junctions/` |
| `python code/28_segmentation_sensitivity.py` | Four segmentation settings on five development examples | `results/segmentation_sensitivity/` |
| `python code/28_segmentation_sensitivity.py --all` | Sensitivity study over available winter tree identities | Same directory |
| `python code/29_visual_segmentation_validation.py` | Repeatable, manually rated 25-tree visual audit (interactive) | `results/visual_validation/` |
| `python code/30_compare_matching_methods.py` | Controlled 113-identity cross-season matching comparison | `results/matching_comparison/` |

The visual-audit script prompts for judgements on correct tree selection, retention of main branches and ground contamination. It preserves its selected sample and previous ratings when rerun; do not delete its CSV outputs if you intend to resume that audit. The matching comparison uses a fixed random seed and saves an explicit gallery manifest.

## Methods

### Segmentation and geometry exploration

The winter-depth experiments threshold grayscale images, apply morphology, remove or estimate ground-connected regions, select candidate components, and attempt skeleton/junction extraction. The segmentation sensitivity study varies thresholds and ground removal and records failure status, elapsed time, selected-pixel fraction and diagnostic ground-contact flags. **These automatic diagnostics are proxies, not ground-truth segmentation accuracy.** Pixelwise intersection-over-union (IoU) and Dice require independently supplied reference masks.

### Cross-season matching

`30_compare_matching_methods.py` pairs summer/winter images by tree ID and compares:

1. **Global statistics:** mean pixel value, standard deviation and percentage of zero-valued pixels.
2. **Spatial 3 × 3 statistics:** those three measures in each of nine image regions.
3. **Exploratory geometry:** normalized widths, offsets, vertical extent and foreground fraction of a threshold-derived component.

Each method uses exactly the same selected tree identities and winter gallery. Feature columns are standardised from the winter gallery without using tree-identity labels to fit the scaling statistics; nearest-neighbour ranking uses squared Euclidean feature distance. Invalid features do not silently reduce the query denominator. This is a controlled comparison of *features*, not a validated trained machine-learning model.

## Evaluation and observed results

### Cross-season matching (113 queries and 113 gallery identities)

| Method | Top-1 correct | Top-1 accuracy | Top-3 correct | Top-3 recall |
| --- | ---: | ---: | ---: | ---: |
| Global statistics | 1 | 0.885% | 3 | 2.655% |
| Spatial 3 × 3 | 1 | 0.885% | 6 | 5.310% |
| Exploratory geometry | 2 | 1.770% | 6 | 5.310% |

For this 113-candidate gallery, random top-1 selection would have an expected success rate of about **0.885%**. Geometry performed slightly better than the two intensity baselines on top-1, but the absolute matching performance remains very low. These numbers should **not** be presented as evidence of dependable identification. See [matching summary](results/matching_comparison/matching_summary.csv), [gallery manifest](results/matching_comparison/gallery_manifest.csv) and [per-tree predictions](results/matching_comparison/per_tree_predictions.csv).

### Segmentation visual audit (25 independently sampled tree IDs)

A manually rated audit reviewed the four methods on the same 25 selected winter images, excluding five known development examples by default. A mask was counted as fully usable only if it selected the target tree, retained major branches and did not contain significant ground contamination.

| Setting | Fully usable masks | Usable rate |
| --- | ---: | ---: |
| `baseline_80` | 2 / 25 | 8% |
| `ground_60` | 5 / 25 | 20% |
| `ground_80` | 6 / 25 | 24% |
| `ground_100` | 7 / 25 | 28% |

Ground removal improved the visual usability measure over the baseline on this sample, but **even the best setting failed the combined criteria for 18 of 25 trees**. These are human review results, not labelled pixelwise ground truth or an independently established final test-set accuracy. See [visual validation summary](results/visual_validation/visual_validation_summary.csv). The smaller [five-image sensitivity summary](results/segmentation_sensitivity/summary.csv) reports diagnostic proxies only; its IoU/Dice fields are empty because no independent pixel masks were supplied.

## Verification, limitations and future work

- Matching evaluation uses fixed identities and a common candidate gallery, with the cohort recorded to CSV.
- The visual audit has a seeded sample, explicit review questions and resumable ratings.
- Intermediate segmentation measurements distinguish failure, proxy statistics and optional ground-truth metrics.
- Current experiments do **not** prove tree-level recognition under independent field conditions; there is no demonstrated generalisation across orchards or camera viewpoints.
- Thresholding can merge the tree with soil/background or lose fine branches. Skeleton junction counts are especially sensitive to segmentation errors.
- The exploratory matching geometry uses thresholded components without evidence that they accurately represent the target tree.
- This repository currently needs dedicated automated unit tests, generated HTML API documentation and a stated licence before it satisfies all submission requirements.

Recommended next steps are synthetic-mask tests of morphological/skeleton operations, independent labels for segmentation quality, repeatable train/development/test protocols where parameters are tuned, and more informative cross-season descriptors. Runtime and robustness should be compared on the same dataset and hardware.

## Academic context and AI disclosure

The code and experiments were developed for MMA3001. AI assistance (ChatGPT) was used during development and in preparing documentation, including ideas for algorithms, debugging and explanation of evaluation methods. The student's project report should separately document the **actual extent of AI contribution**, which outputs were independently checked, any AI errors encountered, and decisions made personally. This README is not a substitute for that reflection.

## Submission checklist

- [x] Public repository containing code, data and experimental outputs
- [x] README with installation, methods, evaluation and limitations
- [x] Python dependency manifest
- [ ] Independently run clean-environment reproduction of the documented commands
- [ ] Automated test functions and saved test report
- [ ] Generated HTML source-code documentation (e.g. pdoc)
- [ ] Licence chosen and added by the repository owner
- [ ] Verify docstrings, code comments and file provenance
- [ ] Submit 5–10-page project report via Moodle
- [ ] Submit zipped clone of the GitHub repository via Moodle
- [ ] Present and explain tests, limitations and AI contribution

**Dataset provenance/licence:** The image files are course-provided. Confirm redistribution permission and provide the dataset citation before reusing the data outside the assessment.
