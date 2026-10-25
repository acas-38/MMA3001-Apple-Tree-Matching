"""
28_segmentation_sensitivity.py

Unsupervised parameter sensitivity study for winter depth segmentation.
No manually drawn masks are needed to run. Optional reference masks can
be supplied independently for genuine segmentation IoU/Dice evaluation.

Output is designed for MMA3001 verification/validation reporting:
  - every image x setting measured, including failures;
  - automatic QUALITY PROXIES explicitly distinguished from ground-truth accuracy;
  - processing time and parameter values recorded;
  - figures for qualitative review.
"""
from pathlib import Path
from time import perf_counter
import argparse
import csv

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # works in VS Code and headless batch mode
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import (
    binary_closing, binary_opening, disk, remove_small_objects
)

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "Data"
OUTPUT_DIR = PROJECT / "results" / "segmentation_sensitivity"
# Keep None to process all trees; five known development examples by default.
DEFAULT_TREES = ["R20N03", "R01N03", "R01N43", "R09N18", "R09N28"]

# Designed to test ONE parameter family at a time (threshold) plus
# a deliberately different cleanup method as a baseline comparison.
EXPERIMENTS = [
    {"name": "baseline_80", "threshold": 80, "closing": 5, "remove_ground": False},
    {"name": "ground_60",   "threshold": 60, "closing": 3, "remove_ground": True},
    {"name": "ground_80",   "threshold": 80, "closing": 3, "remove_ground": True},
    {"name": "ground_100",  "threshold": 100, "closing": 3, "remove_ground": True},
]
MIN_COMPONENT_PIXELS = 120
PLOT_LIMIT = 30
# Ground modelling is image-local, not a global constant row.
# Bands near lower image where the background is typically visible.
GROUND_START_FRAC = 0.48
GROUND_BOTTOM_FRAC = 0.94
GROUND_ROW_FRACTION = 0.58     # fraction foreground per row indicating ground
GROUND_MIN_RUN = 55            # horizontal contiguous pixels
CENTRAL_FRAC = 0.28            # preference for components near image centre

FIELDS = [
    "Tree_ID", "File", "Setting", "Threshold", "Closing_Radius",
    "Ground_Removal", "Status", "Runtime_s", "Input_Nonzero_Pct",
    "Selected_Pixels", "Selected_Pct", "BBox_X0", "BBox_Y0",
    "BBox_X1", "BBox_Y1", "Height_Pct",
    "Bottom_Border_Pixels", "Wide_Lower_Row_Pct",
    "Components_After_Processing", "Ground_Contact_Flag",
    "GT_Available", "IoU_GT", "Dice_GT"
]


def files_for_trees(tree_ids):
    paths = sorted(DATA_DIR.rglob("*_Winter_D-*.png"))
    found = {}
    for p in paths:
        tree_id = p.name.split("_Winter_D-")[0]
        if tree_ids is not None and tree_id not in tree_ids:
            continue
        # deterministic selection; report duplicates rather than pretending all tested
        found.setdefault(tree_id, p)
    return found


def foreground_runs(binary_row):
    """Longest uninterrupted foreground run within one image row."""
    padded = np.r_[False, binary_row.astype(bool), False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def ground_edge_estimate(binary):
    """Return position of first broad ground-like horizontal band.

    This only estimates where ground begins; it is NOT a labelled GT boundary.
    """
    h, w = binary.shape
    a, b = int(h * GROUND_START_FRAC), min(h, int(h * GROUND_BOTTOM_FRAC))
    candidates = []
    for y in range(a, b):
        row = binary[y]
        if row.mean() >= GROUND_ROW_FRACTION and foreground_runs(row) >= GROUND_MIN_RUN:
            candidates.append(y)
    if not candidates:
        return None
    # A few neighbouring rows are needed; isolated horizontal branches aren't ground
    for y in candidates:
        if sum(abs(other - y) <= 5 for other in candidates) >= 3:
            return y
    return None


def process(depth, experiment):
    """Return (selected mask, diagnostics). No manual labels are used."""
    h, w = depth.shape
    binary = depth >= experiment["threshold"]
    ground_y = ground_edge_estimate(binary) if experiment["remove_ground"] else None

    # Conservative guard: pixels significantly below detected soil/ground boundary
    # are removed; an upper trunk ending at ground is intentionally truncated.
    work = binary.copy()
    if ground_y is not None:
        work[min(h, ground_y + 2):, :] = False

    r = experiment["closing"]
    if r:
        work = binary_closing(work, footprint=disk(r))
    work = remove_small_objects(work, min_size=MIN_COMPONENT_PIXELS)

    labels, num = ndi.label(work, structure=np.ones((3, 3), dtype=np.uint8))
    selected = np.zeros_like(binary)
    candidates = []
    for label_id in range(1, num + 1):
        yy, xx = np.where(labels == label_id)
        if len(xx) < MIN_COMPONENT_PIXELS:
            continue
        yspan = yy.max() - yy.min() + 1
        xmed = np.median(xx[yy >= np.quantile(yy, 0.7)])
        # Prioritise a tall tree near centre, penalising horizontally expansive ground.
        xspan = xx.max() - xx.min() + 1
        centrality = max(0, 1 - abs(xmed - w/2) / (w * CENTRAL_FRAC))
        score = 2.5 * yspan/h + 1.2 * centrality - 1.2 * xspan/w
        candidates.append((score, label_id))
    if candidates:
        best_label = max(candidates)[1]
        selected = labels == best_label
        status = "SELECTED"
    else:
        status = "NO_COMPONENT"

    return selected, {
        "status": status,
        "ground_y": ground_y,
        "components": len(candidates),
        "binary": binary
    }


def reference_mask(tree_id, shape, reference_dir):
    """Optional independent GT mask. No reference -> not quantitatively validated."""
    if reference_dir is None:
        return None
    for candidate in (
        reference_dir / f"{tree_id}_mask.npy",
        reference_dir / f"{tree_id}_mask.png",
    ):
        if candidate.exists():
            mask = (
                np.load(candidate).astype(bool)
                if candidate.suffix == ".npy"
                else np.asarray(Image.open(candidate).convert("L")) > 0
            )
            if mask.shape != shape:
                raise ValueError(f"GT shape mismatch: {candidate}")
            return mask
    return None


def metrics(mask, depth, diagnostic, ref):
    h, w = mask.shape
    yy, xx = np.where(mask)
    props = {}
    if not len(xx):
        x0 = y0 = x1 = y1 = ""
        height_pct = 0.0
    else:
        x0, x1, y0, y1 = int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())
        height_pct = 100 * (y1 - y0 + 1) / h

    lower = mask[int(0.52*h):int(0.90*h)]
    wide_lower_rows = float(np.mean(lower.mean(axis=1) > 0.40) * 100) if lower.size else 0.0
    bottom_count = int(mask[-1].sum())
    # Proxy only: wide lower rows/edge contact are warning signs, not true FPR.
    ground_flag = bool(bottom_count > 0 or wide_lower_rows > 10)
    props.update({
        "Input_Nonzero_Pct": round(float(np.mean(depth > 0) * 100), 3),
        "Selected_Pixels": int(mask.sum()),
        "Selected_Pct": round(float(mask.mean() * 100), 3),
        "BBox_X0": x0, "BBox_Y0": y0, "BBox_X1": x1, "BBox_Y1": y1,
        "Height_Pct": round(height_pct, 3),
        "Bottom_Border_Pixels": bottom_count,
        "Wide_Lower_Row_Pct": round(wide_lower_rows, 3),
        "Components_After_Processing": diagnostic["components"],
        "Ground_Contact_Flag": ground_flag,
        "GT_Available": ref is not None,
        "IoU_GT": "", "Dice_GT": ""
    })
    if ref is not None:
        intersection = int(np.count_nonzero(mask & ref))
        union = int(np.count_nonzero(mask | ref))
        denominator = int(mask.sum() + ref.sum())
        props["IoU_GT"] = round(intersection/union, 5) if union else 1.0
        props["Dice_GT"] = round(2*intersection/denominator, 5) if denominator else 1.0
    return props


def diagnostic_figure(depth, entries, tree_id, out_path):
    cols = len(entries) + 1
    fig, axes = plt.subplots(1, cols, figsize=(3.6 * cols, 4.5))
    axes[0].imshow(depth, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original depth")
    for ax, (setting, mask, row) in zip(axes[1:], entries):
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{setting}\n{row['Status']} | {row['Selected_Pct']:.1f}% pixels",
                     fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{tree_id}: segmentation sensitivity (NOT accuracy)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run(tree_ids=DEFAULT_TREES, reference_dir=None, plot_limit=PLOT_LIMIT):
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    found = files_for_trees(tree_ids)
    if not found:
        raise FileNotFoundError(f"No winter depth images found under: {DATA_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (tree_id, path) in enumerate(found.items()):
        try:
            depth = np.asarray(Image.open(path).convert("L"))
            if depth.ndim != 2:
                raise ValueError("Not grayscale")
            ref = reference_mask(tree_id, depth.shape, reference_dir)
            plots = []
            for setting in EXPERIMENTS:
                t0 = perf_counter()
                mask, info = process(depth, setting)
                runtime = perf_counter() - t0
                row = {
                    "Tree_ID": tree_id, "File": path.name,
                    "Setting": setting["name"], "Threshold": setting["threshold"],
                    "Closing_Radius": setting["closing"],
                    "Ground_Removal": setting["remove_ground"],
                    "Status": info["status"], "Runtime_s": round(runtime, 5),
                }
                row.update(metrics(mask, depth, info, ref))
                results.append(row)
                plots.append((setting["name"], mask, row))
            if i < plot_limit:
                diagnostic_figure(depth, plots, tree_id, OUTPUT_DIR / f"{tree_id}_comparison.png")
            print(f"[{i+1}/{len(found)}] {tree_id}: "
                  + ", ".join(f"{r['Setting']}={r['Status']}"
                              for r in results[-len(EXPERIMENTS):]))
        except Exception as exc:
            print(f"ERROR {tree_id}: {exc}")
            for setting in EXPERIMENTS:
                row = dict.fromkeys(FIELDS, "")
                row.update(Tree_ID=tree_id, File=path.name, Setting=setting["name"],
                           Status=f"ERROR:{type(exc).__name__}")
                results.append(row)

    with (OUTPUT_DIR / "measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)

    # Aggregate summary: do not call proxies 'accuracy'.
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["Setting", "N_Images", "N_Selected", "Selection_Rate_Pct",
                      "Mean_Runtime_s", "Mean_Selected_Pct",
                      "Ground_Contact_Warning_Count",
                      "N_With_GT", "Mean_IoU_GT", "Mean_Dice_GT"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for setting in EXPERIMENTS:
            group = [r for r in results if r["Setting"] == setting["name"]]
            selected = [r for r in group if r["Status"] == "SELECTED"]
            gt = [r for r in selected if r["GT_Available"] is True]
            average = lambda field, data: (round(float(np.mean([r[field] for r in data])), 5)
                                          if data else "")
            writer.writerow({
                "Setting": setting["name"], "N_Images": len(group),
                "N_Selected": len(selected),
                "Selection_Rate_Pct": round(100*len(selected)/max(1,len(group)), 2),
                "Mean_Runtime_s": average("Runtime_s", selected),
                "Mean_Selected_Pct": average("Selected_Pct", selected),
                "Ground_Contact_Warning_Count": sum(
                    bool(r["Ground_Contact_Flag"]) for r in selected),
                "N_With_GT": len(gt),
                "Mean_IoU_GT": average("IoU_GT", gt),
                "Mean_Dice_GT": average("Dice_GT", gt)
            })
    print(f"\nSaved {len(results)} measurements to {OUTPUT_DIR}")
    print("IMPORTANT: Without independent GT masks, results are diagnostic "
          "proxies, not validated segmentation accuracy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Run all available winter trees, not only the five test IDs.")
    parser.add_argument("--tree", nargs="*", default=None,
                        help="Choose specific tree IDs, e.g. --tree R01N03 R20N03")
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="Optional independent binary ground-truth masks.")
    parser.add_argument("--plot-limit", type=int, default=PLOT_LIMIT)
    args = parser.parse_args()
    chosen = (None if args.all else args.tree if args.tree else DEFAULT_TREES)
    run(tree_ids=chosen, reference_dir=args.reference_dir, plot_limit=args.plot_limit)
