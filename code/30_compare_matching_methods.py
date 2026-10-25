"""30_compare_matching_methods.py — MMA3001 controlled summer/winter matching.

Run from project root: python code/30_compare_matching_methods.py
Requires numpy, pandas, scipy, Pillow, matplotlib.
No manual mask, prior script output, or precomputed CSV is required.

All compared methods use identical identities/gallery. Set GALLERY_SIZE to 113
for comparability with earlier experiments; the *identities may differ*.
Geometric features from a thresholded component are EXPLORATORY: prior
visual validation demonstrated poor segmentation usability.
"""
from pathlib import Path
from time import perf_counter
import re
import csv
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label, binary_closing, binary_opening
from scipy.ndimage import generate_binary_structure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT / 'results' / 'matching_comparison'
GALLERY_SIZE = 113
RANDOM_SEED = 2026
THRESHOLD = 80
# Match both seasons with the SAME geometry extraction algorithm.
# Do not use Summer_GT: it would create an asymmetric, unfair comparison.


def discover():
    """Find depth images under common dataset folders, and group by tree ID."""
    roots = [PROJECT / 'Data', PROJECT / 'data', PROJECT / 'dataset', PROJECT]
    paths, seen = [], set()
    for root in roots:
        if root.exists():
            for p in root.rglob('*.png'):
                if p in seen:
                    continue
                if ('_Summer_D-' in p.name or '_Winter_D-' in p.name):
                    paths.append(p)
                    seen.add(p)
    records = {}
    for p in sorted(paths):
        m = re.match(r'^(R\d+N\d+)_(Summer|Winter)_D-', p.name, re.I)
        if not m:
            continue
        identity, season = m.group(1).upper(), m.group(2).lower()
        records.setdefault(identity, {}).setdefault(season, []).append(p)
    return {i: (x['summer'][0], x['winter'][0]) for i, x in records.items()
            if x.get('summer') and x.get('winter')}


def image_array(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.float64)


def cell_features(img):
    valid = img[np.isfinite(img)]
    if valid.size == 0:
        return [np.nan, np.nan, np.nan]
    return [float(valid.mean()), float(valid.std()), float(np.mean(valid == 0) * 100)]


def global_features(img):
    return np.asarray(cell_features(img), float)


def spatial_features(img):
    return np.asarray([x for row in np.array_split(img, 3, axis=0)
                       for patch in np.array_split(row, 3, axis=1)
                       for x in cell_features(patch)], float)


def geometry_features(img):
    """Coarse, symmetric geometric shape descriptors.

    No assumption that the selected foreground is a correctly isolated tree.
    These are exploratory image-shape features, NOT verified branch measures.
    """
    binary = img >= THRESHOLD
    binary = binary_opening(binary, iterations=1)
    binary = binary_closing(binary, iterations=2)
    lab, count = label(binary, structure=generate_binary_structure(2, 2))
    h, w = img.shape
    if count == 0:
        return np.full(8, np.nan)
    # Select component with greatest vertical extent, preferring centre;
    # penalty for huge horizontal ground-like components.
    best, best_score = None, -np.inf
    for k in range(1, count + 1):
        yy, xx = np.where(lab == k)
        if len(xx) < 100:
            continue
        span_y = yy.max()-yy.min()+1
        span_x = xx.max()-xx.min()+1
        centre = np.median(xx)
        score = 2*span_y/h - span_x/w - abs(centre-w/2)/w
        if score > best_score:
            best, best_score = (yy, xx), score
    if best is None:
        return np.full(8, np.nan)
    yy, xx = best
    y_min, y_max = yy.min(), yy.max()
    height = max(1, y_max-y_min)
    widths, offsets = [], []
    for frac in (0.25, 0.50, 0.75):
        row = y_min + int(round(height*frac))
        # use short vertical window to reduce single-row noise
        present = xx[np.abs(yy - row) <= 2]
        if len(present) < 2:
            widths.append(np.nan)
            offsets.append(np.nan)
        else:
            widths.append(float((present.max()-present.min()+1)/w))
            offsets.append(float(((present.max()+present.min())/2 - w/2)/w))
    if not np.all(np.isfinite(widths + offsets)):
        return np.full(8, np.nan)
    # Normalise by image dimensions, not per-tree labels or oracle masks
    return np.asarray([
        widths[0], widths[1], widths[2],
        offsets[0], offsets[1], offsets[2],
        float(height/h), float(len(xx)/(h*w))
    ])


METHODS = {'global': global_features, 'spatial_3x3': spatial_features,
           'geometry_exploratory': geometry_features}


def compare(summer, winter):
    """Unsupervised column scaling on WINTER GALLERY only.

    All gallery/query indices refer to same sorted tree list. Missing
    features get failed predictions, not removed from the denominator.
    """
    gallery_ok = np.isfinite(winter).all(axis=1)
    query_ok = np.isfinite(summer).all(axis=1)
    valid = gallery_ok & query_ok
    n, d = summer.shape
    rank = np.full(n, np.nan)
    pred = np.full(n, -1, dtype=int)
    if gallery_ok.any():
        g = winter[gallery_ok]
        mu = np.mean(g, axis=0)
        sd = np.std(g, axis=0)
        sd = np.where(sd < 1e-8, 1, sd)
        gallery_indexes = np.flatnonzero(gallery_ok)
        g = (g - mu) / sd
        for i in np.flatnonzero(query_ok):
            q = (summer[i] - mu)/sd
            scores = np.sum((g-q)**2, axis=1)
            order = gallery_indexes[np.argsort(scores, kind='stable')]
            pred[i] = int(order[0])
            if i in order:
                rank[i] = int(np.where(order == i)[0][0]) + 1
    # Missing features count as NOT CORRECT. Always use full N denominator.
    return rank, pred, valid, int(gallery_ok.sum())


def main():
    pairs = discover()
    if not pairs:
        raise FileNotFoundError('No matching Summer_D/Winter_D pairs found under project Data/ folder')
    all_ids = sorted(pairs)
    if len(all_ids) < GALLERY_SIZE:
        print(f'WARNING: Only {len(all_ids)} paired identities found (requested {GALLERY_SIZE}).')
        print('Metrics will use all available identities. Do not compare random chance to 1/113.')
        ids = all_ids
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        ids = sorted(rng.choice(all_ids, size=GALLERY_SIZE, replace=False).tolist())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Tree_ID': ids,
                  'Summer_File': [pairs[i][0].name for i in ids],
                  'Winter_File': [pairs[i][1].name for i in ids]}).to_csv(
                      OUTPUT / 'gallery_manifest.csv', index=False)
    n = len(ids)
    print(f'Comparing {n} paired trees, seed={RANDOM_SEED}, methods={list(METHODS)}')
    feature_data = {m: {'summer': [], 'winter': [], 'seconds': []} for m in METHODS}
    for j, tree_id in enumerate(ids, 1):
        a = image_array(pairs[tree_id][0]); b = image_array(pairs[tree_id][1])
        if a.shape != b.shape:
            # Different image sizes allowed; each feature extractor is scale-independent
            pass
        for name, fn in METHODS.items():
            tic = perf_counter()
            fa = fn(a); fb = fn(b)
            feature_data[name]['seconds'].append(perf_counter()-tic)
            feature_data[name]['summer'].append(fa)
            feature_data[name]['winter'].append(fb)
        if j % 20 == 0 or j == n:
            print(f'  Processed {j}/{n}')

    summary, detail = [], []
    for name, dat in feature_data.items():
        summer = np.vstack(dat['summer']); winter = np.vstack(dat['winter'])
        rank, pred, both_ok, gallery_count = compare(summer, winter)
        top1 = int(np.sum(rank == 1)); top3 = int(np.sum(rank <= 3))
        summary.append(dict(Method=name, N_Queries=n, N_Gallery=n,
                            N_Feature_Pairs_Valid=int(both_ok.sum()),
                            N_Gallery_Usable=gallery_count,
                            Top1_Correct=top1, Top1_Accuracy_Pct=round(100*top1/n, 3),
                            Top3_Correct=top3, Top3_Recall_Pct=round(100*top3/n, 3),
                            Mean_Feature_Time_s=round(float(np.mean(dat['seconds'])), 6),
                            Failed_Queries=int(np.sum(~np.isfinite(rank)))))
        for i, identity in enumerate(ids):
            detail.append(dict(Method=name, Tree_ID=identity,
                               Predicted_Tree=(ids[pred[i]] if pred[i]>=0 else ''),
                               Rank=(int(rank[i]) if np.isfinite(rank[i]) else ''),
                               Correct_Top1=bool(rank[i] == 1),
                               Correct_Top3=bool(rank[i] <= 3),
                               Feature_Pair_Valid=bool(both_ok[i])))
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT / 'matching_summary.csv', index=False)
    pd.DataFrame(detail).to_csv(OUTPUT / 'per_tree_predictions.csv', index=False)
    print('\n--- CONTROLLED MATCHING COMPARISON ---')
    print(summary_df.to_string(index=False))
    print(f'Random top-1: {100/n:.3f}% | random top-3: {100*min(3,n)/n:.3f}%')
    labels = [x['Method'] for x in summary]
    plt.figure(figsize=(9, 5))
    places = np.arange(len(labels)); bw = 0.36
    plt.bar(places-bw/2, summary_df.Top1_Accuracy_Pct, width=bw, label='Top-1')
    plt.bar(places+bw/2, summary_df.Top3_Recall_Pct, width=bw, label='Top-3')
    plt.xticks(places, labels, rotation=12, ha='right')
    plt.ylabel('Correct matching (%)')
    plt.title(f'Same {n}-tree gallery; exploratory geometry')
    plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT / 'matching_comparison.png', dpi=150); plt.close()
    print(f'\nSaved to: {OUTPUT}')
    print('CAUTION: This is a NEW common-cohort experiment, not a direct reproduction')
    print('of earlier 3.54%/11.50% unless gallery IDs, features and settings match.')
    print('No model fitting/tuning on held-out labels is used; geometry is unvalidated.')


if __name__ == '__main__':
    main()
