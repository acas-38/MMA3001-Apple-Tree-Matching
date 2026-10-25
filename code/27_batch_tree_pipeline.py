from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import binary_closing, remove_small_objects, skeletonize, disk, binary_opening, rectangle

# CONFIGURATION
PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / 'Data'
OUT = PROJECT / 'results' / 'batch_junctions'
OUT.mkdir(parents=True, exist_ok=True)
TREE_IDS = TREE_IDS = [
    "R20N03",
    "R01N03",
    "R01N43",
    "R09N18",
    "R09N28"
]  # None: find every *_Winter_D-*.png ; or ['R01N03', 'R20N03']
THRESHOLD = 80
GROUND_LINE_LENGTH = 110
GROUND_START = 0.55
GROUND_END = 0.86
CLOSING_RADIUS = 5
MIN_COMPONENT_PIXELS = 120
MIN_PATH_LENGTH = 70
MIN_ENDPOINT_SPACING = 30
MIN_BRANCH_LENGTH = 25
MERGE_DISTANCE = 12
MAX_SEED_DISTANCE = 110
SAVE_PLOTS = True


def image_files():
    found = sorted(DATA.rglob('*_Winter_D-*.png'))
    if TREE_IDS is not None:
        found = [p for p in found if any(p.name.startswith(t + '_Winter_D-') for t in TREE_IDS)]
    # one deterministic file per tree; multiple images per tree require a separate policy
    out = {}
    for path in found:
        tree_id = path.name.split('_Winter_D-')[0]
        out.setdefault(tree_id, path)
    return out


def preprocess(depth):
    h, w = depth.shape
    foreground = depth >= THRESHOLD
    # Target the long horizontal ground connection, not horizontal branches in general.
    band = np.zeros_like(foreground)
    band[int(h * GROUND_START):int(h * GROUND_END), :] = True
    horizontal = binary_opening(foreground & band, footprint=rectangle(1, GROUND_LINE_LENGTH))
    separated = foreground & ~horizontal
    separated = binary_closing(separated, footprint=disk(CLOSING_RADIUS))
    separated = remove_small_objects(separated, min_size=MIN_COMPONENT_PIXELS)
    # Guard against lower background merging: keep the central component whose pixels
    # reach relatively far down, without imposing a narrow vertical crop.
    labels, count = ndi.label(separated, structure=np.ones((3, 3), dtype=int))
    if not count:
        return None, horizontal
    candidates = []
    for i in range(1, count + 1):
        yy, xx = np.where(labels == i)
        if len(xx) < MIN_COMPONENT_PIXELS:
            continue
        low = yy >= int(h * .52)
        if not low.any():
            continue
        # image centre is a weak prior, not a hard boundary
        xlow = np.median(xx[low])
        ylow = np.max(yy)
        score = (ylow / h) * 90 - abs(xlow - w / 2) * 0.35 + np.log1p(len(xx)) * 3
        candidates.append((score, i))
    if not candidates:
        return None, horizontal
    component_id = max(candidates)[1]
    mask = labels == component_id
    return mask, horizontal


def graph_from_skeleton(skeleton):
    h, w = skeleton.shape
    ys, xs = np.where(skeleton)
    n = len(xs)
    pixel_index = np.full((h, w), -1, dtype=np.int32)
    pixel_index[ys, xs] = np.arange(n)
    rows, cols, weights = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = ys + dy, xs + dx
            valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            source = np.flatnonzero(valid)
            target = pixel_index[ny[valid], nx[valid]]
            keep = target >= 0
            source, target = source[keep], target[keep]
            rows.extend(source.tolist())
            cols.extend(target.tolist())
            weights.extend([np.hypot(dy, dx)] * len(source))
    return ys, xs, pixel_index, csr_matrix((weights, (rows, cols)), shape=(n, n))


def path_length(path, xs, ys):
    if len(path) < 2:
        return 0.0
    return float(np.hypot(np.diff(xs[path]), np.diff(ys[path])).sum())


def map_junctions(skeleton):
    h, w = skeleton.shape
    ys, xs, ids, graph = graph_from_skeleton(skeleton)
    if len(xs) < 2:
        return None
    # Automatically root at a low skeleton point near the centre of selected component.
    lower = ys >= np.quantile(ys, .90)
    if not np.any(lower):
        return None
    low_ids = np.where(lower)[0]
    score = ys[low_ids] - .22 * abs(xs[low_ids] - np.median(xs[low_ids]))
    start = int(low_ids[np.argmax(score)])
    root_x, root_y = int(xs[start]), int(ys[start])
    dist, pred = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
    neighbour_count = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode='constant') - skeleton.astype(np.uint8)
    ey, ex = np.where(skeleton & (neighbour_count == 1))
    endpoints = []
    for x, y in zip(ex, ey):
        ix = ids[y, x]
        if np.isfinite(dist[ix]) and dist[ix] >= MIN_PATH_LENGTH and y < root_y - 35:
            endpoints.append((float(dist[ix]), int(x), int(y), int(ix)))
    endpoints.sort(reverse=True)
    selected = []
    for e in endpoints:
        _, x, y, _ = e
        if all(np.hypot(x - other[1], y - other[2]) >= MIN_ENDPOINT_SPACING for other in selected):
            selected.append(e)
    paths = []
    for _, x, y, ix in selected:
        path, cur, seen = [], ix, set()
        while cur != start and cur >= 0 and cur not in seen:
            seen.add(cur)
            path.append(int(cur))
            cur = int(pred[cur])
        if cur == start:
            path.append(start)
            paths.append(path[::-1])
    candidate = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            a, b = paths[i], paths[j]
            shared = 0
            for pa, pb in zip(a, b):
                if pa != pb:
                    break
                shared += 1
            if shared < 1:
                continue
            if min(path_length(a[shared - 1:], xs, ys), path_length(b[shared - 1:], xs, ys)) < MIN_BRANCH_LENGTH:
                continue
            k = a[shared - 1]
            candidate.append((int(xs[k]), int(ys[k])))
    # Transitive merge of close junction candidates.
    parent = list(range(len(candidate)))
    def find(a):
        while a != parent[a]:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(len(candidate)):
        for j in range(i + 1, len(candidate)):
            if np.hypot(candidate[i][0] - candidate[j][0], candidate[i][1] - candidate[j][1]) < MERGE_DISTANCE:
                parent[find(j)] = find(i)
    groups = defaultdict(list)
    for i, item in enumerate(candidate):
        groups[find(i)].append(item)
    junctions = []
    for items in groups.values():
        arr = np.asarray(items)
        junctions.append({'X': int(np.rint(arr[:, 0].mean())), 'Y': int(np.rint(arr[:, 1].mean())), 'Support': len(items)})
    junctions.sort(key=lambda p: p['Y'], reverse=True)
    return (root_x, root_y), (ys, xs), paths, selected, junctions


def plot_result(depth, mask, horizontal, skeleton, mapped, tree_id):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(depth, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title('Original winter depth')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Automatically selected component')
    ax[2].imshow(depth, cmap='gray', vmin=0, vmax=255)
    overlay = np.ma.masked_where(~skeleton, skeleton)
    ax[2].imshow(overlay, cmap='autumn', vmin=0, vmax=1, alpha=.5)
    if mapped:
        (rx, ry), (ys, xs), paths, endpoints, junctions = mapped
        for i, p in enumerate(paths):
            ax[2].plot(xs[p], ys[p], linewidth=2, color=plt.cm.tab10(i % 10))
        ax[2].scatter(rx, ry, s=60, color='lime')
        for e in endpoints:
            ax[2].scatter(e[1], e[2], s=15, color='orange')
        for i, j in enumerate(junctions, 1):
            ax[2].scatter(j['X'], j['Y'], marker='x', color='red', s=90)
            ax[2].text(j['X'] + 5, j['Y'] - 5, f'J{i}', color='red')
    ax[2].set_title('Auto junctions (inspect visually)')
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / f'{tree_id}_automatic.png', dpi=130)
    plt.close(fig)


def main():
    files = image_files()
    if not files:
        raise FileNotFoundError(f'No *_Winter_D-*.png images found in {DATA}')
    print(f'Found {len(files)} distinct winter trees')
    summary, junction_rows = [], []
    for number, (tree_id, path) in enumerate(files.items(), 1):
        status = 'OK'
        n_endpoints, n_junctions = 0, 0
        try:
            depth = np.asarray(Image.open(path).convert('L'))
            mask, ground = preprocess(depth)
            if mask is None:
                status = 'NO_CENTRAL_COMPONENT'
            else:
                skeleton = skeletonize(mask)
                mapped = map_junctions(skeleton)
                if mapped is None:
                    status = 'NO_SKELETON'
                else:
                    root, _, paths, endpoints, junctions = mapped
                    n_endpoints, n_junctions = len(paths), len(junctions)
                    if n_endpoints < 2:
                        status = 'TOO_FEW_ENDPOINTS'
                    elif not junctions:
                        status = 'NO_JUNCTIONS'
                    for i, j in enumerate(junctions, 1):
                        junction_rows.append({'Tree_ID': tree_id, 'Junction_ID': i, **j, 'Root_X': root[0], 'Root_Y': root[1]})
                if SAVE_PLOTS:
                    plot_result(depth, mask, ground, skeleton, mapped, tree_id)
        except Exception as ex:
            status = f'ERROR: {type(ex).__name__}: {ex}'
        summary.append({'Tree_ID': tree_id, 'Status': status, 'Upper_Endpoints': n_endpoints, 'Junction_Count': n_junctions, 'Image': str(path)})
        print(f'[{number}/{len(files)}] {tree_id}: {status}, endpoints={n_endpoints}, junctions={n_junctions}')
    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    pd.DataFrame(junction_rows, columns=['Tree_ID','Junction_ID','X','Y','Support','Root_X','Root_Y']).to_csv(OUT / 'junctions.csv', index=False)
    print(f'Finished; results in {OUT}')
    print('IMPORTANT: These are candidates, not validated detections. Review sample plots before using for matching.')

if __name__ == '__main__':
    main()
