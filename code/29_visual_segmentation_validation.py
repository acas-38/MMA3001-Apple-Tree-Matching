"""MMA3001: reproducible human visual audit of segmentation settings.
Run: python code/29_visual_segmentation_validation.py
Optional: --sample-size 25 --seed 2026 --open-images
Uses process() and EXPERIMENTS from 28_segmentation_sensitivity.py.
A visual audit is NOT pixelwise ground-truth IoU or a final independent test.
"""
from pathlib import Path
import argparse
import csv
import importlib.util
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'visual_validation'
KNOWN_DEV = {'R20N03', 'R01N03', 'R01N43', 'R09N18', 'R09N28'}
FIELDS = ['Tree_ID', 'Setting', 'Target_Selected', 'Main_Branches_Preserved',
          'Ground_Contamination', 'Reviewer_Note', 'Review_Status']


def load_previous_script():
    path = Path(__file__).with_name('28_segmentation_sensitivity.py')
    if not path.exists():
        raise FileNotFoundError(f'Place 28_segmentation_sensitivity.py next to this file: {path}')
    spec = importlib.util.spec_from_file_location('segmentation_sensitivity', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review_choice(prompt):
    while True:
        answer = input(prompt + ' [y/n/u]: ').strip().lower()
        if answer in ('y', 'n', 'u'):
            return {'y': 'Yes', 'n': 'No', 'u': 'Uncertain'}[answer]
        print('Type y (yes), n (no), or u (uncertain).')


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_chart(depth, masks, tree_id, path):
    fig, axes = plt.subplots(1, len(masks) + 1, figsize=(17, 5))
    axes[0].imshow(depth, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original: ' + tree_id)
    for ax, (name, mask, info) in zip(axes[1:], masks):
        ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{name}\n{info["status"]} / {mask.mean()*100:.1f}% pixels')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def summarise(rows):
    path = OUT / 'visual_validation_summary.csv'
    fields = ['Setting', 'N_Reviewed', 'N_Target_Yes', 'N_Branches_Yes',
              'N_Ground_Yes', 'N_Fully_Usable', 'Fully_Usable_Pct',
              'N_Uncertain_Any']
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name in sorted({r['Setting'] for r in rows}):
            group = [r for r in rows if r['Setting'] == name]
            usable = sum(r['Target_Selected']=='Yes' and
                         r['Main_Branches_Preserved']=='Yes' and
                         r['Ground_Contamination']=='No' for r in group)
            writer.writerow({
                'Setting': name, 'N_Reviewed': len(group),
                'N_Target_Yes': sum(r['Target_Selected']=='Yes' for r in group),
                'N_Branches_Yes': sum(r['Main_Branches_Preserved']=='Yes' for r in group),
                'N_Ground_Yes': sum(r['Ground_Contamination']=='Yes' for r in group),
                'N_Fully_Usable': usable,
                'Fully_Usable_Pct': round(100 * usable/len(group), 2) if group else '',
                'N_Uncertain_Any': sum('Uncertain' in (r['Target_Selected'],
                         r['Main_Branches_Preserved'], r['Ground_Contamination']) for r in group)
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=25)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--open-images', action='store_true', help='Open comparison PNG automatically on Windows')
    parser.add_argument('--include-development', action='store_true')
    args = parser.parse_args()

    module = load_previous_script()
    found = module.files_for_trees(None)
    if not args.include_development:
        found = {k: v for k, v in found.items() if k not in KNOWN_DEV}
    if not found:
        raise RuntimeError('No eligible winter images found in Data folder')
    ids = sorted(found)
    sample = sorted(random.Random(args.seed).sample(ids, min(len(ids), args.sample_size)))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'validation_sample.csv'
    # On resuming, preserve original sample instead of silently changing it.
    if manifest.exists():
        with manifest.open(newline='', encoding='utf-8') as f:
            sample = [r['Tree_ID'] for r in csv.DictReader(f)]
        missing = [tree for tree in sample if tree not in found]
        if missing:
            raise RuntimeError(f'Previously selected images are missing: {missing}')
    else:
        with manifest.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Tree_ID', 'Filename', 'Random_Seed', 'Development_Included'])
            for tree in sample:
                writer.writerow([tree, found[tree].name, args.seed, args.include_development])

    ratings_path = OUT / 'visual_ratings.csv'
    if ratings_path.exists():
        with ratings_path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []
    completed = {(r['Tree_ID'], r['Setting']) for r in rows}
    print(f'\nReview set: {len(sample)} trees. {len(rows)} previous ratings loaded.')
    print('For each mask: y/n/u for the target tree, main branch completeness, and ground contamination.')
    print('Branch completeness is meaningful only if the correct tree was selected.')
    print(f'Images and ratings will be saved to: {OUT}\n')

    for number, tree_id in enumerate(sample, 1):
        path = found[tree_id]
        depth = np.asarray(Image.open(path).convert('L'))
        masks = [(setting['name'], *module.process(depth, setting))
                 for setting in module.EXPERIMENTS]
        plot_path = OUT / f'{tree_id}_review.png'
        if not plot_path.exists():
            make_chart(depth, masks, tree_id, plot_path)
        pending = [(name, mask) for name, mask, _ in masks if (tree_id, name) not in completed]
        if not pending:
            continue
        print(f'\n[{number}/{len(sample)}] {tree_id}: {plot_path}')
        if args.open_images and os.name == 'nt':
            os.startfile(plot_path)
        print('Open the PNG to review it. Ctrl+C can interrupt safely after completed rows.')
        for name, mask in pending:
            print(f'  {name} / selected {mask.mean()*100:.1f}% of pixels')
            selected = review_choice('    Correct target tree selected?')
            preserved = review_choice('    Main branches preserved?') if selected == 'Yes' else 'Uncertain'
            contaminated = review_choice('    Significant ground included?')
            note = input('    Short note (optional): ').strip()
            rows.append(dict(Tree_ID=tree_id, Setting=name, Target_Selected=selected,
                             Main_Branches_Preserved=preserved,
                             Ground_Contamination=contaminated,
                             Reviewer_Note=note, Review_Status='Completed'))
            completed.add((tree_id, name))
            save_csv(ratings_path, rows)
            summarise(rows)
    print('\nReview complete. See visual_ratings.csv, visual_validation_summary.csv and validation_sample.csv')
    print('Do not claim these judgments are pixel-accurate segmentation IoU or final held-out test accuracy.')


if __name__ == '__main__':
    main()
