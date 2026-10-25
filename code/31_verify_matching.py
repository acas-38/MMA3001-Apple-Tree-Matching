"""31_verify_matching.py — verify and quantify controlled matching results.

Run from project root:
    python code/31_verify_matching.py
    python code/31_verify_matching.py --self-test

Reads the exact output of 30_compare_matching_methods.py, not raw images.
Uses all query identities (including failed feature extraction) as denominator.
Confidence intervals quantify uncertainty; they do NOT establish generalisation.
"""
from pathlib import Path
import argparse
from itertools import combinations
import math
import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT = Path(__file__).resolve().parent.parent
INPUT = PROJECT / 'results' / 'matching_comparison'
OUTPUT = PROJECT / 'results' / 'matching_verification'
ALPHA = 0.05
RANDOM_SEED = 2026
RANDOM_TRIALS = 20000


def wilson_interval(k, n, alpha=ALPHA):
    if n <= 0:
        return math.nan, math.nan
    # Standard normal critical value for 95% two-sided CI
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n))/den
    margin = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0, centre-margin), min(1, centre+margin)


def mcnemar_exact(a, b):
    """Exact two-sided McNemar test from paired binary outcomes."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    discordant = only_a + only_b
    p = float(binomtest(min(only_a, only_b), discordant, 0.5).pvalue) if discordant else 1.0
    return only_a, only_b, discordant, p


def verify_metric_calculation():
    ranks = np.array([1, 2, 3, 4, np.nan])
    assert int(np.sum(ranks == 1)) == 1
    assert int(np.sum(ranks <= 3)) == 3
    assert int(np.sum(~np.isfinite(ranks))) == 1
    assert 0 <= wilson_interval(0, 113)[0] <= wilson_interval(0, 113)[1] <= 1
    assert 0 <= wilson_interval(113, 113)[0] <= wilson_interval(113, 113)[1] <= 1
    assert mcnemar_exact([1,0,1], [0,1,1]) == (1,1,2,1.0)
    assert mcnemar_exact([1,1,0], [1,1,0]) == (0,0,0,1.0)
    # Synthetic rank and top-k behaviour, including missing prediction
    miniature = pd.DataFrame({'Tree_ID':['A','B','C','D'],
                              'Predicted_Tree':['A','X','C',''],
                              'Rank':[1,2,3,np.nan]})
    assert (miniature['Rank'] == 1).sum() == 1
    assert (miniature['Rank'] <= 3).sum() == 3
    return True


def normalise_bool(series):
    s = series.astype(str).str.lower().str.strip()
    if not s.isin(['true','false','1','0']).all():
        raise ValueError('Invalid boolean values in predictions')
    return s.isin(['true', '1']).to_numpy()


def load_and_verify(folder):
    predictions_path = folder / 'per_tree_predictions.csv'
    summary_path = folder / 'matching_summary.csv'
    manifest_path = folder / 'gallery_manifest.csv'
    for f in [predictions_path, summary_path, manifest_path]:
        if not f.exists():
            raise FileNotFoundError(f'Missing {f}. Run script 30 first.')
    detail = pd.read_csv(predictions_path, keep_default_na=True)
    summary = pd.read_csv(summary_path)
    manifest = pd.read_csv(manifest_path)
    required = {'Method','Tree_ID','Predicted_Tree','Rank','Correct_Top1','Correct_Top3','Feature_Pair_Valid'}
    if not required.issubset(detail.columns):
        raise ValueError(f'Missing prediction fields: {sorted(required - set(detail.columns))}')
    if manifest['Tree_ID'].duplicated().any():
        raise ValueError('Gallery manifest has duplicate Tree_ID rows')
    ids = manifest['Tree_ID'].astype(str).tolist()
    if not ids:
        raise ValueError('Empty manifest')
    if summary['Method'].duplicated().any():
        raise ValueError('Duplicate methods in matching_summary.csv')
    methods = summary['Method'].astype(str).tolist()
    if set(detail.Method.unique()) != set(methods):
        raise ValueError('Methods disagree between predictions and summary')
    checked = []
    for method in methods:
        g = detail[detail.Method == method].copy()
        if g['Tree_ID'].duplicated().any() or len(g) != len(ids) or set(g.Tree_ID) != set(ids):
            raise ValueError(f'{method}: missing/duplicate/mismatched query identities')
        g = g.set_index('Tree_ID').loc[ids].reset_index()
        r = pd.to_numeric(g['Rank'], errors='coerce').to_numpy(float)
        top1 = r == 1
        top3 = r <= min(3,len(ids))
        valid = normalise_bool(g.Feature_Pair_Valid)
        if not np.array_equal(top1,normalise_bool(g.Correct_Top1)):
            raise ValueError(f'{method}: Correct_Top1 disagrees with Rank')
        if not np.array_equal(top3,normalise_bool(g.Correct_Top3)):
            raise ValueError(f'{method}: Correct_Top3 disagrees with Rank')
        finite = np.isfinite(r)
        if np.any((r[finite] < 1) | (r[finite] > len(ids)) | (r[finite] != np.floor(r[finite]))):
            raise ValueError(f'{method}: invalid match rank')
        if np.any(~valid & finite):
            raise ValueError(f'{method}: invalid feature pair has a valid rank')
        preds = g.Predicted_Tree.fillna('').astype(str).to_numpy()
        if np.any(top1 & (preds != np.array(ids))):
            raise ValueError(f'{method}: rank 1 does not match predicted identity')
        if np.any(finite & ~np.isin(preds, ids)):
            raise ValueError(f'{method}: predicted identity absent from gallery')
        s = summary.loc[summary.Method == method].iloc[0]
        counts = {'N_Queries':len(ids), 'N_Gallery':len(ids),
                  'N_Feature_Pairs_Valid':int(valid.sum()),
                  'Top1_Correct':int(top1.sum()), 'Top3_Correct':int(top3.sum()),
                  'Failed_Queries':int((~finite).sum())}
        for field, count in counts.items():
            if int(s[field]) != count:
                raise ValueError(f'{method}: {field} CSV={s[field]}, recomputed={count}')
        for field, count in [('Top1_Accuracy_Pct',counts['Top1_Correct']),
                             ('Top3_Recall_Pct',counts['Top3_Correct'])]:
            expected=100*count/len(ids)
            if abs(float(s[field])-expected)>0.005:
                raise ValueError(f'{method}: {field} mismatch')
        checked.append((method,g,top1,top3,valid))
    return ids, checked


def random_ranking_trial(n, trials, seed):
    """Random-ranking baseline: each query receives a uniform random rank.

    This preserves the number of gallery candidates but does not enforce a
    one-to-one assignment across queries. It is NOT a paired permutation test.
    """
    rng=np.random.default_rng(seed)
    counts1=np.empty(trials,dtype=int)
    counts3=np.empty(trials,dtype=int)
    for i in range(trials):
        # random independent ranking of each query: ranks uniform in 1..n
        # (valid even though the true matching is a bijection)
        ranks=rng.integers(1,n+1,size=n)
        counts1[i]=int(np.sum(ranks==1))
        counts3[i]=int(np.sum(ranks<=min(n,3)))
    return counts1,counts3


def run(folder=INPUT, output=OUTPUT, trials=RANDOM_TRIALS):
    verify_metric_calculation()
    ids, checked = load_and_verify(folder)
    n=len(ids)
    output.mkdir(parents=True,exist_ok=True)
    random_top1=1/n
    random_top3=min(3,n)/n
    random_counts1,random_counts3=random_ranking_trial(n,trials,RANDOM_SEED)

    results=[]
    for method,g,top1,top3,valid in checked:
        for metric,correct,chance,random_counts in [
            ('top1',top1,random_top1,random_counts1),
            ('top3',top3,random_top3,random_counts3)]:
            k=int(correct.sum())
            lo,hi=wilson_interval(k,n)
            # Exact one-sided binomial null; 113 queries are not necessarily
            # independent in real image matching, so interpret cautiously.
            p=binomtest(k,n,chance,alternative='greater').pvalue
            p_sim=(1+np.sum(random_counts>=k))/(trials+1)
            results.append(dict(Method=method,Metric=metric,N=n,Correct=k,
                Rate_Pct=round(100*k/n,3),Wilson_95_Lower_Pct=round(100*lo,3),
                Wilson_95_Upper_Pct=round(100*hi,3),Random_Chance_Pct=round(100*chance,3),
                Exact_Binomial_One_Sided_P=round(float(p),6),
                Random_Rank_MC_Tail_P=round(float(p_sim),6),
                N_Feature_Pairs_Valid=int(valid.sum())))

    comparisons=[]
    for (a,ga,a1,a3,_),(b,gb,b1,b3,_) in combinations(checked,2):
        for metric,aa,bb in [('top1',a1,b1),('top3',a3,b3)]:
            only_a,only_b,discord,p=mcnemar_exact(aa,bb)
            comparisons.append(dict(Method_A=a,Method_B=b,Metric=metric,
                A_Only_Correct=only_a,B_Only_Correct=only_b,
                Both_Correct=int(np.sum(aa & bb)),Both_Incorrect=int(np.sum(~aa & ~bb)),
                N_Discordant=discord,Exact_McNemar_P=round(p,6),
                Difference_B_Minus_A_Pct=round(100*(bb.mean()-aa.mean()),3)))
    df=pd.DataFrame(results)
    comp=pd.DataFrame(comparisons)
    df.to_csv(output/'verification_summary.csv',index=False)
    comp.to_csv(output/'paired_method_tests.csv',index=False)
    with (output/'verification_notes.txt').open('w',encoding='utf-8') as f:
        f.write(f'Queries = gallery = {n}\n')
        f.write('CSV metrics independently recalculated and consistency-checked.\n')
        f.write('Failed queries remain in denominators.\n')
        f.write('Wilson 95% confidence intervals describe uncertainty under a binomial model;\n')
        f.write('repeated measurements/correlated queries can violate independence.\n')
        f.write('One-sided binomial p-values compare to uniform random ranking;\n')
        f.write('do not treat non-significance as proof of equivalence.\n')
        f.write(f'Monte Carlo rank-baseline uses {trials} independent simulated ranking sets, seed={RANDOM_SEED}.\n')
        f.write('McNemar exact tests compare success on the SAME identities.\n')
        f.write('Pairwise p-values are exploratory and not multiple-comparison adjusted.\n')
        f.write('No pixel-level ground-truth segmentation accuracy is estimated here.\n')
    print('\n--- VERIFIED MATCHING RESULTS ---')
    print(df.to_string(index=False))
    print('\n--- PAIRED METHOD COMPARISONS ---')
    print(comp.to_string(index=False))
    print(f'\nPASS: all summary counts/flags agree with prediction rankings; saved to {output}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--input-dir',type=Path,default=INPUT)
    p.add_argument('--trials',type=int,default=RANDOM_TRIALS)
    args=p.parse_args()
    if args.self_test:
        verify_metric_calculation()
        print('PASS: metric, Wilson interval, McNemar and missing-rank unit tests')
        return
    if args.trials<100:
        p.error('--trials must be at least 100')
    run(folder=args.input_dir,trials=args.trials)

if __name__=='__main__':
    main()