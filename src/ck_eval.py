"""
Evaluation metrics for pairwise distance matrices from CK-1 audio similarity.

Design notes
------------
* Every metric here depends only on the RANKING of distances, not their
  absolute values. CK-1 has a large, preprocessing-dependent floor
  (empirically ~0.6-0.9), so any metric that is not rank-invariant will
  mostly measure that floor moving around.
* Unlabeled onsets are treated as singleton classes: a pair is "positive"
  iff both members belong to the same *labeled* group. This handles the
  multi-group case (raven + gibbon in one recording) without special-casing.

Metrics
-------
auc      : normalized Mann-Whitney U / area under ROC. Probability that a
           random same-class pair is ranked closer than a random
           different-class pair. 0.5 = chance. Rank-invariant.
map      : mean average precision over queries drawn from labeled groups.
           Standard MIR retrieval measure; more sensitive than AUC to the
           top of the ranking.
nn1      : fraction of labeled onsets whose nearest neighbour shares its
           group (1-NN accuracy, as used throughout the Keogh-group papers).
motif_top1 : is the global off-diagonal argmin a within-group pair?
             This is Definition 5 of Hao et al. evaluated directly.
"""

import numpy as np


def _rankdata_average(a):
    """Ranks 1..n with ties averaged. Avoids a scipy dependency."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def make_labels(n, groups):
    """
    Build a per-onset class label array.

    n      : total number of onsets
    groups : list of index lists, one per repeated-sound group, e.g.
             [[4, 5, 6, 7], [11, 12, 14, 15]]

    Unlabeled onsets get unique negative labels (singleton classes).
    """
    labels = np.full(n, -1, dtype=int)
    for g_id, idxs in enumerate(groups):
        labels[np.asarray(idxs, dtype=int)] = g_id
    singleton_id = len(groups)
    for i in range(n):
        if labels[i] == -1:
            labels[i] = singleton_id
            singleton_id += 1
    return labels


def _pair_arrays(matrix, labels, origin=None):
    """
    Flatten the strict upper triangle into (distances, is_same_class).

    origin : optional array giving, for each row of `matrix`, the index of the
             onset it came from. Pairs whose two members share an origin are
             dropped. Only relevant during bootstrap resampling, where the same
             onset can be drawn twice and would otherwise contribute a
             free zero-distance positive pair.
    """
    n = matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    keep = np.ones(len(iu[0]), dtype=bool)
    if origin is not None:
        keep = origin[iu[0]] != origin[iu[1]]
    dists = matrix[iu][keep]
    same = (labels[iu[0]] == labels[iu[1]])[keep]
    return dists, same


def pairwise_auc(matrix, labels, origin=None):
    """
    AUC over pairs. Positives are same-class pairs, scored by -distance.
    Rank-invariant: unaffected by any monotone transform of the matrix.
    """
    dists, same = _pair_arrays(matrix, labels, origin)
    n_pos, n_neg = int(same.sum()), int((~same).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = _rankdata_average(-dists)          # higher score = more similar
    r_pos = ranks[same].sum()
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def mean_average_precision(matrix, labels):
    """MAP over every onset belonging to a group of size >= 2."""
    n = matrix.shape[0]
    aps = []
    for q in range(n):
        rel_total = int((labels == labels[q]).sum()) - 1
        if rel_total < 1:
            continue
        others = np.array([i for i in range(n) if i != q])
        order = others[np.argsort(matrix[q, others], kind="mergesort")]
        hits, ap = 0, 0.0
        for rank, idx in enumerate(order, start=1):
            if labels[idx] == labels[q]:
                hits += 1
                ap += hits / rank
        aps.append(ap / rel_total)
    return float(np.mean(aps)) if aps else np.nan


def nn1_accuracy(matrix, labels):
    """Fraction of non-singleton onsets whose nearest neighbour matches."""
    n = matrix.shape[0]
    correct, total = 0, 0
    for q in range(n):
        if int((labels == labels[q]).sum()) < 2:
            continue
        others = np.array([i for i in range(n) if i != q])
        nn = others[np.argmin(matrix[q, others])]
        correct += int(labels[nn] == labels[q])
        total += 1
    return correct / total if total else np.nan


def motif_top1(matrix, labels):
    """Definition 5: is the closest off-diagonal pair a within-group pair?"""
    dists, same = _pair_arrays(matrix, labels)
    return bool(same[np.argmin(dists)])


def bootstrap_ci(matrix, labels, fn=pairwise_auc, n_boot=2000, seed=0):
    """
    Percentile CI by resampling ONSETS (not pairs), since pairs sharing an
    onset are not independent. With ~10 onsets the interval will be wide --
    that is the honest picture, not a defect of the procedure.
    """
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub_m = matrix[np.ix_(idx, idx)].astype(float).copy()
        np.fill_diagonal(sub_m, 0.0)
        v = fn(sub_m, labels[idx], origin=idx)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def evaluate(matrix, groups, name="", n_boot=2000):
    """
    Full report for one recording.

    matrix : symmetric distance matrix, zero diagonal
    groups : list of index lists, e.g. [[3, 5, 7, 9]]
    """
    matrix = np.asarray(matrix, dtype=float)
    labels = make_labels(matrix.shape[0], groups)
    dists, same = _pair_arrays(matrix, labels)

    auc = pairwise_auc(matrix, labels)
    lo, hi = bootstrap_ci(matrix, labels, pairwise_auc, n_boot=n_boot)

    return {
        "name": name,
        "n_onsets": int(matrix.shape[0]),
        "n_pos_pairs": int(same.sum()),
        "n_neg_pairs": int((~same).sum()),
        "auc": auc,
        "auc_ci95": (lo, hi),
        "map": mean_average_precision(matrix, labels),
        "nn1": nn1_accuracy(matrix, labels),
        "motif_top1": motif_top1(matrix, labels),
        # Diagnostics -- not scores. Use these to explain WHY a score moved.
        "median_within": float(np.median(dists[same])) if same.any() else np.nan,
        "median_between": float(np.median(dists[~same])) if (~same).any() else np.nan,
        "range_all": (float(dists.min()), float(dists.max())),
    }


def summarize(reports):
    """One line per recording, for scanning a parameter sweep."""
    header = (f"{'recording':<22}{'AUC':>7}{'95% CI':>16}{'MAP':>7}"
              f"{'1NN':>6}{'top1':>7}{'medW':>7}{'medB':>7}")
    lines = [header, "-" * len(header)]
    for r in reports:
        ci = f"[{r['auc_ci95'][0]:.2f},{r['auc_ci95'][1]:.2f}]"
        lines.append(
            f"{r['name']:<22}{r['auc']:>7.3f}{ci:>16}{r['map']:>7.3f}"
            f"{r['nn1']:>6.2f}{str(r['motif_top1']):>7}"
            f"{r['median_within']:>7.3f}{r['median_between']:>7.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    microwave = np.array([
        [0.,.87874668,.87067534,.85031292,.87690419,.86579167,.88457374,.86631565,.89200817,.86745516,.88798176],
        [.87874668,0.,.84592964,.84367089,.81924816,.86019944,.82881164,.8551811,.82747057,.8683945,.82981939],
        [.87067534,.84592964,0.,.86835529,.83104051,.86881271,.83980546,.86955645,.83955453,.8824598,.84165246],
        [.85031292,.84367089,.86835529,0.,.84525486,.63468797,.82401802,.65454888,.88298855,.67291012,.88307406],
        [.87690419,.81924816,.83104051,.84525486,0.,.84846996,.80538302,.86371595,.83138739,.85485907,.83098935],
        [.86579167,.86019944,.86881271,.63468797,.84846996,0.,.86210526,.63308497,.88682588,.64729848,.90195098],
        [.88457374,.82881164,.83980546,.82401802,.80538302,.86210526,0.,.83313493,.84155865,.84387269,.83524344],
        [.86631565,.8551811,.86955645,.65454888,.86371595,.63308497,.83313493,0.,.89174091,.64153894,.89289789],
        [.89200817,.82747057,.83955453,.88298855,.83138739,.88682588,.84155865,.89174091,0.,.89536283,.80401184],
        [.86745516,.8683945,.8824598,.67291012,.85485907,.64729848,.84387269,.64153894,.89536283,0.,.89019776],
        [.88798176,.82981939,.84165246,.88307406,.83098935,.90195098,.83524344,.89289789,.80401184,.89019776,0.],
    ])
    raven = np.array([
        [0.,.80544926,.80741808,.80104142,.78995774,.79705387,.79202159,.79401882,.76672254,.76831774],
        [.80544926,0.,.79624299,.8,.79473815,.80164734,.81869474,.8182352,.79694157,.80031912],
        [.80741808,.79624299,0.,.80204568,.80909463,.80800599,.80947219,.81627907,.80591163,.80540703],
        [.80104142,.8,.80204568,0.,.79433042,.79417044,.80532717,.81677045,.78926226,.79115877],
        [.78995774,.79473815,.80909463,.79433042,0.,.80066049,.82266961,.82379863,.79385857,.77121013],
        [.79705387,.80164734,.80800599,.79417044,.80066049,0.,.81859856,.81445604,.78800803,.78956159],
        [.79202159,.81869474,.80947219,.80532717,.82266961,.81859856,0.,.81900495,.79061584,.79693775],
        [.79401882,.8182352,.81627907,.81677045,.82379863,.81445604,.81900495,0.,.79911534,.79781648],
        [.76672254,.79694157,.80591163,.78926226,.79385857,.78800803,.79061584,.79911534,0.,.77974435],
        [.76831774,.80031912,.80540703,.79115877,.77121013,.78956159,.79693775,.79781648,.77974435,0.],
    ])

    for r in [evaluate(microwave, [[3, 5, 7, 9]], "microwave"),
              evaluate(raven, [[4, 5, 6, 7]], "raven")]:
        print(f"{r['name']:<12} AUC={r['auc']:.3f} "
              f"CI=[{r['auc_ci95'][0]:.2f},{r['auc_ci95'][1]:.2f}] "
              f"MAP={r['map']:.3f} 1NN={r['nn1']:.2f} "
              f"top1={r['motif_top1']} "
              f"medW={r['median_within']:.3f} medB={r['median_between']:.3f}")
