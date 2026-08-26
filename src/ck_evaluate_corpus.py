"""
Corpus-level evaluation for CK repetition detection.

Two things this adds over per-recording evaluate():

1. SEPARATION, not just ranking. AUC says a threshold exists for THIS
   recording. It does not say the threshold is usable, or that the same one
   works on the next recording. For "is this segment a repetition?" you need
   margin, not just correct ordering.

2. POOLED vs PER-RECORDING. Per-recording AUC can be 1.000 on every file
   while a single global threshold fails, because each file's distances sit
   at a different absolute level. Pooling all pairs into one ranking tests
   whether one threshold generalises. A large mean-minus-pooled gap means the
   method works only with per-recording calibration.
"""

import numpy as np


def separation(matrix, groups):
    """
    Threshold quality for one recording.

    gap_abs uses single extreme pairs and is very noisy with small groups --
    reported for completeness but do not optimise against it. Track cohens_d
    and best_f1 instead.
    """
    n = matrix.shape[0]
    lab = {}
    for gid, g in enumerate(groups):
        for i in g:
            lab[i] = gid

    iu = np.triu_indices(n, 1)
    same = np.array([(i in lab and j in lab and lab[i] == lab[j])
                     for i, j in zip(*iu)])
    d = matrix[iu]
    w, b = d[same], d[~same]
    if len(w) == 0 or len(b) == 0:
        return None

    pooled_sd = np.sqrt((w.var() + b.var()) / 2)

    # best F1 over all candidate thresholds: "distance <= t" predicts same-class
    order = np.argsort(d)
    best_f1, best_t = 0.0, float(d.min())
    tp = fp = 0
    n_pos = int(same.sum())
    for k in order:
        if same[k]:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / n_pos
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(d[k])

    return {
        "cohens_d": float((b.mean() - w.mean()) / pooled_sd) if pooled_sd > 0 else np.inf,
        "best_f1": best_f1,
        "best_threshold": best_t,
        # percentile gap: robust alternative to min/max, still in raw units
        "gap_p90_p10": float(np.percentile(b, 10) - np.percentile(w, 90)),
        "gap_abs": float(b.min() - w.max()),
        "medW": float(np.median(w)),
        "medB": float(np.median(b)),
        "n_within": len(w),
        "n_between": len(b),
    }


def _auc(d, same):
    n_pos, n_neg = int(same.sum()), int((~same).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(-d, kind="mergesort")
    ranks = np.empty(len(d), float)
    ranks[order] = np.arange(1, len(d) + 1)
    return (ranks[same].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def evaluate_corpus(items, verbose=True):
    """
    items : list of (name, matrix, groups)

    Returns a dict with per-recording rows plus pooled statistics.
    """
    rows, all_d, all_same = [], [], []

    for name, m, groups in items:
        m = np.asarray(m, dtype=float)
        n = m.shape[0]
        lab = {}
        for gid, g in enumerate(groups):
            for i in g:
                lab[i] = gid
        iu = np.triu_indices(n, 1)
        same = np.array([(i in lab and j in lab and lab[i] == lab[j])
                         for i, j in zip(*iu)])
        d = m[iu]
        if same.sum() == 0 or (~same).sum() == 0:
            continue

        s = separation(m, groups)
        rows.append({"name": name, "auc": _auc(d, same), **s})
        all_d.append(d)
        all_same.append(same)

    all_d = np.concatenate(all_d)
    all_same = np.concatenate(all_same)

    pooled_auc = _auc(all_d, all_same)
    mean_auc = float(np.mean([r["auc"] for r in rows]))
    thresholds = np.array([r["best_threshold"] for r in rows])

    summary = {
        "n_recordings": len(rows),
        "mean_auc": mean_auc,
        "pooled_auc": pooled_auc,
        "calibration_gap": mean_auc - pooled_auc,
        "mean_cohens_d": float(np.mean([r["cohens_d"] for r in rows])),
        "mean_best_f1": float(np.mean([r["best_f1"] for r in rows])),
        "threshold_mean": float(thresholds.mean()),
        "threshold_sd": float(thresholds.std()),
        "threshold_cv": float(thresholds.std() / thresholds.mean())
                        if thresholds.mean() != 0 else np.inf,
    }

    if verbose:
        print(f"{'recording':<28}{'AUC':>7}{'d':>7}{'F1':>7}{'thr':>7}"
              f"{'medW':>7}{'medB':>7}")
        print("-" * 70)
        for r in sorted(rows, key=lambda x: x["auc"]):
            print(f"{r['name'][:27]:<28}{r['auc']:>7.3f}{r['cohens_d']:>7.2f}"
                  f"{r['best_f1']:>7.2f}{r['best_threshold']:>7.3f}"
                  f"{r['medW']:>7.3f}{r['medB']:>7.3f}")
        print("-" * 70)
        print(f"mean AUC (per-recording) : {mean_auc:.3f}")
        print(f"pooled AUC (all pairs)   : {pooled_auc:.3f}")
        print(f"calibration gap          : {summary['calibration_gap']:+.3f}"
              f"   (large => needs per-recording threshold)")
        print(f"mean Cohen's d           : {summary['mean_cohens_d']:.2f}")
        print(f"mean best F1             : {summary['mean_best_f1']:.2f}")
        print(f"threshold spread (CV)    : {summary['threshold_cv']:.2f}"
              f"   (small => one global threshold viable)")

    return {"rows": rows, "summary": summary}
