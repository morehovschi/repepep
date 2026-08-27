"""
Corpus run for CK repetition detection.

Computes CK plus two trivial baselines over the annotated recordings and
reports per-recording and pooled metrics for each.

No matrix caching: every change under test (window length, frequency scaling,
image shape, quantizer) invalidates every matrix, so a cache would miss every
time. Instead each run is appended to an experiment log, so the full sweep --
including rejected configs -- can be tabulated at the end without re-running.

Usage from the notebook:

    from run_corpus import run_corpus, report_all, log_experiment, CONFIG

    results = run_corpus(fpath_annotations)
    summaries = report_all(results, split_at=10)
    log_experiment("baseline", CONFIG, results, summaries)

    # later, after changes:
    cfg = dict(CONFIG, window_ms=350)
    results = run_corpus(fpath_annotations, config=cfg)
    summaries = report_all(results, split_at=10)
    log_experiment("window_350ms", cfg, results, summaries)

    from run_corpus import show_log
    show_log()
"""

import os
import time
import pickle
import numpy as np
import essentia.standard as es

from helpers import load_and_detect_onsets
from ck_helpers import compute_ck1_distance, compute_spectrogram_from_chunk
from baselines import dumb_energy, dumb_euclidean
from ck_evaluate_corpus import evaluate_corpus

LOG_PATH = "../data/ck_logs/cache/experiments.pkl"

# The validated configuration. Copy and modify rather than editing in place,
# so the logged config always matches the run that produced the numbers.
CONFIG = {
    "window_ms": 125,
    "target_size": (80, 128),
    "quality": 25,
    "subtract_overhead": False,
    "crop": 128,
    "sr": 44100,
    "log_freq": False,
    "hop_size": 64,
}

# --------------------------------------------------------------------------
# extraction and matrix building
# --------------------------------------------------------------------------

def extract_specs(fpath, config=CONFIG):
    """
    Onsets -> list of dB spectrograms, uncropped.

    Cropping happens at distance time so the baselines can be given exactly
    the same images CK sees.
    """
    _, onset_times = load_and_detect_onsets(fpath)
    audio = es.MonoLoader(filename=fpath, sampleRate=config["sr"])()
    win = int(config["window_ms"] / 1000.0 * config["sr"])

    specs, times = [], []
    for t in onset_times:
        a = int(t * config["sr"])
        if a + win > len(audio):
            continue
        s = compute_spectrogram_from_chunk(
            audio[a:a + win],
            hop_size=config["hop_size"],
            n_bands=config["target_size"][1],
            log_freq=config["log_freq"],
            sr=config["sr"])

        if s is not None:
            specs.append(s)
            times.append(t)
    return specs, times


def build_matrix(specs, dist_fn):
    n = len(specs)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = dist_fn(specs[i], specs[j])
    return m


def _cropped(dist_fn, crop):
    """Give a baseline the same cropped input CK receives."""
    return lambda a, b: dist_fn(a[:crop], b[:crop])


def make_methods(config):
    return {
        "CK": lambda a, b: compute_ck1_distance(
            a, b,
            target_size=config["target_size"],
            quality=config["quality"],
            subtract_overhead=config["subtract_overhead"]),
        "euclidean": _cropped(dumb_euclidean, config["crop"]),
        "energy": _cropped(dumb_energy, config["crop"]),
    }


def run_corpus(fpath_annotations, config=CONFIG, methods=None, verbose=True):
    """
    fpath_annotations : {id: (path, [target onset indices])}

    Returns {method: [(name, matrix, groups), ...]}, ready for evaluate_corpus.
    Iteration order follows fpath_annotations, so the tune/holdout split stays
    consistent across runs as long as that dict is not rebuilt.
    """
    methods = methods or make_methods(config)
    out = {k: [] for k in methods}
    t0 = time.time()

    for idx, (rec_id, (fpath, targets)) in enumerate(fpath_annotations.items(), 1):
        name = os.path.basename(os.path.dirname(fpath)) + "/" + str(rec_id)
        if verbose:
            print(f"[{idx}/{len(fpath_annotations)}] {name}", flush=True)

        specs, _ = extract_specs(fpath, config)
        n = len(specs)

        # Target indices point into the detected-onset list. extract_specs
        # drops onsets whose window runs past the end of the buffer, so an
        # index at or beyond n would silently label nothing.
        kept = [t for t in targets if t < n]
        if len(kept) < len(targets):
            print(f"    WARNING: targets {sorted(set(targets) - set(kept))} "
                  f"exceed {n} usable onsets -- dropped")
        if len(kept) < 2:
            print(f"    SKIP: fewer than 2 target onsets survive")
            continue

        for mname, fn in methods.items():
            out[mname].append((name, build_matrix(specs, fn), [kept]))

    if verbose:
        print(f"\ndone in {time.time() - t0:.0f}s")
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def report_all(results, split_at=10, verbose=True):
    """
    Full comparison across methods, then the tune/holdout split.

    split_at follows the order of fpath_annotations: the first `split_at`
    recordings are the tuning half. Do not look at holdout numbers until you
    have stopped making changes.
    """
    summaries = {}
    for mname, items in results.items():
        if verbose:
            print(f"\n{'=' * 70}\n{mname}\n{'=' * 70}")
        summaries[mname] = evaluate_corpus(items, verbose=verbose)["summary"]

    print(f"\n\n{'=' * 70}\nSUMMARY -- all recordings\n{'=' * 70}")
    print(f"{'method':<12}{'meanAUC':>9}{'poolAUC':>9}{'gap':>8}"
          f"{'meanF1':>8}{'thrCV':>8}{'d':>7}")
    for mname, s in summaries.items():
        print(f"{mname:<12}{s['mean_auc']:>9.3f}{s['pooled_auc']:>9.3f}"
              f"{s['calibration_gap']:>+8.3f}{s['mean_best_f1']:>8.2f}"
              f"{s['threshold_cv']:>8.2f}{s['mean_cohens_d']:>7.2f}")

    n_items = len(next(iter(results.values())))
    if split_at is not None and 0 < split_at < n_items:
        print(f"\n{'=' * 70}\nTUNE (first {split_at}) vs HOLDOUT (rest)\n{'=' * 70}")
        print(f"{'method':<12}{'tune_AUC':>10}{'hold_AUC':>10}"
              f"{'tune_pool':>11}{'hold_pool':>11}")
        for mname, items in results.items():
            t = evaluate_corpus(items[:split_at], verbose=False)["summary"]
            h = evaluate_corpus(items[split_at:], verbose=False)["summary"]
            print(f"{mname:<12}{t['mean_auc']:>10.3f}{h['mean_auc']:>10.3f}"
                  f"{t['pooled_auc']:>11.3f}{h['pooled_auc']:>11.3f}")

    return summaries


# --------------------------------------------------------------------------
# experiment log
# --------------------------------------------------------------------------

def log_experiment(label, config, results, summaries=None, path=LOG_PATH):
    """
    Append one experiment. Stores the matrices too, so any metric added later
    can be re-derived without re-encoding anything.

    Log rejected configs as well -- the sweep table showing what was tried and
    discarded is worth more in a thesis than the winning row alone.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "rb") as f:
            log = pickle.load(f)
    except (FileNotFoundError, EOFError):
        log = []

    if summaries is None:
        summaries = {m: evaluate_corpus(items, verbose=False)["summary"]
                     for m, items in results.items()}

    log.append({"label": label, "config": dict(config),
                "summaries": summaries, "results": results,
                "ts": time.time()})
    with open(path, "wb") as f:
        pickle.dump(log, f)
    print(f"logged '{label}' (entry {len(log)})")
    return len(log)


def load_log(path=LOG_PATH):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return []


def show_log(method="CK", path=LOG_PATH):
    """One row per experiment: the sweep table, straight out of the log."""
    log = load_log(path)
    if not log:
        print("no experiments logged")
        return log
    print(f"{'#':>3}  {'label':<22}{'meanAUC':>9}{'poolAUC':>9}"
          f"{'meanF1':>8}{'thrCV':>8}{'d':>7}")
    print("-" * 68)
    for i, e in enumerate(log, 1):
        s = e["summaries"].get(method)
        if s is None:
            continue
        print(f"{i:>3}  {e['label'][:21]:<22}{s['mean_auc']:>9.3f}"
              f"{s['pooled_auc']:>9.3f}{s['mean_best_f1']:>8.2f}"
              f"{s['threshold_cv']:>8.2f}{s['mean_cohens_d']:>7.2f}")
    return log


def compare_to(baseline_label, candidate_label, method="CK", path=LOG_PATH):
    """
    Apply the accept rule to two logged experiments.

    ACCEPT if mean AUC improves by >= 0.02 AND pooled AUC does not drop by
    more than 0.02. The 0.02 floor reflects the standard error on a mean over
    ~10 recordings -- smaller differences are not distinguishable from noise.
    """
    log = load_log(path)
    by_label = {e["label"]: e for e in log}
    b, c = by_label.get(baseline_label), by_label.get(candidate_label)
    if not b or not c:
        print("label not found; available:", sorted(by_label))
        return None

    bs, cs = b["summaries"][method], c["summaries"][method]
    d_mean = cs["mean_auc"] - bs["mean_auc"]
    d_pool = cs["pooled_auc"] - bs["pooled_auc"]
    accept = (d_mean >= 0.02) and (d_pool >= -0.02)

    print(f"{'metric':<16}{'baseline':>10}{'candidate':>11}{'delta':>9}")
    print("-" * 46)
    for k in ("mean_auc", "pooled_auc", "mean_best_f1",
              "threshold_cv", "mean_cohens_d"):
        print(f"{k:<16}{bs[k]:>10.3f}{cs[k]:>11.3f}{cs[k] - bs[k]:>+9.3f}")
    print(f"\n{'ACCEPT' if accept else 'REJECT'}"
          f"  (mean {d_mean:+.3f} vs +0.02 required, "
          f"pooled {d_pool:+.3f} vs -0.02 floor)")
    return accept
