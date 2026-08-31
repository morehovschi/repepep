"""
MFCC baseline.

The other three distances all operate on the spectrogram images built for the
CK pipeline. MFCC is a different feature entirely and is computed from the
audio segments directly, so it needs its own extraction path. Output is in the
same (name, matrix, groups) format the corpus evaluation expects, and the same
onsets and target indices are used, so the results line up row-for-row with
the other methods.

Configuration follows common practice for timbre similarity rather than the
CK pipeline's STFT settings: a 1024-sample frame with 512 hop, 40 mel bands,
13 coefficients. Using the CK pipeline's 512/64 framing would handicap the
baseline for no reason -- the point is to compare against a properly
configured MFCC representation, not a deliberately matched one.

Two variants are reported:
  drop_c0=True   coefficient 0 tracks overall level, so excluding it makes the
                 representation loudness-invariant and therefore comparable to
                 CK and to the Euclidean spectrogram baseline
  drop_c0=False  retains level information, comparable to the loudness baseline
"""

import numpy as np
import essentia.standard as es

from helpers import load_and_detect_onsets


def segment_mfccs(fpath, window_ms=125, sr=44100,
                  frame_size=1024, hop_size=512,
                  n_bands=40, n_coeffs=13):
    """
    One mean-MFCC vector per onset segment.

    Frames within the 125 ms window are averaged, which discards temporal
    structure -- a deliberate simplification, and one reason MFCC means are a
    weaker representation than they might be for percussive events whose
    information is concentrated in the attack.
    """
    _, onset_times = load_and_detect_onsets(fpath)
    audio = es.MonoLoader(filename=fpath, sampleRate=sr)()
    win = int(window_ms / 1000.0 * sr)

    windowing = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    mfcc = es.MFCC(inputSize=frame_size // 2 + 1,
                   numberBands=n_bands,
                   numberCoefficients=n_coeffs,
                   sampleRate=sr)

    vectors, times = [], []
    for t in onset_times:
        a = int(t * sr)
        if a + win > len(audio):
            continue
        seg = audio[a:a + win]
        coeffs = []
        for frame in es.FrameGenerator(seg, frameSize=frame_size,
                                       hopSize=hop_size):
            _, c = mfcc(spectrum(windowing(frame)))
            coeffs.append(c)
        if not coeffs:
            continue
        vectors.append(np.mean(np.array(coeffs), axis=0))
        times.append(t)

    return vectors, times


def cosine_distance(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def run_mfcc(fpath_annotations, window_ms=125, sr=44100, drop_c0=True,
             verbose=True):
    """
    Returns [(name, matrix, groups), ...] matching the format and ordering of
    run_corpus output, so it can be evaluated with evaluate_corpus and placed
    alongside the other methods.
    """
    import os
    out = []
    for idx, (rec_id, (fpath, targets)) in enumerate(fpath_annotations.items(), 1):
        name = os.path.basename(os.path.dirname(fpath)) + "/" + str(rec_id)
        if verbose:
            print(f"[{idx}/{len(fpath_annotations)}] {name}", flush=True)

        vectors, _ = segment_mfccs(fpath, window_ms=window_ms, sr=sr)
        n = len(vectors)

        kept = [t for t in targets if t < n]
        if len(kept) < 2:
            print(f"    SKIP: fewer than 2 target onsets survive")
            continue

        feats = [v[1:] if drop_c0 else v for v in vectors]

        m = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                m[i, j] = m[j, i] = cosine_distance(feats[i], feats[j])
        out.append((name, m, [kept]))

    return out
