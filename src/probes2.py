"""
Probes v2 -- unit-correct, audio-domain.

Changes from v1:
  * hop size and output width are parameters, not hardcoded constants, so the
    reported ms and pixel figures stay honest when you change the pipeline
  * the shift probe slides the window along the real audio instead of rolling
    the spectrogram, which is what onset jitter actually does (np.roll wraps
    the tail around to the head and fabricates a discontinuity)
  * shift_tolerance_ms() collapses the curve to one scalar you can put in a
    sweep table next to AUC
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def shift_probe_audio(audio, onset_time, ck_fn, spec_fn,
                      shifts_ms=(0, 1, 2, 4, 6, 8, 12, 16, 24, 32),
                      window_ms=125, sr=44100, verbose=True):
    """
    Slide the extraction window along the audio and measure CK against the
    unshifted segment. This simulates onset-detector jitter directly.

    audio       : full-file mono buffer
    onset_time  : timestamp (s) of a clean, representative event
    ck_fn       : (spec_a, spec_b) -> distance
    spec_fn     : (audio_segment) -> spectrogram, i.e. your
                  compute_spectrogram_from_chunk with any params bound

    Returns a list of (shift_ms, ck) pairs.
    """
    win = int(window_ms / 1000.0 * sr)
    base_start = int(onset_time * sr)
    ref = spec_fn(audio[base_start:base_start + win])

    if verbose:
        print(f"{'shift_ms':>9}{'CK':>9}")
    out = []
    for ms in shifts_ms:
        off = int(ms / 1000.0 * sr)
        a = base_start + off
        if a < 0 or a + win > len(audio):
            continue
        d = ck_fn(ref, spec_fn(audio[a:a + win]))
        out.append((ms, d))
        if verbose:
            print(f"{ms:>9}{d:>9.4f}")
    return out


def shift_tolerance_ms(curve, threshold=0.2):
    """
    Largest shift (ms) at which CK is still below `threshold`, with linear
    interpolation between the bracketing samples.

    This is the number to track across configs. Onset detectors commonly
    jitter 10-20 ms, so a tolerance below ~15 ms means alignment error alone
    can push two instances of the same sound to the no-match plateau.
    """
    prev_ms, prev_ck = 0.0, 0.0
    for ms, ck in curve:
        if ck >= threshold:
            if ck == prev_ck:
                return prev_ms
            frac = (threshold - prev_ck) / (ck - prev_ck)
            return prev_ms + frac * (ms - prev_ms)
        prev_ms, prev_ck = ms, ck
    return float(curve[-1][0])


def blur_probe(spec, ck_fn, hop=64, sr=44100,
               sigmas_ms=(0.0, 1.0, 2.0, 4.0, 8.0), verbose=True):
    """
    Smooth along time by a given number of MILLISECONDS (converted to frames
    using the actual hop), so results stay comparable when hop changes.

    High sensitivity here means the encoder is spending its bits on fine
    temporal texture. Two utterances of the same call differ at exactly that
    scale, so a steep curve caps how well CK can ever match natural variation.
    """
    frame_ms = hop / sr * 1000.0
    if verbose:
        print(f"{'sigma_ms':>9}{'sigma_fr':>9}{'CK':>9}")
    out = []
    for ms in sigmas_ms:
        s = ms / frame_ms
        b = spec if s == 0 else gaussian_filter1d(spec, s, axis=1)
        d = ck_fn(spec, b)
        out.append((ms, d))
        if verbose:
            print(f"{ms:>9.1f}{s:>9.2f}{d:>9.4f}")
    return out


def gain_probe(spec, ck_fn, gains_db=(0, 3, 6, 12), verbose=True):
    """
    Constant dB offset. Should be ~0 under any max-relative normalization.
    Re-run this ONLY when you change normalization -- in particular, a
    global per-file scheme is deliberately NOT gain-invariant, so nonzero
    values there are the intended behaviour, not a bug.
    """
    out = []
    if verbose:
        print(f"{'dB':>6}{'CK':>9}")
    for g in gains_db:
        d = ck_fn(spec, spec + g)
        out.append((g, d))
        if verbose:
            print(f"{g:>6}{d:>9.4f}")
    return out


def distractor_check(specs, ck_fn, same_pairs, diff_pairs, verbose=True):
    """
    Guard against winning the shift probe by destroying discrimination.

    Blur, downsampling, and coarse quantization all raise shift tolerance --
    and all of them, taken far enough, make every sound look identical. This
    reports the gap between pairs that SHOULD match and pairs that should
    NOT. If tolerance improves while this gap shrinks toward zero, the change
    is buying invariance by throwing away signal.

    same_pairs / diff_pairs : lists of (i, j) index tuples into `specs`
    """
    same = [ck_fn(specs[i], specs[j]) for i, j in same_pairs]
    diff = [ck_fn(specs[i], specs[j]) for i, j in diff_pairs]
    gap = float(np.mean(diff) - np.mean(same))
    if verbose:
        print(f"  mean CK same-class: {np.mean(same):.4f}")
        print(f"  mean CK diff-class: {np.mean(diff):.4f}")
        print(f"  gap (want large):   {gap:+.4f}")
    return gap
