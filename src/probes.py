"""
Single-segment probes for CK-1.

These take ONE spectrogram, perturb it in a controlled way, and measure the
CK distance to the unperturbed original. Because both images come from the
same audio, any distance above ~0 is measurement noise rather than acoustic
difference -- which makes these the cleanest way to find out what the metric
is actually sensitive to.

Interpreting the shift probe is the point of this module. Two onsets of the
same sound never align perfectly: the detector fires a few milliseconds early
or late. If CK degrades sharply with small shifts, then onset jitter alone
explains your 0.82 scores and no amount of spectrogram tuning will help until
you align segments. If CK is flat under shift, MPEG's motion compensation is
doing its job and the problem lies elsewhere.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def shift_probe(spec, ck_fn, shifts_frames=(0, 1, 2, 4, 8, 16), axis=1):
    """
    Roll the spectrogram along time (axis=1) by k native frames and measure
    CK against the original.

    ck_fn : a callable taking (spec_a, spec_b) -> distance, i.e. your
            compute_ck1_distance with any keyword args already bound, e.g.
            lambda a, b: compute_ck1_distance(a, b, target_size=(256,256))

    Note shifts are in NATIVE frames, not output pixels. With 79 native
    frames upsampled to 256, one native frame is ~3.2 pixels.
    """
    print("shift probe (time axis)")
    print(f"{'frames':>8}{'~pixels':>9}{'ms @hop64':>11}{'CK':>9}")
    scale = 256.0 / spec.shape[axis]
    out = []
    for k in shifts_frames:
        rolled = np.roll(spec, k, axis=axis)
        d = ck_fn(spec, rolled)
        ms = k * 64 / 44100.0 * 1000
        out.append((k, d))
        print(f"{k:>8}{k * scale:>9.1f}{ms:>11.1f}{d:>9.4f}")
    return out


def blur_probe(spec, ck_fn, sigmas=(0.0, 0.5, 1.0, 2.0, 4.0)):
    """
    Smooth along time and measure CK against the original. Tells you how much
    of your byte budget is going into fine temporal detail. If small sigmas
    already produce large CK, the encoder is spending most of its bits on
    texture that carries no acoustic meaning.
    """
    print("blur probe (time axis)")
    print(f"{'sigma':>8}{'CK':>9}")
    out = []
    for s in sigmas:
        b = spec if s == 0 else gaussian_filter1d(spec, s, axis=1)
        d = ck_fn(spec, b)
        out.append((s, d))
        print(f"{s:>8.1f}{d:>9.4f}")
    return out


def gain_probe(spec, ck_fn, gains_db=(0, 3, 6, 12)):
    """
    Add a constant dB offset (a pure loudness change) and measure CK.

    Under per-image min-max normalization this SHOULD return ~0 for every
    gain, since the offset cancels. If it does not, the normalization is
    interacting with the dB clip floor at -100 and loudness is leaking into
    your distances -- which would mean quiet and loud instances of the same
    call can never match.
    """
    print("gain probe")
    print(f"{'dB':>8}{'CK':>9}")
    out = []
    for g in gains_db:
        d = ck_fn(spec, spec + g)
        out.append((g, d))
        print(f"{g:>8}{d:>9.4f}")
    return out


def run_all(spec, ck_fn):
    shift_probe(spec, ck_fn)
    print()
    blur_probe(spec, ck_fn)
    print()
    gain_probe(spec, ck_fn)
