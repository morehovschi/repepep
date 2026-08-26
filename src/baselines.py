"""
Sanity baselines for CK-1.

The question these answer: is the MPEG compression machinery actually doing
work, or would something trivial score just as well on these four recordings?

If a dumb baseline matches CK's 0.99, then either the task is too easy to be
informative or CK has degenerated into a proxy for that dumb quantity. Either
way you would not have learned that CK works -- only that these particular
recordings are separable.
"""

import subprocess
import numpy as np
from scipy.ndimage import zoom


def build_matrix(specs, dist_fn):
    """
    Generic pairwise matrix from any symmetric distance function.
    Use this with dumb_energy, dumb_euclidean, or a bound compute_ck1_distance
    so all three go through identical plumbing and stay comparable.
    """
    n = len(specs)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(specs[i], specs[j])
            m[i, j] = m[j, i] = d
    return m


def dumb_energy(a, b):
    """
    Absolute difference in mean dB level. Carries zero pattern information --
    it cannot distinguish a caw from a chirp of equal loudness.

    If this scores well, your recordings are separable by loudness alone and
    cannot demonstrate anything about spectral matching.
    """
    return float(abs(a.mean() - b.mean()))


def dumb_euclidean(a, b, size=(32, 32)):
    """
    Euclidean distance between coarsely downsampled, per-image normalized
    log-spectrograms. Uses pattern, but with no compression, no motion
    compensation, and no shift tolerance whatsoever.

    This is the real competitor. CK's whole claim is that MPEG's motion
    search buys robustness that naive pixel differencing lacks. If plain
    Euclidean matches CK here, that claim is not being demonstrated.
    """
    def prep(s):
        z = zoom(s, (size[0] / s.shape[0], size[1] / s.shape[1]), order=1)
        rng = z.max() - z.min()
        return (z - z.min()) / rng if rng > 0 else np.zeros_like(z)
    x, y = prep(a), prep(b)
    return float(np.linalg.norm(x - y))


def measure_overhead(target_size=(80, 128), quality=25):
    """
    Bytes an MPEG-1 stream costs before encoding any image content.

    Encodes a pair of uniform mid-grey frames: no detail in the I-frame, a
    perfect match for the P-frame. What comes back is essentially sequence +
    GOP + picture + slice headers.

    This constant sits in BOTH the numerator and denominator of CK, so it
    does not cancel -- it biases every distance toward zero and shrinks your
    usable range. At ~370-byte frames it is not negligible.
    """
    w, h = target_size
    flat = np.full((h, w), 128, dtype=np.uint8).tobytes()
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
           '-s', f'{w}x{h}', '-r', '25', '-i', 'pipe:0',
           '-vcodec', 'mpeg1video', '-bf', '0', '-g', '10',
           '-q:v', str(quality), '-f', 'mpeg1video', 'pipe:1']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = p.communicate(input=flat + flat)
    return len(out)


def ck_overhead_corrected(spec_x, spec_y, ck_raw_fn, overhead):
    """
    Recompute CK with the fixed stream overhead removed from all four terms.

    Requires ck_raw_fn to return the four byte counts rather than the final
    distance -- see the note in the module docstring of your ck helpers about
    exposing them. Provided here as a reference implementation:

        num = (c_xy - H) + (c_yx - H)
        den = (c_xx - H) + (c_yy - H)
        ck  = num / den - 1
    """
    c_xy, c_yx, c_xx, c_yy = ck_raw_fn(spec_x, spec_y)
    H = overhead
    num = (c_xy - H) + (c_yx - H)
    den = (c_xx - H) + (c_yy - H)
    return max(0.0, num / den - 1.0) if den > 0 else 0.0
