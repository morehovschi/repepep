import subprocess
import os
import glob
import random
import json
import itertools
import essentia
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import essentia.standard as es
import numpy as np

from PIL import Image
from datetime import datetime
from scipy.interpolate import interp1d

from helpers import load_and_detect_onsets

_OVERHEAD = {}

def load_audio_as_spectrogram_essentia(file_path):
    """
    Loads an audio file and computes its log-magnitude spectrogram using Essentia.
    """
    # 1. Load the audio file 
    # MonoLoader automatically downmixes multi-channel files to mono 
    # and resamples to a uniform 44100Hz by default.
    audio = es.MonoLoader(filename=file_path)()
    
    # 2. Instantiate the required spectral algorithms
    windowing = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    
    # 3. Slice the audio into frames and compute the linear magnitude spectrum
    # frameSize=2048 and hopSize=512 perfectly match standard STFT dimensions
    spec_list = []
    for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
        windowed_frame = windowing(frame)
        frame_spectrum = spectrum(windowed_frame)
        spec_list.append(frame_spectrum)
        
    # 4. Stack frames into a 2D NumPy array (Frequency Bins x Time Frames)
    spectrogram = np.array(spec_list).T
    
    # 5. Convert linear amplitude to Decibel (Log) scale
    # We use np.maximum to clip values at 1e-5 to avoid log(0) baseline crashes
    spectrogram_db = 20 * np.log10(np.maximum(spectrogram, 1e-5))
    
    return spectrogram_db

def compute_ck1_distance(spec_x, spec_y, target_size=(80,128), quality=25,
                         subtract_overhead=False, norm_range=None):
    """
    Computes the Campana-Keogh (CK-1) distance between two 2D spectrogram arrays.
    
    Parameters:
    -----------
    spec_x : np.ndarray
        2D array of the first spectrogram.
    spec_y : np.ndarray
        2D array of the second spectrogram.
    target_size : tuple (width, height)
        Dimensions to resize spectrograms. Must be multiples of 16 for MPEG-1 (e.g., 256x256).
    quality : int
        MPEG fixed quality scale (1-31). Lower means higher quality/finer detail resolution.
        Default 5 is a robust sweet spot for texture discovery.
    """
    
    # 1. Helper function to normalize and resize spectrograms to grayscale frames
    def preprocess_spectrogram(spec):
        spec = spec[:target_size[1]]
        s_min, s_max = norm_range if norm_range else (spec.min(), spec.max())
        spec = np.clip(spec, s_min, s_max)
        if s_max > s_min:
            spec_norm = 255.0 * (spec - s_min) / (s_max - s_min)
        else:
            spec_norm = np.zeros_like(spec)

        img = Image.fromarray(spec_norm.astype(np.uint8))
        img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
        return np.array(img_resized)

    # 2. Helper function to pass 2 frames to FFmpeg and get the compressed size in bytes
    def get_mpeg1_compressed_size(frame_1, frame_2):
        f1_bytes = frame_1.tobytes()
        f2_bytes = frame_2.tobytes()
        
        # FFmpeg command optimized for exact conditional algorithmic complexity estimation
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-s', f'{width}x{height}',
            '-r', '25',                    # Standard framerate input
            '-i', 'pipe:0',                # Read from stdin pipe
            '-vcodec', 'mpeg1video',       # Force legacy MPEG-1 as per the original spec
            '-bf', '0',                    # Disable B-frames (forces frame 2 to be a P-frame)
            '-g', '10',                    # Prevent forcing frame 2 into a new GOP/I-frame
            '-sc_threshold', '1000000000',
            '-me_method', 'esa',
            '-me_range', '16',
            '-q:v', str(quality),          # CRITICAL: Use constant quality scale instead of fixed bitrate
            '-f', 'mpeg1video',            # Raw video stream container (minimal overhead)
            'pipe:1'                       # Output to stdout pipe
        ]
        
        # Open asynchronous pipe to FFmpeg
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Stream frames into stdin and grab output bitstream from stdout
        stdout, _ = process.communicate(input=f1_bytes + f2_bytes)
        if process.returncode !=0 or len(stdout) == 0:
            raise RuntimeError(f"ffmpeg failed (rc={process.returncode})")
        return len(stdout)

    def stream_overhead(target_size, quality):
        key = (target_size, quality)
        if key not in _OVERHEAD:
            w, h = target_size
            flat = np.full((h, w), 128, dtype=np.uint8)
            _OVERHEAD[key] = get_mpeg1_compressed_size(flat, flat)
        return _OVERHEAD[key]

    # Preprocess both inputs
    x = preprocess_spectrogram(spec_x)
    y = preprocess_spectrogram(spec_y)
    width, height = target_size


    # 3. Compute the four compression elements required by the formula
    c_x_given_y = get_mpeg1_compressed_size(y, x)  # y is I-frame, x is P-frame
    c_y_given_x = get_mpeg1_compressed_size(x, y)  # x is I-frame, y is P-frame
    c_x_given_x = get_mpeg1_compressed_size(x, x)  # Identity baseline x
    c_y_given_y = get_mpeg1_compressed_size(y, y)  # Identity baseline y

    # 4. Final CK-1 Metric calculation
    H = stream_overhead(target_size, quality) if subtract_overhead else 0
    numerator   = (c_x_given_y - H) + (c_y_given_x - H)
    denominator = (c_x_given_x - H) + (c_y_given_y - H)
    ck1_distance = (numerator / denominator) - 1.0

    return max(0.0, ck1_distance) # Clamp near zero minor float variations

def compute_spectrogram_from_chunk(audio_segment, hop_size=64, n_bands=128,
                                   log_freq=False, fmin=50, fmax=11000,
                                   sr=44100):
    windowing = es.Windowing(type='hann')
    spectrum  = es.Spectrum()

    spec_list = []
    for frame in es.FrameGenerator(audio_segment, frameSize=512, hopSize=hop_size):
        spec_list.append(spectrum(windowing(frame)))
    if not spec_list:
        return None

    spec_db = 20 * np.log10(np.maximum(np.array(spec_list).T, 1e-5))

    if log_freq:
        # Resample the linear frequency axis onto a log-spaced grid.
        # Equivalent to plotting the spectrogram with a log y-axis, and free
        # of the triangular-band width constraint that MelBands imposes.
        bin_freqs = np.linspace(0, sr / 2, spec_db.shape[0])
        targets = np.geomspace(fmin, fmax, n_bands)
        f = interp1d(bin_freqs, spec_db, axis=0, kind='linear',
                     bounds_error=False,
                     fill_value=(spec_db[0], spec_db[-1]))
        spec_db = f(targets)

    return spec_db

def analyze_recording_onsets_ck(file_path, window_ms=125, target_sr=44100, quality=25,
                                subtract_overhead=False, log_freq=False):
    """
    Loads high-res audio, uses the filtered high-res timestamps from the helper,
    extracts a fixed window after each onset, and computes the pairwise CK matrix.
    """
    # 1. Extract the filtered onset timestamps from your existing helper
    # We ignore the low-res audio array it returns for this specific high-res test
    _, onset_times = load_and_detect_onsets(file_path)
    
    # 2. Load the actual high-res audio buffer for high-fidelity spectrograms
    audio_high = es.MonoLoader(filename=file_path, sampleRate=target_sr)()
    
    print(f"Processing: {os.path.basename(file_path)}")
    print(f"Filtered Onsets to process: {len(onset_times)}")
    
    if len(onset_times) < 2:
        print("Not enough onsets to perform pairwise comparison.")
        return None, None
        
    # 3. Extract fixed-size spectrograms for each onset using 44.1kHz math
    window_samples = int((window_ms / 1000.0) * target_sr)
    onset_spectrograms = []
    valid_onset_times = []
    
    for onset in onset_times:
        start_idx = int(onset * target_sr)
        end_idx = start_idx + window_samples
        
        # Now this boundary check matches the audio_high buffer perfectly
        if end_idx > len(audio_high):
            continue
            
        segment = audio_high[start_idx:end_idx]
        spec = compute_spectrogram_from_chunk(segment, log_freq=log_freq,
                                              sr=target_sr)
        
        if spec is not None:
            onset_spectrograms.append(spec)
            valid_onset_times.append(onset)
            
    num_onsets = len(onset_spectrograms)
    print(f"Successfully generated {num_onsets} onset spectrograms.")
    
    # 4. Compute the pairwise CK distance matrix
    ck_matrix = np.zeros((num_onsets, num_onsets))
    
    for i in range(num_onsets):
        for j in range(i, num_onsets):
            dist = compute_ck1_distance(onset_spectrograms[i], onset_spectrograms[j],
                                        quality=quality, subtract_overhead=subtract_overhead)
            ck_matrix[i, j] = dist
            ck_matrix[j, i] = dist  # Symmetric fill
                
    return ck_matrix, valid_onset_times, onset_spectrograms

