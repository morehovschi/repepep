import os
import glob
import freesound
import pandas as pd
import essentia
import essentia.standard as es
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import itertools
import matplotlib.transforms as transforms

from tqdm.auto import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from matplotlib.lines import Line2D
from IPython.display import Audio, display

_ENVELOPE = es.Envelope()
_LOG_ATTACK = es.LogAttackTime()
_STRONG_DECAY = es.StrongDecay()
_SPECTRUM = es.Spectrum()
_MFCC = es.MFCC(inputSize=513)
_WINDOWING = es.Windowing(type='hann')

def query_freesound(query, filter, client, fs_store_metadata_fields, num_results=10):
    """Queries freesound with the given query and filter values.
    If no filter is given, a default filter is added to only get sounds shorter than 10 seconds.
    """
    if filter is None:
        filter = 'duration:[0 TO 10]'

    pager = client.search(
        query=query,
        filter=filter,
        fields=','.join(fs_store_metadata_fields),
        group_by_pack=1,
        page_size=num_results
    )

    return [sound for sound in pager]


def retrieve_sound_preview(client, sound, directory):
    """Download the high-quality OGG sound preview of a given Freesound sound object to the given directory.
    """
    os.makedirs(directory, exist_ok=True)

    filename = os.path.basename(sound.previews.preview_hq_ogg)
    path = os.path.join(directory, filename)

    if os.path.exists(path):
        print(f"File {path} already downloaded.")
        return path

    return freesound.FSRequest.retrieve(
        sound.previews.preview_hq_ogg,
        client,
        path,
    )


def sounds_to_dataframe(sounds, save_dir):
    rows = []

    for sound in sounds:
        filename = os.path.basename(sound.previews.preview_hq_ogg)

        rows.append({
            "sound_id": sound.id,
            "name": sound.name,
            "username": sound.username,
            "license": sound.license,
            "tags": ",".join(sound.tags),
            "preview_url": sound.previews.preview_hq_ogg,
            "local_path": os.path.join(save_dir, filename)
        })

    return pd.DataFrame(rows)


def delete_dataset_with_gc(meta_csv_path, meta_dir, audio_dir, dry_run=True):
    """
    Delete a dataset metadata file and garbage-collect unreferenced audio files.

    Parameters
    ----------
    meta_csv_path : str
        Path to the metadata CSV to delete.
    meta_dir : str
        Directory containing all metadata CSV files.
    audio_dir : str
        Directory containing all audio files.
    dry_run : bool
        If True, do not delete anything, only report.
    """

    meta_csv_path = os.path.abspath(meta_csv_path)
    meta_dir = os.path.abspath(meta_dir)
    audio_dir = os.path.abspath(audio_dir)

    # 1. Delete metadata file
    if os.path.exists(meta_csv_path):
        json_path = meta_csv_path[:meta_csv_path.rfind(".")] + ".json"

        if dry_run:
            print(f"[DRY RUN] Would delete metadata: {meta_csv_path}")
            if os.path.exists(json_path):
                print(f"[DRY RUN] Would delete query file: {json_path}")
        else:
            os.remove(meta_csv_path)
            print(f"Deleted metadata file: {meta_csv_path}")

            if os.path.exists(json_path):
                os.remove(json_path)
            print(f"Deleted query file: {json_path}")
    else:
        print(f"Metadata file not found: {meta_csv_path}")

    # 2. Collect referenced audio files
    referenced_audio = set()

    for meta_path in glob.glob(os.path.join(meta_dir, "*.csv")):
        # if dry run, the target file wasn't actually deleted, so we need to
        # ensure we skip reading it
        if dry_run and meta_path == meta_csv_path:
            continue

        df = pd.read_csv(meta_path)
        for path in df["local_path"]:
            referenced_audio.add(os.path.abspath(path))

    # 3. Garbage collect audio
    deleted = 0
    kept = 0

    if dry_run:
        print()

    for root, _, files in os.walk(audio_dir):
        for fname in files:
            audio_path = os.path.abspath(os.path.join(root, fname))

            if audio_path not in referenced_audio:
                if dry_run:
                    print(f"[DRY RUN] Would delete audio: {audio_path}")
                else:
                    os.remove(audio_path)
                    print(f"Deleted audio: {audio_path}")
                deleted += 1
            else:
                kept += 1

    # 4. Clean up empty subdirectories left behind
    for root, dirs, _ in os.walk(audio_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                if dry_run:
                    print(f"[DRY RUN] Would delete empty directory: {dir_path}")
                else:
                    os.rmdir(dir_path)
                    print(f"Deleted empty directory: {dir_path}")

    print(f"\nGC summary:")
    print(f"  Kept audio files: {kept}")
    print(f"  Deleted audio files: {deleted}")


def ensure_datasets_audio(meta_dir,client,verbose=True):
    """
    Ensure that all audio files referenced by metadata CSVs exist.
    Missing files are re-downloaded.
    """

    meta_files = glob.glob(os.path.join(meta_dir, "*.csv"))

    if verbose:
        print(f"Found {len(meta_files)} metadata files")

    total_missing = 0

    for meta_path in meta_files:
        df = pd.read_csv(meta_path)

        if verbose:
            print(f"\nChecking {os.path.basename(meta_path)}")

        for _, row in df.iterrows():
            local_path = row["local_path"]
            preview_url = row["preview_url"]

            if os.path.exists(local_path):
                continue

            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            if verbose:
                print(f"  Re-downloading missing file: {local_path}")

            freesound.FSRequest.retrieve(
                preview_url,
                client,
                local_path
            )

            total_missing += 1

    if verbose:
        print(f"\nDone. Re-downloaded {total_missing} missing files.")

def load_and_detect_onsets(audio_path, high_sr=44100, low_sr=16000, min_ioi=0.083):
    """
    Loads audio at two sample rates: detects precise onsets at high_sr,
    but returns the audio array downsampled to low_sr for fast feature extraction.
    """
    # 1. Load high-res audio for sharp onset detection
    audio_high = es.MonoLoader(filename=audio_path, sampleRate=high_sr)()
    raw_onset_times, _ = es.OnsetRate()(audio_high)

    # 2. Load low-res audio for fast feature extraction downstream
    audio_low = es.MonoLoader(filename=audio_path, sampleRate=low_sr)()

    # 3. Apply unified structural filtering using low_sr boundaries
    filtered_onsets = []
    last_onset_time = -1.0

    for onset in raw_onset_times:
        if (onset - last_onset_time) < min_ioi:
            continue

        # Ensure the segment won't overshoot the bounds of the LOW-RES audio buffer
        # during the 1024-frame MFCC window computation
        start_idx_low = int(onset * low_sr)
        if (len(audio_low) - start_idx_low) < 1024:
            continue

        filtered_onsets.append(onset)
        last_onset_time = onset

    return audio_low, np.array(filtered_onsets)

def plot_waveform_with_onsets(
    audio,
    onset_times,
    sample_rate=16_000,
    start_time=None,
    end_time=None,
    max_annotations=20,
    show_spectrogram=True
):
    # Convert times to sample indices
    start_sample = int(start_time * sample_rate) if start_time is not None else 0
    end_sample = int(end_time * sample_rate) if end_time is not None else len(audio)

    audio_seg = audio[start_sample:end_sample]
    time_axis = np.arange(len(audio_seg)) / sample_rate

    rel_onsets = []
    for orig_idx, onset in enumerate(onset_times):
        if (start_time is None or onset >= start_time) and \
           (end_time is None or onset <= end_time):
            rel_onsets.append({
                'rel_time': onset - (start_time or 0),
                'orig_idx': orig_idx
            })

    # Setup layout conditionally based on whether we need the spectrogram
    if show_spectrogram:
        fig, (ax_wave, ax_spec) = plt.subplots(
            2, 1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )
        offset_spec = transforms.offset_copy(ax_spec.transData, fig=fig, x=5, y=0, units='points')
    else:
        # Create a single, much shorter plot space to prevent scrolling fatigue
        fig, ax_wave = plt.subplots(1, 1, figsize=(14, 3))

    offset_wave = transforms.offset_copy(ax_wave.transData, fig=fig, x=5, y=0, units='points')

    # ---- Waveform Subplot ----
    ax_wave.plot(time_axis, audio_seg, linewidth=0.8, color='steelblue')
    y_text_pos_wave = np.max(np.abs(audio_seg)) * 1.05 if len(audio_seg) > 0 else 0.9

    should_annotate = len(rel_onsets) <= max_annotations

    for item in rel_onsets:
        t = item['rel_time']
        idx = item['orig_idx']

        ax_wave.axvline(t, color="red", linestyle="--", alpha=0.7, linewidth=1.2)

        if should_annotate:
            ax_wave.text(
                x=t,
                y=y_text_pos_wave,
                s=str(idx),
                color="red",
                fontsize=9,
                fontweight='bold',
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=offset_wave
            )

    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title("Waveform with Indexed Onsets")
    ax_wave.set_ylim(-y_text_pos_wave * 1.1, y_text_pos_wave * 1.25)

    # If the spectrogram is hidden, the waveform needs its own X-axis label
    if not show_spectrogram:
        ax_wave.set_xlabel("Time (s)")

    # ---- Spectrogram Subplot (Optional) ----
    if show_spectrogram:
        ax_wave.set_title("Waveform and Spectrogram with Indexed Onsets")
        Pxx, freqs, bins, im = ax_spec.specgram(
            audio_seg, NFFT=1024, Fs=sample_rate, noverlap=512, scale="dB", cmap="magma"
        )

        for item in rel_onsets:
            t = item['rel_time']
            idx = item['orig_idx']

            ax_spec.axvline(t, color="white", linestyle=":", alpha=0.6, linewidth=1.2)

            if should_annotate:
                ax_spec.text(
                    x=t,
                    y=sample_rate / 2 * 0.9,
                    s=str(idx),
                    color="white",
                    fontsize=9,
                    fontweight='bold',
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=offset_spec
                )

        ax_spec.set_ylabel("Frequency (Hz)")
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylim(0, sample_rate / 2)

    plt.tight_layout()
    plt.show()

def analyze_repetitiveness(audio, onset_times, filename="Audio Array", sr=16000, verbose=True,
                           crest_factor_threshold=5.0, window_ms=200, lat_threshold=0.15,
                           min_decay=0.0, eps=0.15, nn_extractor=None, feature_cache=None):
    """
    Analyzes pre-loaded low-res audio and pre-detected high-res onsets.
    Uses 'sr' (default 16000) to map time locations to low-res sample indices.
    Dynamically adjusts window size based on Inter-Onset Intervals (IOI).
    """
    def print_func(string):
        if verbose:
            print(string)

    print_func(f"--- Analyzing: {filename} ---")
    print_func(f"Total structured onsets to process: {len(onset_times)}")

    if len(onset_times) == 0:
        return None

    if feature_cache is None:
        feature_cache = {}
    if filename not in feature_cache:
        feature_cache[filename] = {}

    features = []

    for orig_idx, onset in enumerate(onset_times):
        # 1. Calculate adaptive window size based on the next onset
        if orig_idx < len(onset_times) - 1:
            next_onset = onset_times[orig_idx + 1]
            next_ioi_ms = (next_onset - onset) * 1000.0

            # Dynamic scaling (ioi * 1.5) capped at a maximum of 500ms
            curr_window_ms = min(next_ioi_ms * 1.5, 500.0)
        else:
            # Fallback for the last onset since there is no "next" event
            curr_window_ms = 500.0

        # 2. Extract the dynamic sample window
        window_samples = int((curr_window_ms / 1000.0) * sr)
        start_idx = int(onset * sr)
        end_idx = min(start_idx + window_samples, len(audio))
        segment = audio[start_idx:end_idx]

        # Signal calculations
        peak_amp = np.max(np.abs(segment)) if len(segment) > 0 else 0
        rms_amp = np.sqrt(np.mean(segment**2)) + 1e-9 if len(segment) > 0 else 1e-9
        crest_factor = peak_amp / rms_amp

        if crest_factor < crest_factor_threshold:
            continue

        env = _ENVELOPE(segment)
        try:
            lat = _LOG_ATTACK(env)[0]
        except:
            lat = 1.0

        try:
            decay = _STRONG_DECAY(segment)
        except:
            decay = 0.0

        if orig_idx not in feature_cache[filename]:
            if nn_extractor is None:
                segment_mfccs = []
                for frame in es.FrameGenerator(segment, frameSize=1024, hopSize=512, startFromZero=True):
                    _, mfcc_coeffs = _MFCC(_WINDOWING(_SPECTRUM(frame)))
                    segment_mfccs.append(mfcc_coeffs[1:]) # Drop 0th coefficient

                if not segment_mfccs:
                    continue
                # Classical representation: Average MFCC frame profile
                feature_vector = np.mean(segment_mfccs, axis=0)
            else:
                # Pad segment to exactly 1 second (16000 samples) to satisfy VGGish's receptive field
                required_samples = sr  # 16000 samples for a 16kHz model
                if len(segment) < required_samples:
                    padded_segment = np.pad(segment, (0, required_samples - len(segment)), mode='constant')
                else:
                    padded_segment = segment[:required_samples]

                # Neural representation: Extract the 128-D vector natively via Essentia
                # (Ensuring it returns a flat 1D array)
                feature_vector = nn_extractor(padded_segment).flatten()
            feature_cache[filename][orig_idx] = feature_vector
        else:
            feature_vector = feature_cache[filename][orig_idx]

        features.append({
            'orig_idx': orig_idx,
            'time': onset,
            'lat': lat,
            'decay': decay,
            'crest': crest_factor,
            'feature': feature_vector,
            'window_ms_used': curr_window_ms  # Tracked for debugging/analysis
        })

    if not features:
        print_func("No onsets survived the Crest Factor pre-filtering.")
        return None

    df = pd.DataFrame(features)
    percussive_df = df[(df['lat'] < lat_threshold) & (df['decay'] > min_decay)].copy()
    print_func(f"Onsets passing LAT (< {lat_threshold}s) & Decay (> {min_decay}) filters: {len(percussive_df)}")

    if len(percussive_df) < 2:
        print_func("Not enough percussive/decaying events to cluster.\n")
        return None

    X_mfcc = np.vstack(percussive_df['feature'].values)
    dist_matrix = pairwise_distances(X_mfcc, metric='cosine')

    clusterer = DBSCAN(eps=eps, min_samples=3, metric='precomputed')
    percussive_df['cluster'] = clusterer.fit_predict(dist_matrix)

    clusters = percussive_df[percussive_df['cluster'] != -1]

    if not clusters.empty:
        largest_cluster = clusters['cluster'].value_counts().idxmax()
        rep_count = len(clusters[clusters['cluster'] == largest_cluster])
        print_func(f"--> Found repeating event! Repetitions count: {rep_count}\n")
    else:
        print_func("--> No repeating clusters found.\n")

    return percussive_df

def inspect_sound_with_repetition(meta_dir, sound_id,  metadata_df=None,
                                  show_plot=True, verbose=True,
                                  crest_factor_threshold=5.0,
                                  lat_threshold=0.15, window_ms=200,
                                  min_decay=0.0, eps=0.15, audio=None,
                                  onset_times=None, nn_extractor=None,
                                  feature_cache=None):
    """
    Wrapper function updated to accept a pre-loaded metadata DataFrame for efficiency,
    and an option to disable plotting during batch evaluation.
    """
    def print_func(string):
        if verbose:
            print(string)

    match = None

    # 1. Efficient lookup: Use the pre-loaded DataFrame if provided
    if metadata_df is not None:
        match = metadata_df[metadata_df["sound_id"] == sound_id]
    else:
        # Fallback to the old behavior (globbing CSVs)
        for csv_path in glob.glob(os.path.join(meta_dir, "dataset_*.csv")):
            df = pd.read_csv(csv_path)
            temp_match = df[df["sound_id"] == sound_id]
            if not temp_match.empty:
                match = temp_match
                break


    if match is not None and not match.empty:
        row = match.iloc[0]
        local_path = row["local_path"]
        filename = os.path.basename(local_path)

        if audio is None or onset_times is None:
            audio, onset_times = load_and_detect_onsets(local_path)

        analysis_df = analyze_repetitiveness(
            audio=audio,
            onset_times=onset_times,
            filename=filename,
            crest_factor_threshold=crest_factor_threshold,
            lat_threshold=lat_threshold,
            window_ms=window_ms,
            min_decay=min_decay,
            eps=eps,
            verbose=verbose,
            sr=16_000,
            nn_extractor=nn_extractor,
            feature_cache=feature_cache,
        )

        rep_times = []
        rep_indices = []

        if analysis_df is not None:
            valid_clusters = analysis_df[analysis_df['cluster'] != -1]
            if not valid_clusters.empty:
                largest_cluster = valid_clusters['cluster'].value_counts().idxmax()

                best_cluster = valid_clusters[valid_clusters['cluster'] == largest_cluster]
                rep_times = best_cluster['time'].values
                rep_indices = best_cluster['orig_idx'].values.tolist()

        # 2. Conditionally render the visual and audio outputs
        if show_plot:
            print_func(f"--- Sound ID: {sound_id} ---")
            print_func(f"File: {filename}")
            print_func(f"Method: Freesound Native (es.OnsetRate) | Detected: {len(onset_times)} | In Pattern: {len(rep_times)}")
            print_func(f"Detected Repetitive Indices: {rep_indices}")

            sample_rate = 16_000
            time_axis = np.arange(len(audio)) / sample_rate

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(time_axis, audio, color='gray', alpha=0.3, linewidth=0.8)

            for t in onset_times:
                ax.axvline(t, color="red", linestyle="--", linewidth=1, alpha=0.4)

            for rt in rep_times:
                ax.axvline(rt, color="green", linestyle="-", linewidth=2.5, alpha=0.9)

            title = f"\n{row['name']} (ID: {sound_id})"
            ax.set_title(title)
            ax.set_ylabel("Amplitude")
            ax.set_xlabel("Time (s)")

            custom_lines = [Line2D([0], [0], color='red', lw=1, linestyle='--', alpha=0.4),
                            Line2D([0], [0], color='green', lw=2.5)]
            ax.legend(custom_lines, ['Raw Onsets (OnsetRate)', 'Detected Repetitive Pattern'])

            plt.tight_layout()
            plt.show()

            display(Audio(local_path))

        return rep_indices

    if show_plot:
        print_func(f"Sound ID {sound_id} not found.")

    return []

def evaluate_pipeline(ground_truth, meta_dir, df, output_csv="eval_results.csv", show_plot=True,
                      verbose=True, crest_factor_threshold=5.0, lat_threshold=0.15, window_ms=125,
                      min_decay=0.0, eps=0.15, audio_cache=None, nn_extractor=None,
                      feature_cache=None):
    """
    ground_truth: dict mapping sound_id -> list of true indices, e.g. {655124: [0, 1, 2]}
    df: Pre-assembled master DataFrame containing 'sound_id' and 'search_query'
    """
    def print_func(string):
        if verbose:
            print(string)

    audio_cache = audio_cache or {}

    # Store per-sound results here
    results_log = []

    for sound_id, onset_data in tqdm(ground_truth.items(), desc="Evaluating repetition detection", leave=False):
        true_indices = onset_data["repetitive_onset_indices"]

        sound_id_int = int(sound_id)
        if sound_id_int in audio_cache:
            audio, onset_times = audio_cache[sound_id_int]
        else:
            audio, onset_times = None, None

        pred_indices = inspect_sound_with_repetition(
            meta_dir, int(sound_id), metadata_df=df, show_plot=show_plot,
            crest_factor_threshold=crest_factor_threshold, lat_threshold=lat_threshold,
            window_ms=window_ms, min_decay=min_decay, eps=eps, verbose=verbose,
            audio=audio, onset_times=onset_times, nn_extractor=nn_extractor,
            feature_cache=feature_cache,
        )

        true_set = set(true_indices)
        pred_set = set(pred_indices)

        hits = len(true_set.intersection(pred_set))

        file_has_reps = len(true_set) > 0
        file_pred_reps = len(pred_set) > 0

        # --- PER-SOUND METRICS ---
        if not file_has_reps:
            p, r, f = None, None, None
        else:
            p = hits / len(pred_set) if len(pred_set) > 0 else 0.0
            r = hits / len(true_set)
            f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        # Direct row lookup from your pre-cleaned DataFrame
        query_used = df.loc[df['sound_id'] == int(sound_id), 'search_query'].iloc[0]

        # Append to log with structural tier metrics
        results_log.append({
            'sound_id': sound_id,
            'query': query_used,
            'true_onsets': ",".join(map(str, sorted(list(true_set)))),
            'predicted_onsets': ",".join(map(str, sorted(list(pred_set)))),
            'file_has_reps': int(file_has_reps),
            'file_pred_reps': int(file_pred_reps),
            'onset_hits': hits,
            'onset_true_count': len(true_set),
            'onset_pred_count': len(pred_set),
            'precision': round(p, 3) if p is not None else None,
            'recall': round(r, 3) if r is not None else None,
            'f1': round(f, 3) if f is not None else None
        })

        print_func(f"ID {sound_id} | True Count: {len(true_set)} | Pred Count: {len(pred_set)} | Onset F1: {f if f is not None else 'NaN'}\n")

    # --- TIED GLOBAL METRICS ---
    # Tier 1: Gatekeeper Classification Counts
    rec_tp = sum(1 for row in results_log if row['file_has_reps'] and row['file_pred_reps'])
    rec_tn = sum(1 for row in results_log if not row['file_has_reps'] and not row['file_pred_reps'])
    rec_fp = sum(1 for row in results_log if not row['file_has_reps'] and row['file_pred_reps'])
    rec_fn = sum(1 for row in results_log if row['file_has_reps'] and not row['file_pred_reps'])

    g_acc = (rec_tp + rec_tn) / len(results_log) if results_log else 0.0
    g_spec = rec_tn / (rec_tn + rec_fp) if (rec_tn + rec_fp) > 0 else 0.0
    g_prec = rec_tp / (rec_tp + rec_fp) if (rec_tp + rec_fp) > 0 else 0.0
    g_rec = rec_tp / (rec_tp + rec_fn) if (rec_tp + rec_fn) > 0 else 0.0
    g_f1 = 2 * (g_prec * g_rec) / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0

    # Tier 2: Tracker Localizer Counts (Evaluated strictly on true positive files)
    t_true = sum(row['onset_true_count'] for row in results_log if row['file_has_reps'])
    t_pred = sum(row['onset_pred_count'] for row in results_log if row['file_has_reps'])
    t_hits = sum(row['onset_hits'] for row in results_log if row['file_has_reps'])

    t_precision = t_hits / t_pred if t_pred > 0 else 0.0
    t_recall = t_hits / t_true if t_true > 0 else 0.0
    t_f1 = 2 * (t_precision * t_recall) / (t_precision + t_recall) if (t_precision + t_recall) > 0 else 0.0

    print_func("\n=== FINAL TWO-TIER EVALUATION ===")
    print_func(f"Layer 1 (Gatekeeper) Accuracy:    {g_acc:.3f}")
    print_func(f"Layer 1 (Gatekeeper) Specificity: {g_spec:.3f} (Correct Noise Rejections)")
    print_func(f"Layer 1 (Gatekeeper) F1 Score:    {g_f1:.3f}")
    print_func(f"Layer 2 (Tracker) Global Precision: {t_precision:.3f}")
    print_func(f"Layer 2 (Tracker) Global Recall:    {t_recall:.3f}")
    print_func(f"Layer 2 (Tracker) Global F1 Score:  {t_f1:.3f}")

    # Save the CSV
    results_df = pd.DataFrame(results_log)
    results_df.to_csv(output_csv, index=False)
    print_func(f"\nSaved detailed two-tier evaluation to '{output_csv}'")

def run_pipeline_grid_search(ground_truth, meta_dir, df, output_dir, param_grid, nn_extractor=None):
    """
    Executes a grid search across multiple parameter combinations for the audio pipeline.
    Saves individual run details and a master summary sheet tracking both evaluation tiers.
    """
    os.makedirs(output_dir, exist_ok=True)

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # cache audio and onset data, to be used for all runs
    audio_cache = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting audio & onsets cache"):
        audio_cache[row["sound_id"]] = load_and_detect_onsets(row["local_path"])

    feature_cache = {}

    print(f"Starting Grid Search. Total configurations to evaluate: {len(combinations)}")
    summary_log = []

    for i, params in enumerate(combinations):
        run_id = f"run_{i+1:03d}"
        print(f"[{run_id}/{len(combinations)}] Testing: {params}")

        detailed_csv_path = os.path.join(output_dir, f"{run_id}_details.csv")

        evaluate_pipeline(
            ground_truth=ground_truth,
            meta_dir=meta_dir,
            df=df,
            output_csv=detailed_csv_path,
            show_plot=False,
            verbose=False,
            audio_cache=audio_cache,
            nn_extractor=nn_extractor,
            feature_cache=feature_cache,
            **params
        )

        run_df = pd.read_csv(detailed_csv_path)

        # Tier 1 Math: Recording level confusion matrix
        rec_tp = len(run_df[(run_df['file_has_reps'] == 1) & (run_df['file_pred_reps'] == 1)])
        rec_tn = len(run_df[(run_df['file_has_reps'] == 0) & (run_df['file_pred_reps'] == 0)])
        rec_fp = len(run_df[(run_df['file_has_reps'] == 0) & (run_df['file_pred_reps'] == 1)])
        rec_fn = len(run_df[(run_df['file_has_reps'] == 1) & (run_df['file_pred_reps'] == 0)])

        g_acc = (rec_tp + rec_tn) / len(run_df) if len(run_df) > 0 else 0.0
        g_spec = rec_tn / (rec_tn + rec_fp) if (rec_tn + rec_fp) > 0 else 0.0
        g_prec = rec_tp / (rec_tp + rec_fp) if (rec_tp + rec_fp) > 0 else 0.0
        g_rec = rec_tp / (rec_tp + rec_fn) if (rec_tp + rec_fn) > 0 else 0.0
        g_f1 = 2 * (g_prec * g_rec) / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0

        # Tier 2 Math: Onset tracking matrix (Positive files only)
        pos_df = run_df[run_df['file_has_reps'] == 1]
        t_true = pos_df['onset_true_count'].sum()
        t_pred = pos_df['onset_pred_count'].sum()
        t_hits = pos_df['onset_hits'].sum()

        t_precision = t_hits / t_pred if t_pred > 0 else 0.0
        t_recall = t_hits / t_true if t_true > 0 else 0.0
        t_f1 = 2 * (t_precision * t_recall) / (t_precision + t_recall) if (t_precision + t_recall) > 0 else 0.0

        summary_row = {
            'run_id': run_id,
            **params,
            'gatekeeper_accuracy': round(g_acc, 3),
            'gatekeeper_specificity': round(g_spec, 3),
            'gatekeeper_precision': round(g_prec, 3),
            'gatekeeper_recall': round(g_rec, 3),
            'gatekeeper_f1': round(g_f1, 3),
            'tracker_precision': round(t_precision, 3),
            'tracker_recall': round(t_recall, 3),
            'tracker_f1': round(t_f1, 3)
        }
        summary_log.append(summary_row)

    summary_df = pd.DataFrame(summary_log)
    summary_csv_path = os.path.join(output_dir, "grid_search_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"\n==== GRID SEARCH COMPLETE ====")
    print(f"Master summary file generated at: {summary_csv_path}")

    return summary_df
