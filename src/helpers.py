import os
import glob
import freesound
import pandas as pd
import essentia
import essentia.standard as es
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import matplotlib.transforms as transforms

from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from matplotlib.lines import Line2D
from IPython.display import Audio, display

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

def load_and_detect_onsets(audio_path, sr=44100):
    """
    Loads an audio file and detects its onsets using Essentia's combined OnsetRate algorithm.
    Ensures file I/O and onset detection happen exactly once per file.
    """
    # Load the audio at the strict 44100Hz requirement for OnsetRate
    audio = es.MonoLoader(filename=audio_path, sampleRate=sr)()

    # OnsetRate returns: (onset_times, onset_rate_per_sec)
    # We unpack and discard the rate calculation using '_'
    onset_times, _ = es.OnsetRate()(audio)

    return audio, onset_times

def plot_waveform_with_onsets(
    audio,
    onset_times,
    sample_rate=44100,
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

def analyze_repetitiveness(audio, onset_times, filename="Audio Array", sr=44100,
                           window_ms=200, lat_threshold=0.15, min_ioi=0.10, min_decay=0.0, eps=0.15):
    """
    Analyzes pre-loaded audio and pre-detected onsets for repetitive events.
    Now tracks and returns the original onset indices.
    """
    print(f"--- Analyzing: {filename} ---")
    print(f"Total raw onsets detected: {len(onset_times)}")

    if len(onset_times) == 0:
        return None

    window_samples = int((window_ms / 1000.0) * sr)

    envelope = es.Envelope()
    log_attack = es.LogAttackTime()
    strong_decay = es.StrongDecay()
    spectrum = es.Spectrum()
    mfcc = es.MFCC(inputSize=513)
    w = es.Windowing(type='hann')

    features = []
    last_onset_time = -1.0

    # --> Change 1: Use enumerate to capture the original index
    for orig_idx, onset in enumerate(onset_times):
        if (onset - last_onset_time) < min_ioi:
            continue

        start_idx = int(onset * sr)
        end_idx = min(start_idx + window_samples, len(audio))
        segment = audio[start_idx:end_idx]

        if len(segment) < 1024:
            continue

        peak_amp = np.max(np.abs(segment))
        rms_amp = np.sqrt(np.mean(segment**2)) + 1e-9
        crest_factor = peak_amp / rms_amp

        if crest_factor < 5.0:
            continue

        env = envelope(segment)
        try:
            lat = log_attack(env)[0]
        except:
            lat = 1.0

        try:
            decay = strong_decay(segment)
        except:
            decay = 0.0

        segment_mfccs = []
        for frame in es.FrameGenerator(segment, frameSize=1024, hopSize=512, startFromZero=True):
            _, mfcc_coeffs = mfcc(spectrum(w(frame)))
            segment_mfccs.append(mfcc_coeffs[1:])

        mean_mfcc = np.mean(segment_mfccs, axis=0)
        last_onset_time = onset

        # --> Change 2: Save orig_idx to the dictionary
        features.append({
            'orig_idx': orig_idx,
            'time': onset,
            'lat': lat,
            'decay': decay,
            'crest': crest_factor,
            'mfcc': mean_mfcc
        })

    if not features:
        print("No onsets survived the initial pre-filtering (IOI, segment length, or Crest Factor).")
        return None

    df = pd.DataFrame(features)

    percussive_df = df[(df['lat'] < lat_threshold) & (df['decay'] > min_decay)].copy()
    print(f"Onsets passing LAT (< {lat_threshold}s) & Decay (> {min_decay}) filters: {len(percussive_df)}")

    if len(percussive_df) < 2:
        print("Not enough percussive/decaying events to cluster.\n")
        return None

    X_mfcc = np.vstack(percussive_df['mfcc'].values)
    dist_matrix = pairwise_distances(X_mfcc, metric='cosine')

    clusterer = DBSCAN(eps=eps, min_samples=3, metric='precomputed')
    percussive_df['cluster'] = clusterer.fit_predict(dist_matrix)

    clusters = percussive_df[percussive_df['cluster'] != -1]

    if not clusters.empty:
        largest_cluster = clusters['cluster'].value_counts().idxmax()
        rep_count = len(clusters[clusters['cluster'] == largest_cluster])
        print(f"--> Found repeating event! Repetitions count: {rep_count}\n")
    else:
        print("--> No repeating clusters found.\n")

    return percussive_df

def inspect_sound_with_repetition(meta_dir, sound_id, window_ms=200):
    """
    Wrapper function updated to match the new default window_ms of 200.
    """
    for csv_path in glob.glob(os.path.join(meta_dir, "dataset_*.csv")):
        df = pd.read_csv(csv_path)
        match = df[df["sound_id"] == sound_id]

        if not match.empty:
            row = match.iloc[0]
            local_path = row["local_path"]
            filename = os.path.basename(local_path)

            audio, onset_times = load_and_detect_onsets(local_path)

            analysis_df = analyze_repetitiveness(
                audio=audio,
                onset_times=onset_times,
                filename=filename,
                window_ms=window_ms
            )

            rep_times = []
            rep_indices = []  # --> New array to hold the indices

            if analysis_df is not None:
                valid_clusters = analysis_df[analysis_df['cluster'] != -1]
                if not valid_clusters.empty:
                    largest_cluster = valid_clusters['cluster'].value_counts().idxmax()

                    # Get both times and indices for the winning cluster
                    best_cluster = valid_clusters[valid_clusters['cluster'] == largest_cluster]
                    rep_times = best_cluster['time'].values
                    rep_indices = best_cluster['orig_idx'].values.tolist()

            print(f"--- Sound ID: {sound_id} ---")
            print(f"File: {filename}")
            print(f"Method: Freesound Native (es.OnsetRate) | Detected: {len(onset_times)} | In Pattern: {len(rep_times)}")
            print(f"Detected Repetitive Indices: {rep_indices}")

            sample_rate = 44100
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

    print(f"Sound ID {sound_id} not found.")
