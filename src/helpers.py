import os
import glob
import freesound
import pandas as pd
import essentia
import essentia.standard as es
import matplotlib.pyplot as plt
import numpy as np

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


def sounds_to_dataframe(sounds, audio_dir):
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
            "local_path": os.path.join(audio_dir, filename)
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

def detect_onsets(audio_path, sample_rate=44100):
    audio = es.MonoLoader(
        filename=audio_path,
        sampleRate=sample_rate
    )()
    
    # The OnsetDetection algorithm provides various ODFs.
    od_complex = es.OnsetDetection(method='complex')
    
    # We need the auxilary algorithms to compute magnitude and phase.
    w = es.Windowing(type='hann')
    fft = es.FFT() # Outputs a complex FFT vector.
    c2p = es.CartesianToPolar() # Converts it into a pair of magnitude and phase vectors.
    
    # Compute both ODF frame by frame. Store results to a Pool.
    pool = essentia.Pool()
    for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512):
        magnitude, phase = c2p(fft(w(frame)))
        pool.add('odf.complex', od_complex(magnitude, phase))
    
    # 2. Detect onset locations.
    onsets = es.Onsets()
    onset_times = onsets(essentia.array([pool['odf.complex']]), [1])

    return audio, onset_times

def plot_waveform_with_onsets(
    audio,
    onset_times,
    sample_rate=44100,
    start_time=None,
    end_time=None,
):
    # Convert times to sample indices
    start_sample = int(start_time * sample_rate) if start_time is not None else 0
    end_sample = int(end_time * sample_rate) if end_time is not None else len(audio)

    audio_seg = audio[start_sample:end_sample]
    time_axis = np.arange(len(audio_seg)) / sample_rate

    # Convert onset times to relative times within the segment
    rel_onsets = []
    for onset in onset_times:
        if (start_time is None or onset >= start_time) and \
           (end_time is None or onset <= end_time):
            rel_onsets.append(onset - (start_time or 0))

    fig, (ax_wave, ax_spec) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}  # waveform taller than spectrogram
    )

    # ---- Waveform ----
    ax_wave.plot(time_axis, audio_seg, linewidth=0.8)
    for onset in rel_onsets:
        ax_wave.axvline(onset, color="red", linestyle="--", alpha=0.7)

    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title("Waveform and Spectrogram with Detected Onsets")

    # ---- Spectrogram ----
    Pxx, freqs, bins, im = ax_spec.specgram(
        audio_seg,
        NFFT=1024,
        Fs=sample_rate,
        noverlap=512,
        scale="dB",
        cmap="magma",
    )

    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

