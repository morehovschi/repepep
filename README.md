# Detecting and Counting Repetition in Recordings of Sound Effects

Accompanying repository for the Master's Thesis.

## Dataset Overview

The proposed dataset is located at:
`data/complete_annotations.json`

The dataset consists of annotations of precise timestamps of repetitive event instances in Freesound recordings. It is formatted as a JSON object where each key represents a Freesound ID pointing to the following metadata and annotations:

* **title**: Recording title on Freesound
* **author**: Freesound username of the uploader
* **freesound_url**: Direct web link to the audio recording
* **license**: Creative Commons license associated with the recording
* **search_query**: Query term used to retrieve the recording during annotation
* **repetitive_onset_times**: List of timestamps (in seconds) corresponding to repetitive sound event instances
* **repetitive_onset_indices**: List of onset indices for repetitions, detected via Essentia's `OnsetRate` (default parameters, 44.1 kHz)
