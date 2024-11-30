import os
import numpy as np
from pyedflib import EdfReader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def parse_summary_file(summary_file_path):
    """Parses the summary file to extract seizure information for each file."""
    seizure_info = {}
    try:
        with open(summary_file_path, 'r') as file:
            current_file = None
            seizures = []
            for line in file:
                line = line.strip()
                if line.startswith('File Name:'):
                    if current_file and seizures:
                        seizure_info[current_file] = seizures
                    current_file = line.split(':')[1].strip()
                    seizures = []
                elif 'Seizure Start Time' in line:
                    start_time = float(line.split(':')[1].strip().split()[0])
                elif 'Seizure End Time' in line:
                    end_time = float(line.split(':')[1].strip().split()[0])
                    seizures.append((start_time, end_time))
            if current_file and seizures:
                seizure_info[current_file] = seizures
    except Exception as e:
        print(f"[ERROR] Error parsing summary file: {e}")
    return seizure_info

def extract_signal(edf_file, channel_name):
    """Extracts the signal for a specific channel from an EDF file."""
    try:
        with EdfReader(edf_file) as edf_reader:
            channel_labels = edf_reader.getSignalLabels()
            if channel_name not in channel_labels:
                print(f"[INFO] Channel '{channel_name}' not found in {edf_file}. Skipping...")
                return None
            channel_index = channel_labels.index(channel_name)
            return edf_reader.readSignal(channel_index)
    except Exception as e:
        print(f"[ERROR] Error extracting signal from {edf_file}: {e}")
        return None

def normalize_signal(signal):
    """Normalizes the signal."""
    return (signal - np.mean(signal)) / np.std(signal)

def extract_features(signal, sampling_rate):
    """Extracts statistical features for non-overlapping windows in the signal."""
    window_size = sampling_rate
    return np.array([
        [np.mean(window), np.std(window), np.max(window), np.min(window)]
        for i in range(0, len(signal), window_size)
        if len((window := signal[i:i + window_size])) == window_size
    ])

def label_data(signal_length, sampling_rate, seizures):
    """Generates labels for each time window based on seizure intervals."""
    labels = np.zeros(signal_length // sampling_rate, dtype=int)
    if seizures:
        for start, end in seizures:
            start_idx, end_idx = int(start * sampling_rate), int(end * sampling_rate)
            labels[start_idx // sampling_rate:end_idx // sampling_rate] = 1
    return labels

def process_file(args):
    data_dir, summary_info, channels, sampling_rate = args
    file_features = []
    file_labels = []

    edf_files = [file for file in os.listdir(data_dir) if file.endswith(".edf")]
    for file in tqdm(edf_files, desc=f"Processing {data_dir}", unit="file"):
        edf_path = os.path.join(data_dir, file)
        seizures = summary_info.get(file, [])
        combined_features = None  # Initialize as None
        for channel_name in channels:
            signal = extract_signal(edf_path, channel_name)
            if signal is None:
                continue
            normalized_signal = normalize_signal(signal)
            channel_features = extract_features(normalized_signal, sampling_rate)
            if combined_features is None:
                combined_features = channel_features
            else:
                combined_features = np.hstack((combined_features, channel_features))
        if combined_features is not None:  # Ensure valid features exist
            labels = label_data(len(signal), sampling_rate, seizures)
            file_features.extend(combined_features)
            file_labels.extend(labels)
    return file_features, file_labels

def preprocess_multiple_patients(data_dirs, summary_files, channels, sampling_rate, output_dir):
    if os.path.exists(os.path.join(output_dir, "features.npy")) and os.path.exists(os.path.join(output_dir, "labels.npy")):
        print("[INFO] Processed data already exists. Skipping preprocessing.")
        return

    all_features = []
    all_labels = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []
    for data_dir, summary_file in zip(data_dirs, summary_files):
        print(f"[INFO] Preparing task for patient directory: {data_dir}")
        summary_info = parse_summary_file(summary_file)
        tasks.append((data_dir, summary_info, channels, sampling_rate))

    print("[INFO] Starting parallel processing.")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_file, tasks), total=len(tasks), desc="Patients", unit="patient"))

    for features, labels in results:
        all_features.extend(features)
        all_labels.extend(labels)

    print("[INFO] Finished preprocessing all patient data.")
    np.save(os.path.join(output_dir, "features.npy"), np.array(all_features))
    np.save(os.path.join(output_dir, "labels.npy"), np.array(all_labels))
    print(f"[INFO] Data saved to {output_dir}/features.npy and {output_dir}/labels.npy")

if __name__ == "__main__":
    patients = ["chb01", "chb02", "chb05", "chb16", "chb21", "chb23"]
    data_directories = [f"../data/{patient}" for patient in patients]
    summary_files = [f"../data/{patient}/{patient}-summary.txt" for patient in patients]

    channels = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FZ-CZ", "CZ-PZ", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8",
        "T8-P8", "P8-O2"
    ]

    preprocess_multiple_patients(data_directories, summary_files, channels, sampling_rate=256, output_dir="../processed_data")
