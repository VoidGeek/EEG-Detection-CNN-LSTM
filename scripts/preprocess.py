import os
import numpy as np
from pyedflib import EdfReader

def parse_summary_file(summary_file_path):
    seizure_info = {}
    current_file = None
    seizures = []
    try:
        with open(summary_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('File Name:'):
                    if current_file and seizures:
                        seizure_info[current_file] = seizures
                        seizures = []
                    current_file = line.split(':')[1].strip()
                elif 'Seizure' in line and 'Start Time' in line:
                    start_time = float(line.split(':')[1].strip().split()[0])
                elif 'Seizure' in line and 'End Time' in line:
                    end_time = float(line.split(':')[1].strip().split()[0])
                    seizures.append((start_time, end_time))
            if current_file and seizures:
                seizure_info[current_file] = seizures
    except Exception as e:
        print(f"[ERROR] Error parsing summary file: {e}")
    return seizure_info

def extract_signal(edf_file, channel_name):
    try:
        edf_reader = EdfReader(edf_file)
        channel_labels = edf_reader.getSignalLabels()
        if channel_name not in channel_labels:
            print(f"[INFO] Channel '{channel_name}' not found in {edf_file}. Skipping...")
            edf_reader.close()
            return None
        channel_index = channel_labels.index(channel_name)
        signal = edf_reader.readSignal(channel_index)
        edf_reader.close()
        return signal
    except Exception as e:
        print(f"[ERROR] Error extracting signal from {edf_file}: {e}")
        return None

def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def extract_features(signal, sampling_rate):
    features = []
    window_size = sampling_rate
    for i in range(0, len(signal), window_size):
        window = signal[i:i + window_size]
        if len(window) == window_size:
            features.append([
                np.mean(window),
                np.std(window),
                np.max(window),
                np.min(window)
            ])
    return np.array(features)

def label_data(signal_length, sampling_rate, seizures):
    labels = np.zeros(signal_length // sampling_rate)
    if seizures:
        for start, end in seizures:
            start_idx = int(start * sampling_rate)
            end_idx = int(end * sampling_rate)
            for idx in range(start_idx, end_idx, sampling_rate):
                window_idx = idx // sampling_rate
                if window_idx < len(labels):
                    labels[window_idx] = 1
    return labels

def preprocess_multiple_patients(data_dirs, summary_files, channels, sampling_rate=256, output_dir="./processed_data"):
    all_features = []
    all_labels = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for data_dir, summary_file in zip(data_dirs, summary_files):
        print(f"[INFO] Processing patient directory: {data_dir}")
        seizure_info = parse_summary_file(summary_file)
        for file in os.listdir(data_dir):
            if file.endswith(".edf"):
                edf_path = os.path.join(data_dir, file)
                seizures = seizure_info.get(file, [])
                combined_features = []
                for channel_name in channels:
                    signal = extract_signal(edf_path, channel_name)
                    if signal is None:
                        continue
                    normalized_signal = normalize_signal(signal)
                    channel_features = extract_features(normalized_signal, sampling_rate)
                    if len(combined_features) == 0:
                        combined_features = channel_features
                    else:
                        combined_features = np.hstack((combined_features, channel_features))
                if combined_features is not None and len(combined_features) > 0:
                    labels = label_data(len(signal), sampling_rate, seizures)
                    all_features.extend(combined_features)
                    all_labels.extend(labels)

    print("[INFO] Finished preprocessing all patient data.")
    np.save(os.path.join(output_dir, "features.npy"), np.array(all_features))
    np.save(os.path.join(output_dir, "labels.npy"), np.array(all_labels))
    print(f"[INFO] Data saved to {output_dir}/features.npy and {output_dir}/labels.npy")
