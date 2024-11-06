from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_session_data(session_folder):
    """Load data from a single session's HDF5 file."""
    hdf5_file_path = session_folder / f"{session_folder.name}_w2v_dino_labels_data.hdf5"
    if hdf5_file_path.exists():
        with h5py.File(hdf5_file_path, 'r') as f:
            return {
                'w2v_features': f['w2v_features'][:],
                'infant_dino_features': f['infant_dino_features'][:],
                'caretaker_dino_features': f['caretaker_dino_features'][:],
                'labels_infant': [label.decode('utf-8') for label in f['infant_labels'][:]],
                'labels_caretaker': [label.decode('utf-8') for label in f['caretaker_labels'][:]],
                'frame_numbers': f['frame_numbers'][:] if 'frame_numbers' in f else np.array([], dtype=np.int32),
                'session_names': f['session_names'][:]
            }
    return None


def sequential_normalization(features):
    """Apply sequential normalization to features."""
    scaler_standard = StandardScaler()
    features_standardized = scaler_standard.fit_transform(features)
    scaler_min_max = MinMaxScaler(feature_range=(-1, 1))
    features_min_max_scaled = scaler_min_max.fit_transform(features_standardized)
    return features_min_max_scaled

def aggregate_data(data_directory):
    """Aggregate data from all session folders."""
    base_directory = Path(data_directory)
    session_folders = [d for d in base_directory.iterdir() if d.is_dir()]
    aggregated_data = {
        'w2v_features': [],
        'infant_dino_features': [],
        'caretaker_dino_features': [],
        'labels_infant': [],
        'labels_caretaker': [],
        'frame_numbers': [],
        'session_names': []
    }

    for session_folder in tqdm(session_folders, desc="Processing Sessions"):
        session_data = load_session_data(session_folder)

        if session_data:
            for key in aggregated_data:
                aggregated_data[key].extend(session_data[key])


    for key in ['w2v_features', 'infant_dino_features', 'caretaker_dino_features']:
        aggregated_data[key] = np.array(aggregated_data[key], dtype=np.float32)
    for key in ['labels_infant', 'labels_caretaker', 'session_names']:
        aggregated_data[key] = np.array(aggregated_data[key], dtype='S')
    aggregated_data['frame_numbers'] = np.array(aggregated_data['frame_numbers'], dtype=np.int32)

    return aggregated_data
def normalize_features(aggregated_data):
    """Normalize features separately and save the processed data."""
    for feature_key in ['w2v_features', 'infant_dino_features', 'caretaker_dino_features']:
        aggregated_data[feature_key] = sequential_normalization(aggregated_data[feature_key])

    return aggregated_data


def save_processed_data(unified_file_path, aggregated_data):
    """Save the aggregated and normalized data into a unified HDF5 file."""
    with h5py.File(unified_file_path, 'w') as f:
        for key, value in aggregated_data.items():

            if isinstance(value[0], str):
                value = np.array(value, dtype='S')
            f.create_dataset(key, data=value)
    print(f"Processed data saved to {unified_file_path}")



def main():
    data_directory = "S:/nova/data/DFG_A1_A2b"
    unified_file_name = "processed_unified_dataset.hdf5"
    unified_file_path = Path(data_directory) / unified_file_name


    aggregated_data = aggregate_data(data_directory)


    normalized_features = normalize_features(aggregated_data)


    save_processed_data(unified_file_path, normalized_features)

if __name__ == "__main__":
    main()
