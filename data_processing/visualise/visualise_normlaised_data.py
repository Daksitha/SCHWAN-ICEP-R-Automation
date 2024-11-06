import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data_from_hdf5(file_path):
    """Load data from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        labels_infant = np.array([label.decode('utf-8') for label in f['labels_infant'][:]])
        labels_caretaker = np.array([label.decode('utf-8') for label in f['labels_caretaker'][:]])
        w2v_features = f['w2v_features'][:]
        infant_dino_features = f['infant_dino_features'][:]
        caretaker_dino_features = f['caretaker_dino_features'][:]
    return labels_infant, labels_caretaker, w2v_features, infant_dino_features, caretaker_dino_features


def plot_label_counts(labels_infant, labels_caretaker):
    """Plot the counts of infant and caretaker labels."""
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(y=labels_infant)
    plt.title('Infant Label Counts')
    plt.xlabel('Count')
    plt.ylabel('Labels')

    plt.subplot(1, 2, 2)
    sns.countplot(y=labels_caretaker)
    plt.title('Caretaker Label Counts')
    plt.xlabel('Count')
    plt.ylabel('Labels')

    plt.tight_layout()
    plt.show()


def plot_feature_distributions(w2v_features, infant_dino_features, caretaker_dino_features):
    """Plot distributions of the first dimension of features for visualization."""
    plt.figure(figsize=(18, 6))


    plt.subplot(1, 3, 1)
    sns.histplot(w2v_features[:, 0], kde=True, bins=30)
    plt.title('Distribution of First Dimension of W2V Features')

    plt.subplot(1, 3, 2)
    sns.histplot(infant_dino_features[:, 0], kde=True, bins=30)
    plt.title('Distribution of First Dimension of Infant Dino Features')

    plt.subplot(1, 3, 3)
    sns.histplot(caretaker_dino_features[:, 0], kde=True, bins=30)
    plt.title('Distribution of First Dimension of Caretaker Dino Features')

    plt.tight_layout()
    plt.show()


def main():
    file_path = "S:/nova/data/DFG_A1_A2b/processed_unified_dataset.hdf5"
    labels_infant, labels_caretaker, w2v_features, infant_dino_features, caretaker_dino_features = load_data_from_hdf5(
        file_path)

    plot_label_counts(labels_infant, labels_caretaker)
    plot_feature_distributions(w2v_features, infant_dino_features, caretaker_dino_features)


if __name__ == '__main__':
    main()
