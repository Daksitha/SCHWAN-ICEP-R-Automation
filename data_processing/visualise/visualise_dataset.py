import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

import logging

logging.basicConfig(filename='analysis_log.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_scale_features(hdf5_file_path):
    """Load raw features from an HDF5 file and scale them."""
    with h5py.File(hdf5_file_path, 'r') as f:
        w2v_features_raw = f['w2v_features'][:]
        infant_dino_features_raw = f['infant_dino_features'][:]
        caretaker_dino_features_raw = f['caretaker_dino_features'][:]

        scaler_w2v = StandardScaler()
        scaler_infant_dino = StandardScaler()
        scaler_caretaker_dino = StandardScaler()

        w2v_features = scaler_w2v.fit_transform(w2v_features_raw)
        infant_dino_features = scaler_infant_dino.fit_transform(infant_dino_features_raw)
        caretaker_dino_features = scaler_caretaker_dino.fit_transform(caretaker_dino_features_raw)


        infant_labels = np.array([label.decode('utf-8') for label in f['infant_labels'][:]])
        caretaker_labels = np.array([label.decode('utf-8') for label in f['caretaker_labels'][:]])

    return w2v_features, infant_dino_features, caretaker_dino_features, infant_labels, caretaker_labels


def clean_features(df):
    """Clean the DataFrame by replacing Inf values with NaN and filling NaN values for numeric columns only."""
    logging.info("Before cleaning:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logging.info("NaN counts in numeric columns:\n%s", df[numeric_cols].isna().sum())
    logging.info("\nInf counts in numeric columns:\n%s", np.isinf(df[numeric_cols]).sum())

    for col in numeric_cols:
        df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)

    logging.info("\nAfter cleaning:")
    logging.info("NaN counts in numeric columns:\n%s", df[numeric_cols].isna().sum())


def analyze_features_statistics(df):
    """Analyze and print statistical report for the numeric columns in the DataFrame."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for column in numeric_cols:
        logging.info("\nStatistics for %s:", column)
        mean = df[column].mean()
        variance = df[column].var()
        min_val = df[column].min()
        max_val = df[column].max()
        median = df[column].median()
        skewness = df[column].skew()
        kurtosis = df[column].kurtosis()

        logging.info("Mean: %.2f, Variance: %.2f, Min: %.2f, Max: %.2f, Median: %.2f, Skewness: %.2f, Kurtosis: %.2f", mean, variance, min_val, max_val, median, skewness, kurtosis)

        _, p = stats.normaltest(df[column].dropna())
        logging.info("Normality test (D'Agostino and Pearson's) p-value: %.3f", p)


def visualize_data(df):
    """Visualize basic plots, the correlation matrix of the dataset, and mean plots for all numeric columns."""
    sns.set_style("whitegrid")


    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x='Infant Labels').set_title('Distribution of Infant Labels')
    plt.savefig('distribution_infant_labels.png')
    logging.info("Plot saved: distribution_infant_labels.png")

    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x='Caretaker Labels').set_title('Distribution of Caretaker Labels')
    plt.savefig('distribution_caretaker_labels.png')
    logging.info("Plot saved: distribution_caretaker_labels.png")
    plt.tight_layout()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    logging.info("Plot saved: correlation_matrix.png")



    for column in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df[column], label=f'Mean of {column}')
        plt.title(f'Mean Plot of {column}')
        plt.ylabel('Value')
        plt.xlabel('Index')
        plt.legend()
        plt.savefig(f'mean_plot_{column}.png')
        logging.info(f"Plot saved: mean_plot_{column}.png")
        plt.show()


def load_and_normalize_features(hdf5_file_path):
    """Load and normalize features from an HDF5 file."""
    with h5py.File(hdf5_file_path, 'r') as f:

        w2v_features_raw = f['w2v_features'][:]
        infant_dino_features_raw = f['infant_dino_features'][:]
        caretaker_dino_features_raw = f['caretaker_dino_features'][:]


    scaler = StandardScaler()
    w2v_features_normalized = scaler.fit_transform(w2v_features_raw)
    infant_dino_features_normalized = scaler.fit_transform(infant_dino_features_raw)
    caretaker_dino_features_normalized = scaler.fit_transform(caretaker_dino_features_raw)

    return w2v_features_normalized, infant_dino_features_normalized, caretaker_dino_features_normalized


def plot_normalized_feature_means(w2v_features_normalized, infant_dino_features_normalized,
                                  caretaker_dino_features_normalized):
    """Plot the mean of normalized features for comparison."""

    w2v_features_mean = np.mean(w2v_features_normalized, axis=1)
    infant_dino_features_mean = np.mean(infant_dino_features_normalized, axis=1)
    caretaker_dino_features_mean = np.mean(caretaker_dino_features_normalized, axis=1)


    plt.figure(figsize=(12, 6))
    plt.plot(w2v_features_mean, label='W2V Features Mean', alpha=0.7)
    plt.plot(infant_dino_features_mean, label='Infant Dino Features Mean', alpha=0.7)
    plt.plot(caretaker_dino_features_mean, label='Caretaker Dino Features Mean', alpha=0.7)

    plt.title('Comparison of Normalized Feature Means')
    plt.xlabel('Observation Index')
    plt.ylabel('Normalized Mean Value')
    plt.legend()
    plt.show()


def load_features(hdf5_file_path):
    """Load features from an HDF5 file without normalization."""
    with h5py.File(hdf5_file_path, 'r') as f:
        w2v_features_raw = f['w2v_features'][:]
        infant_dino_features_raw = f['infant_dino_features'][:]
        caretaker_dino_features_raw = f['caretaker_dino_features'][:]
    return w2v_features_raw, infant_dino_features_raw, caretaker_dino_features_raw


def normalize_features(features_raw):
    """Apply StandardScaler normalization to the features."""
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_raw)
    return features_normalized


def plot_feature_means_comparison(features_raw, features_normalized, feature_name):
    """Plot the mean of features before and after normalization for comparison."""
    features_raw_mean = np.mean(features_raw, axis=1)
    features_normalized_mean = np.mean(features_normalized, axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(features_raw_mean, label=f'{feature_name} Mean (Raw)', alpha=0.7)
    plt.plot(features_normalized_mean, label=f'{feature_name} Mean (Normalized)', alpha=0.7)

    plt.title(f'Comparison of {feature_name} Mean Before and After Normalization')
    plt.xlabel('Observation Index')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.show()

from pathlib import  Path
if __name__ == "__main__":
    #hdf5_file_path = "S:/nova/data/DFG_A1_A2b/unified_dataset.hdf5"
    hdf5_file_path = "S:/nova/data/DFG_A1_A2b/processed_unified_dataset.hdf5"

    if Path(hdf5_file_path).exists():

        w2v_features_raw, infant_dino_features_raw, caretaker_dino_features_raw = load_features(hdf5_file_path)


        df = pd.DataFrame({
            'W2V Features Mean': np.mean(w2v_features_raw, axis=1),
            'Infant Dino Features Mean': np.mean(infant_dino_features_raw, axis=1),
            'Caretaker Dino Features Mean': np.mean(caretaker_dino_features_raw, axis=1),
            #'Infant Labels': infant_labels,
            #'Caretaker Labels': caretaker_labels
        })

        #clean_features(df)
        #feature_columns = ['W2V Features Mean', 'Infant Dino Features Mean', 'Caretaker Dino Features Mean']
        analyze_features_statistics(df)


        w2v_features_normalized = normalize_features(w2v_features_raw)
        infant_dino_features_normalized = normalize_features(infant_dino_features_raw)
        caretaker_dino_features_normalized = normalize_features(caretaker_dino_features_raw)


        plot_feature_means_comparison(w2v_features_raw, w2v_features_normalized, "W2V Features")
        plot_feature_means_comparison(infant_dino_features_raw, infant_dino_features_normalized, "Infant Dino Features")
        plot_feature_means_comparison(caretaker_dino_features_raw, caretaker_dino_features_normalized,
                                      "Caretaker Dino Features")
    else:
        print(f"Unified dataset file not found at {hdf5_file_path}")

    # # Load and scale features
    # w2v_features, infant_dino_features, caretaker_dino_features, infant_labels, caretaker_labels = load_and_scale_features(
    #     hdf5_file_path)
    #
    # # Create DataFrame from the raw features for analysis
    # df = pd.DataFrame({
    #     'W2V Features Mean': np.mean(w2v_features, axis=1),
    #     'Infant Dino Features Mean': np.mean(infant_dino_features, axis=1),
    #     'Caretaker Dino Features Mean': np.mean(caretaker_dino_features, axis=1),
    #     'Infant Labels': infant_labels,
    #     'Caretaker Labels': caretaker_labels
    # })
    #
    # #clean_features(df)
    # #feature_columns = ['W2V Features Mean', 'Infant Dino Features Mean', 'Caretaker Dino Features Mean']
    # analyze_features_statistics(df)
    # visualize_data(df)
