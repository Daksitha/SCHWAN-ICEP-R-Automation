import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from pathlib import Path
from torch.utils.data import Dataset
import h5py
from sklearn.preprocessing import LabelEncoder
import time
def load_dataset(file_path, role):
    #logging.info(f"Loading data for {role}...")
    print(f"Loading data for {role}...")

    with h5py.File(file_path, 'r') as hf:
        features = hf[f'{role}_dino_features'][:]
        labels = np.array([label.decode('utf-8') for label in hf[f'labels_{role}'][:]])
        session_names = np.array([name.decode('utf-8') for name in hf['session_names'][:]])

        valid_indices = labels != "Garbage"
        features = features[valid_indices]
        labels = labels[valid_indices]
        session_names = session_names[valid_indices]

        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        classes = encoder.classes_

    print(f"Data loaded: Data shape: {features.shape}")
    return features, encoded_labels, session_names, classes, encoder

# Define your dataset class, logger, and neural network as previously defined.
class SimpleLinearClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        """
        Initializes the SimpleLinearClassifier.

        Parameters:
        input_features (int): Number of input features.
        num_classes (int): Number of output classes.
        """
        super(SimpleLinearClassifier, self).__init__()
        # Define the linear layer
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the classifier.

        Parameters:
        x (Tensor): Input tensor of shape (batch_size, input_features).

        Returns:
        Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.linear(x)

class DinoFeaturesDataset(Dataset):
    """Dataset class for loading data."""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def session_based_k_fold(X, y, session_names, n_splits=5):
    """Generate session-based k-fold splits."""
    unique_sessions = np.unique(session_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(unique_sessions):
        train_sessions = unique_sessions[train_index]
        test_sessions = unique_sessions[test_index]

        train_idx = np.isin(session_names, train_sessions)
        test_idx = np.isin(session_names, test_sessions)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        yield X_train, X_test, y_train, y_test


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """Train and evaluate the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss = total_loss / len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    return val_loss, accuracy


def grid_search_cv(file_path, role, learning_rates, regularizations, num_epochs=10, batch_size=32, n_splits=3):
    features, labels, session_names, class_names, encoder = load_dataset(file_path, role)
    num_classes = len(class_names)

    results = []

    for lr in learning_rates:
        for reg in regularizations:
            fold_results = []
            for X_train, X_val, y_train, y_val in session_based_k_fold(features, labels, session_names,
                                                                       n_splits=n_splits):
                train_dataset = DinoFeaturesDataset(X_train, y_train)
                val_dataset = DinoFeaturesDataset(X_val, y_val)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                model = SimpleLinearClassifier(features.shape[1], num_classes)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

                val_loss, accuracy = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer,
                                                        num_epochs)
                fold_results.append((val_loss, accuracy))

            avg_val_loss, avg_accuracy = np.mean(fold_results, axis=0)
            results.append({'lr': lr, 'reg': reg, 'val_loss': avg_val_loss, 'accuracy': avg_accuracy})
            print(f"LR: {lr}, Reg: {reg}, Avg Val Loss: {avg_val_loss:.4f}, Avg Accuracy: {avg_accuracy:.2f}%")

    return results


# Main execution block
if __name__ == "__main__":
    data_directory = "/media/Korpora/nova/data/DFG_A1_A2b"
    unified_file_name = "processed_unified_dataset.hdf5"
    file_path = Path(data_directory) / unified_file_name

    roles = 'infant'

    learning_rates = [0.001, 0.01, 0.1]
    regularizations = [1e-5, 1e-4, 1e-3]

    results = grid_search_cv(file_path, roles, learning_rates, regularizations, num_epochs=300, batch_size=64)
    print(results)
