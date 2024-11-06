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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_dataset(file_path, role):
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

    print(f"Data loaded: Data shape: {features.shape}, Labels shape: {labels.shape}, Session names shape: {session_names.shape}")
    return features, encoded_labels, session_names, classes, encoder

class SimpleLinearClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(SimpleLinearClassifier, self).__init__()
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x):
        return self.linear(x)

class DinoFeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def session_based_k_fold(X, y, session_names, n_splits=5):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        total_train_samples = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            total_train_samples += data.size(0)

        avg_train_loss = train_loss / total_train_samples
        train_accuracy = 100. * train_correct / total_train_samples
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        model.eval()
        total_loss = 0
        correct = 0
        total_val_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_val_samples += data.size(0)
                all_preds.extend(pred.view(-1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_loss = total_loss / total_val_samples
        accuracy = 100. * correct / total_val_samples
        unweighted_precision = precision_score(all_targets, all_preds, average='macro')
        unweighted_recall = recall_score(all_targets, all_preds, average='macro')
        unweighted_f1 = f1_score(all_targets, all_preds, average='macro')
        weighted_accuracy = accuracy_score(all_targets, all_preds)
        weighted_precision = precision_score(all_targets, all_preds, average='weighted')
        weighted_recall = recall_score(all_targets, all_preds, average='weighted')
        weighted_f1 = f1_score(all_targets, all_preds, average='weighted')

        print(f'Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
        print(f'Unweighted Precision: {unweighted_precision:.4f}, Recall: {unweighted_recall:.4f}, F1: {unweighted_f1:.4f}')
        print(f'Weighted Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}')

    return val_loss, accuracy, unweighted_precision, unweighted_recall, unweighted_f1, weighted_precision, weighted_recall, weighted_f1

def grid_search_cv(file_path, role, learning_rates, regularizations, num_epochs=10, batch_size=32, n_splits=3):
    features, labels, session_names, class_names, encoder = load_dataset(file_path, role)
    num_classes = len(class_names)

    results = []

    for lr in learning_rates:
        for reg in regularizations:
            fold_results = []
            print(f'Starting grid search for LR={lr}, Reg={reg}')
            for X_train, X_val, y_train, y_val in session_based_k_fold(features, labels, session_names, n_splits=n_splits):
                train_dataset = DinoFeaturesDataset(X_train, y_train)
                val_dataset = DinoFeaturesDataset(X_val, y_val)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                model = SimpleLinearClassifier(features.shape[1], num_classes)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

                metrics = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs)
                fold_results.append(metrics)

            avg_metrics = np.mean(fold_results, axis=0)
            results.append({'lr': lr, 'reg': reg, 'val_loss': avg_metrics[0], 'accuracy': avg_metrics[1],
                            'unweighted_precision': avg_metrics[2], 'unweighted_recall': avg_metrics[3],
                            'unweighted_f1': avg_metrics[4], 'weighted_precision': avg_metrics[5],
                            'weighted_recall': avg_metrics[6], 'weighted_f1': avg_metrics[7]})
            print(f"LR: {lr}, Reg: {reg}, Avg Val Loss: {avg_metrics[0]:.4f}, Avg Accuracy: {avg_metrics[1]:.2f}%, "
                  f"Unweighted Precision: {avg_metrics[2]:.4f}, Recall: {avg_metrics[3]:.4f}, F1: {avg_metrics[4]:.4f}, "
                  f"Weighted Precision: {avg_metrics[5]:.4f}, Recall: {avg_metrics[6]:.4f}, F1: {avg_metrics[7]:.4f}")

    best_parameters = max(results, key=lambda x: x['accuracy'])
    print(f"Best parameters found: {best_parameters}")

    return results

if __name__ == "__main__":
    data_directory = "/media/Korpora/nova/data/DFG_A1_A2b"
    unified_file_name = "processed_unified_dataset.hdf5"
    file_path = Path(data_directory) / unified_file_name

    roles = 'infant'
    learning_rates = [0.001, 0.01, 0.1]
    regularizations = [1e-5, 1e-4, 1e-3]

    results = grid_search_cv(file_path, roles, learning_rates, regularizations, num_epochs=300, batch_size=64)
    print(results)
