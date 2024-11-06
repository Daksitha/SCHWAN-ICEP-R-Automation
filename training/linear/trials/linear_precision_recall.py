import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class LinearClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.linear(x)

class DinoFeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

def load_dataset(file_path, role):
    with h5py.File(file_path, 'r') as hf:
        print(f"Loading data for {role}")
        features = hf[f'{role}_dino_features'][:]
        labels = np.array([label.decode('utf-8') for label in hf[f'labels_{role}'][:]])
        sessions = np.array([session.decode('utf-8') for session in hf[f'session_names'][:]])

        valid_indices = labels != "Garbage"
        features = features[valid_indices]
        labels = labels[valid_indices]
        sessions = sessions[valid_indices]

        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        classes = encoder.classes_
    return features, encoded_labels, classes, sessions

def train_test_split_by_session(features, labels, session_names, test_size=0.2, val_size=0.1):
    unique_sessions = np.unique(session_names)
    train_sessions, test_sessions = train_test_split(unique_sessions, test_size=test_size, random_state=42)
    train_sessions, val_sessions = train_test_split(train_sessions, test_size=val_size, random_state=42)

    train_idx = np.isin(session_names, train_sessions)
    val_idx = np.isin(session_names, val_sessions)
    test_idx = np.isin(session_names, test_sessions)

    return features[train_idx], features[val_idx], features[test_idx], labels[train_idx], labels[val_idx], labels[test_idx]

def scale_lr(learning_rate, batch_size):
    return learning_rate * batch_size / 1024

def evaluate(model, device, loader, num_classes, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(loader)
    overall_accuracy = correct / len(loader.dataset)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall = recall_score(all_targets, all_preds, average=None, zero_division=0)

    # Break down confusion matrix into TP, TN, FP, FN for each class
    TPs = np.diag(cm)  # True Positives are on the diagonal
    FNs = np.sum(cm, axis=1) - TPs  # False Negatives are the sum of the row minus TP
    FPs = np.sum(cm, axis=0) - TPs  # False Positives are the sum of the column minus TP
    TNs = np.sum(cm) - (FPs + FNs + TPs)  # True Negatives are the total sum minus the sum of FP, FN, and TP

    return avg_loss, overall_accuracy, TNs, FPs, FNs, TPs, precision, recall

def plot_metrics(metrics):
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))
    metric_names = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives', 'Precision', 'Recall']
    for i, metric in enumerate(metric_names):
        for class_index in range(4):
            axs[i//3, i%3].plot(metrics[metric][:, class_index], label=f'Class {class_index}')
        axs[i//3, i%3].set_title(metric)
        axs[i//3, i%3].legend()
    plt.tight_layout()
    plt.show()

def main():
    data_directory = "S:/nova/data/DFG_A1_A2b"
    unified_file_name = "processed_unified_dataset.hdf5"
    file_path = Path(data_directory) / unified_file_name

    role = 'infant'
    features, labels, class_names, session_names = load_dataset(file_path, role)
    num_classes = len(class_names)
    num_features = features.shape[1]

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_by_session(features, labels, session_names)

    train_dataset = DinoFeaturesDataset(X_train, y_train)
    val_dataset = DinoFeaturesDataset(X_val, y_val)
    test_dataset = DinoFeaturesDataset(X_test, y_test)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]

    model = LinearClassifier(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=scale_lr(learning_rates[0], batch_size))

    metrics = {'True Positives': [], 'True Negatives': [], 'False Positives': [], 'False Negatives': [], 'Precision': [], 'Recall': []}

    for epoch in range(20):  # Number of epochs
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_accuracy, tn, fp, fn, tp, precision, recall = evaluate(model, device, val_loader, num_classes, criterion)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Precision: {precision}, Recall: {recall}')

        # Store metrics for each epoch
        metrics['True Positives'].append(tp)
        metrics['True Negatives'].append(tn)
        metrics['False Positives'].append(fp)
        metrics['False Negatives'].append(fn)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)

    # Convert lists to arrays for easier indexing in plotting
    for key in metrics:
        metrics[key] = np.array(metrics[key])

    plot_metrics(metrics)

if __name__ == "__main__":
    main()
