import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna


class EnhancedLinearClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(EnhancedLinearClassifier, self).__init__()
        self.layer1 = nn.Linear(num_features, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer2 = nn.Linear(512, 128)
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        return self.output_layer(x)


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


def evaluate(model, device, loader, criterion):
    model.eval()
    total_loss = 0
    total_preds = []
    total_targets = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, preds = torch.max(output, 1)
            total_preds.extend(preds.tolist())
            total_targets.extend(target.tolist())
    return total_loss / len(loader), total_preds, total_targets


def objective(trial):
    # Hyperparameters to be tuned
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

    # Model setup
    model = EnhancedLinearClassifier(1024, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(device))

    # Training process
    for epoch in range(10):  # Short epochs for demo
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        val_loss, val_preds, val_targets = evaluate(model, device, val_loader, criterion)
        weighted_f1 = f1_score(val_targets, val_preds, average='weighted')
        trial.report(weighted_f1, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return weighted_f1


def load_dataset(file_path, role, target_labels):
    with h5py.File(file_path, 'r') as hf:
        features = hf[f'{role}_dino_features'][:]
        labels = np.array([label.decode('utf-8') for label in hf[f'labels_{role}'][:]])
        valid_indices = np.isin(labels, target_labels)
        features = features[valid_indices]
        labels = labels[valid_indices]
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        classes = encoder.classes_
    return features, encoded_labels, classes


# def train_test_split_by_session(features, labels, test_size=0.2, val_size=0.1):
#     # Use scikit-learn's train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size + val_size)
#     X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_size / (test_size + val_size))
#     return X_train, X_val, X_test, y_train, y_val, y_test

def train_test_split_by_session(features, labels, session_names, test_size=0.4, val_size=0.1):
    unique_sessions = np.unique(session_names)
    train_sessions, test_sessions = train_test_split(unique_sessions, test_size=test_size, random_state=42)
    train_sessions, val_sessions = train_test_split(train_sessions, test_size=val_size, random_state=42)

    train_idx = np.isin(session_names, train_sessions)
    val_idx = np.isin(session_names, val_sessions)
    test_idx = np.isin(session_names, test_sessions)

    return features[train_idx], features[val_idx], features[test_idx], labels[train_idx], labels[val_idx], labels[test_idx]

def main():
    file_path = Path("/media/Korpora/nova/data/DFG_A1_A2b/processed_unified_dataset.hdf5")
    role = 'infant'
    target_labels = ['INON', 'INEU']
    features, labels, class_names = load_dataset(file_path, role, target_labels)
    num_classes = len(class_names)

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_by_session(features, labels)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

    train_dataset = DinoFeaturesDataset(X_train, y_train)
    val_dataset = DinoFeaturesDataset(X_val, y_val)
    test_dataset = DinoFeaturesDataset(X_test, y_test)

    global train_loader, val_loader, device, class_names, num_classes
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)  # Number of trials can be adjusted

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
