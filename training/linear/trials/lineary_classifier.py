import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
import numpy as np
import h5py
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
from pathlib import Path
from torch.optim.lr_scheduler import StepLR

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

def load_dataset(file_path, role):
    with h5py.File(file_path, 'r') as hf:
        print(f"loading data for {role}")
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
    return features, encoded_labels, session_names, classes

def train_test_split_by_session(features, labels, session_names, test_size=0.2, val_size=0.1):
    unique_sessions = np.unique(session_names)
    train_sessions, test_sessions = train_test_split(unique_sessions, test_size=test_size, random_state=42)
    train_sessions, val_sessions = train_test_split(train_sessions, test_size=val_size, random_state=42)

    train_idx = np.isin(session_names, train_sessions)
    val_idx = np.isin(session_names, val_sessions)
    test_idx = np.isin(session_names, test_sessions)

    return features[train_idx], features[val_idx], features[test_idx], labels[train_idx], labels[val_idx], labels[test_idx]


def train_epoch(model, device, train_loader, criterion, optimizer):
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
    return total_loss / len(train_loader)

def evaluate(model, device, loader, criterion, num_classes):
    model.eval()
    correct = 0
    total_loss = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
            c = (preds == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            all_preds.extend(output.softmax(dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    acc_per_class = [100 * class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in range(num_classes)]
    overall_accuracy = correct / sum(class_total)
    avg_loss = total_loss / len(loader)
    return overall_accuracy, avg_loss, acc_per_class, all_preds, all_targets


def plot_roc_curve(all_targets_one_hot, all_preds, num_classes, role):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_targets_one_hot[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{role} Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

# def train_test_split_by_session(features, labels, session_names, test_size=0.2, val_size=0.1):
#     unique_sessions = np.unique(session_names)
#     train_sessions, test_sessions = train_test_split(unique_sessions, test_size=test_size, random_state=42)
#
#
#     train_sessions, val_sessions = train_test_split(train_sessions, test_size=val_size, random_state=42)
#
#     train_idx = np.isin(session_names, train_sessions)
#     val_idx = np.isin(session_names, val_sessions)
#     test_idx = np.isin(session_names, test_sessions)
#
#     return features[train_idx], features[val_idx], features[test_idx], labels[train_idx], labels[val_idx], labels[test_idx]

def main():
    data_directory = "/media/Korpora/nova/data/DFG_A1_A2b"
    unified_file_name = "processed_unified_dataset.hdf5"
    file_path = Path(data_directory) / unified_file_name

    role = 'infant'
    features, labels, session_names, class_names = load_dataset(file_path, role)
    num_classes = len(class_names)
    num_epochs = 40
    batch_size = 64 * 2

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_by_session(features, labels, session_names, test_size=0.2, val_size=0.1)

    train_dataset = DinoFeaturesDataset(X_train, y_train)
    val_dataset = DinoFeaturesDataset(X_val, y_val)
    test_dataset = DinoFeaturesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedLinearClassifier(1024, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, device, train_loader, criterion, optimizer)

        scheduler.step()
        val_accuracy, val_loss, acc_per_class, all_preds, all_targets = evaluate(model, device, val_loader, criterion, num_classes)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

    test_accuracy, test_loss, acc_per_class, all_preds, all_targets = evaluate(model, device, test_loader, criterion, num_classes)
    print(f'Test Overall Accuracy: {test_accuracy:.4f}')
    for i, class_acc in enumerate(acc_per_class):
        print(f'Accuracy of class {i} ({class_names[i]}): {class_acc:.2f}%')

    all_targets_one_hot = label_binarize(all_targets, classes=np.arange(num_classes))
    plot_roc_curve(all_targets_one_hot, np.array(all_preds), num_classes, role)

    writer.close()

if __name__ == "__main__":
    main()
