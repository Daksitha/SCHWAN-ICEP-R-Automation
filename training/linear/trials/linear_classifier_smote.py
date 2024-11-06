import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from pathlib import Path
from tensorboardX import SummaryWriter
import pandas as pd
from colorama import init, Fore, Style
init(autoreset=True)
# Configure logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ColorLogger:
    def __init__(self, log_file='linear_classifier_smote.log'):
        # Set up basic configuration for logging
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)  # Set the logger to debug level

        # Create console handler with higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)  # Set console to debug level for all messages
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)  # Set file handler to debug level
        fh.setFormatter(formatter)

        # Add both handlers to the logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def log(self, message, level, color):
        """
        Log a message with the specified level and color.

        :param message: str, the message to log
        :param level: str, the logging level ('info', 'warning', 'error', 'critical')
        :param color: str, the color code from Colorama (e.g., 'Fore.RED')
        """
        color_dict = {
            'red': Fore.RED,
            'green': Fore.GREEN,
            'yellow': Fore.YELLOW,
            'blue': Fore.BLUE,
            'magenta': Fore.MAGENTA,
            'cyan': Fore.CYAN,
            'white': Fore.WHITE
        }

        # Get the color attribute from Colorama
        color_attr = color_dict.get(color.lower(), Fore.WHITE)

        # Set the level method based on the provided level
        log_method = {
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical
        }.get(level.lower(), self.logger.info)

        # Prepare colored message for console output
        colored_message = f'{color_attr}{message}{Style.RESET_ALL}'

        # Log the message
        if self.logger.handlers[0].stream.isatty():  # Check if the handler is a terminal
            log_method(colored_message)  # Use colored output in console
        else:
            log_method(message)  # Log without color in non-terminal outputs

logger = ColorLogger()
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

def load_dataset(file_path, role, logger=logger):
    #logging.info(f"Loading data for {role}...")
    logger.log(f"Loading data for {role}...", "info", "magenta")
    start_time = time.time()
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

    load_time = time.time() - start_time
    logger.log(f"Data loaded in {load_time:.2f}s. Data shape: {features.shape}", "info", "magenta")
    return features, encoded_labels, session_names, classes, encoder

def apply_smote(X, y, logger=logger):

    logger.log("Applying SMOTE...", "info", "magenta")
    start_time = time.time()
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    smote_time = time.time() - start_time
    logger.log(f"SMOTE applied in {smote_time:.2f}s. Original shape: {X.shape}, Resampled shape: {X_res.shape}", "info", "magenta")

    return X_res, y_res

# def train_test_split_by_session(features, labels, session_names, test_size=0.4, val_size=0.1):
#     logging.info("Starting train-test split by session...")
#     start_time = time.time()
#     df = pd.DataFrame(data=np.column_stack((features, labels, session_names)), columns=['features', 'label', 'session'])
#     df['session_label'] = df['session'].astype(str) + "_" + df['label'].astype(str)
#     train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['session_label'], random_state=42)
#     train_df, val_df = train_test_split(train_val_df, test_size=val_size / (1 - test_size), stratify=train_val_df['session_label'], random_state=42)
#     split_time = time.time() - start_time
#     logging.info(f"Split completed in {split_time:.2f}s")
#     return train_df['features'].tolist(), val_df['features'].tolist(), test_df['features'].tolist(), train_df['label'].astype(int).tolist(), val_df['label'].astype(int).tolist(), test_df['label'].astype(int).tolist()
# def train_test_split_by_session(features, labels, session_names, test_size=0.3, val_size=0.1):
#     """ split data without considering even distribution of samples accross"""
#     unique_sessions = np.unique(session_names)
#     train_sessions, test_sessions = train_test_split(unique_sessions, test_size=test_size, random_state=42)
#     train_sessions, val_sessions = train_test_split(train_sessions, test_size=val_size, random_state=42)
#
#     train_idx = np.isin(session_names, train_sessions)
#     val_idx = np.isin(session_names, val_sessions)
#     test_idx = np.isin(session_names, test_sessions)
#
#     return features[train_idx], features[val_idx], features[test_idx], labels[train_idx], labels[val_idx], labels[test_idx]
#

def train_test_split_by_session(features, labels, session_names, test_size=0.4, val_size=0.1, logger=logger):
    """ split test, train data considering even distribution of samples accross"""
    #logging.info("Starting train-test split by session...")
    logger.log("Starting train-test split by session...", "info",
               "magenta")
    start_time = time.time()

    # Create a session-label combination identifier
    session_labels = np.array([f"{session}_{label}" for session, label in zip(session_names, labels)])
    unique_session_labels = np.unique(session_labels)

    # Split unique session-label combinations into train-test first
    train_val_session_labels, test_session_labels = train_test_split(
        unique_session_labels, test_size=test_size, random_state=42)

    # Further split train-val into train and validation sets
    train_session_labels, val_session_labels = train_test_split(
        train_val_session_labels, test_size=val_size / (1 - test_size), random_state=42)

    # Map session-label combinations back to indices
    train_idx = np.isin(session_labels, train_session_labels)
    val_idx = np.isin(session_labels, val_session_labels)
    test_idx = np.isin(session_labels, test_session_labels)

    # Extract corresponding subsets using indices
    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    split_time = time.time() - start_time
    #logging.info(f"Split completed in {split_time:.2f}s")
    logger.log(f"Split completed in {split_time:.2f}s", "info",
               "magenta")
    return X_train, X_val, X_test, y_train, y_val, y_test
def main():

    data_directory = "/media/Korpora/nova/data/DFG_A1_A2b"
    unified_file_name = "processed_unified_dataset.hdf5"
    file_path = Path(data_directory) / unified_file_name

    role = 'infant'

    features, labels, session_names, class_names, encoder = load_dataset(file_path, role)
    num_classes = len(class_names)
    num_epochs = 40
    batch_size = 128
    early_stopping_patience = 10

    logger.log(f"starting the script:{time.time()}, file_path{file_path}, num_epochs:{num_epochs},batch_size:{batch_size},early_stopping_patience:{early_stopping_patience}, num_classes:{num_classes},role:{role} ","info", "green")

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_by_session(features, labels, session_names)
    X_train, y_train = apply_smote(X_train, y_train)

    train_dataset = DinoFeaturesDataset(X_train, y_train)
    val_dataset = DinoFeaturesDataset(X_val, y_val)
    test_dataset = DinoFeaturesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedLinearClassifier(features.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    writer = SummaryWriter(log_dir='runs/dino_classifier')
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    logger.log("Starting training...", "info", "magenta")
    #logger.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        train_loss = total_loss / len(train_loader)
        logger.log(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s', "info", "green")
        #logger.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s')
        writer.add_scalar('Loss/Train', train_loss, epoch)

        # Validation
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_val_loss += loss.item()
                _, preds = torch.max(output, 1)
                correct += (preds == target).sum().item()
                total += target.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_loss = total_val_loss / len(val_loader)
        accuracy = 100 * correct / total
        logger.log(f'Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%', "info", "yellow")
        #logger.info(f'Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', accuracy, epoch)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                #logger.info('Early stopping triggered')
                logger.log("Early stopping triggered.", "info", "red")
                break

    # Evaluation on test set
    #logger.info("Evaluating on test set...")
    logger.log("Evaluating on test set...", "info", "magenta")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    total_test_loss = 0
    correct = 0
    total = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_test_loss += loss.item()
            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss = total_test_loss / len(test_loader)
    accuracy = 100 * correct / total
    logger.log(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%', "info", "green")
    #logger.info(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    #logger.info("Detailed classification report for test set:")
    logger.log("Detailed classification report for test set:", "info", "magenta")
    #report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    logger.log('\n' + classification_report(all_targets, all_preds, target_names=class_names), "info", "magenta")
    #logger.info('\n' + classification_report(all_targets, all_preds, target_names=class_names))

    writer.close()

if __name__ == "__main__":
    main()
