import torch
import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from collections import Counter
import logging
import time
import os
import json
from itertools import product
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torch.nn import DataParallel
# Constants
DATA_DIRECTORY = "/media/Korpora/nova/data/DFG_A1_A2b"
FILE_NAME = "processed_unified_dataset.hdf5"
FILE_PATH = Path(DATA_DIRECTORY) / FILE_NAME

IGNORED_LABELS = ['Garbage', 'NoAnno']
BATCH_SIZE = [ 64, 128]
EPOCHS = 100
LEARNING_RATES = [0.001, 0.0001]
WINDOW_SIZES = [25, 50]
STEP_SIZES = [8, 25, 50]
DROPOUT_RATES = [0.2, 0.5]
TARGET_LABELS = ['IPOS', 'INEG'] # focus on evaluate on the minority classes
# Setup logging
def setup_logging(session_name):
    # Create directory for logs if it doesn't exist
    log_dir = f'logs/WS{WINDOW_SIZES}_SS{STEP_SIZES}_BS{BATCH_SIZE}_LR{LEARNING_RATES}'
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(session_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f'{log_dir}/{session_name}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, log_dir

class SlidingWindowDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# def create_sliding_windows(features, labels, window_size, step_size, logger):
#     logger.info(f"Creating sliding windows: window_size={window_size}, step_size={step_size}")
#     windowed_features = []
#     windowed_labels = []
#     for i in range(0, features.shape[0] - window_size + 1, step_size):
#         window = features[i:i + window_size]
#         if window.shape[0] == window_size:
#             windowed_features.append(window)
#             label_counts = Counter(labels[i:i + window_size])
#             windowed_labels.append(label_counts.most_common(1)[0][0])
#
#     #logger.info(f"createed sliding_windows {len(windowed_features), len(windowed_labels)}")
#     print(f"createed sliding_windows {len(windowed_features), len(windowed_labels)}")
#     return windowed_features, windowed_labels
def create_sliding_windows(features, labels, frame_numbers, window_size, step_size):
    windowed_features = []
    windowed_labels = []
    for i in range(0, len(frame_numbers) - window_size + 1, step_size):
        window_start_frame = frame_numbers[i]
        window_end_frame = frame_numbers[i + window_size - 1]

        # Verify continuity of the frame numbers in the window
        if window_end_frame - window_start_frame + 1 == window_size:
            windowed_features.append(features[i:i + window_size])
            # Use the most common label within the window
            window_label = Counter(labels[i:i + window_size]).most_common(1)[0][0]
            windowed_labels.append(window_label)

    windowed_features = torch.stack(windowed_features)
    windowed_labels = torch.tensor(windowed_labels, dtype=torch.long)
    return windowed_features, windowed_labels


class AdvancedLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, bidirectional=True, dropout_rate=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_rate if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Use dropout and select the last time step
        out = self.fc(out)
        return out
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def load_data(feature_type):
    start_time = time.time()
    with h5py.File(FILE_PATH, "r") as f:
        labels = [label.decode('utf-8') for label in f['labels_infant'][:]]
        session_names = [session.decode('utf-8') for session in f['session_names'][:]]
        frame_numbers = np.array(f['frame_numbers'][:])
        valid_indices = [i for i, label in enumerate(labels) if label not in IGNORED_LABELS]
        labels = np.array(labels)[valid_indices]
        session_names = np.array(session_names)[valid_indices]
        frame_numbers = frame_numbers[valid_indices]

        if feature_type == "both":
            dino_features = torch.tensor(f['infant_dino_features'][:][valid_indices], dtype=torch.float32)
            w2v_features = torch.tensor(f['w2v_features'][:][valid_indices], dtype=torch.float32)
            features = torch.cat((dino_features, w2v_features), dim=1)
        elif feature_type == "dino":
            features = torch.tensor(f['infant_dino_features'][:][valid_indices], dtype=torch.float32)
        elif feature_type == "w2v":
            features = torch.tensor(f['w2v_features'][:][valid_indices], dtype=torch.float32)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = torch.tensor(labels, dtype=torch.long)
    elapsed_time = time.time() - start_time
    print(f"Data loaded and filtered in {elapsed_time:.2f}s. Feature type '{feature_type}' selected, shape {features.shape}")

    return features, labels, session_names, frame_numbers, label_encoder


def select_sessions_for_training(labels, sessions):
    session_label_counts = Counter()
    for label, session in zip(labels, sessions):
        if label in TARGET_LABELS:
            session_label_counts[session] += 1
    selected_sessions = [session for session, count in session_label_counts.most_common(10)]
    return selected_sessions


# def train_and_evaluate(train_features, train_labels, val_features, val_labels,label_encoder, device, logger):
#     logger.info(f"Start training for {len(train_features)} training shape. train_labels shape '{len(train_labels)}'")
#     all_classes = np.arange(len(label_encoder.classes_))
#     train_dataset = SlidingWindowDataset(train_features, train_labels)
#     val_dataset = SlidingWindowDataset(val_features, val_labels)
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train_and_evaluate(train_features, train_labels, val_features, val_labels, model_params, label_encoder, logger, log_name):
    log_dir = f'tensorboard_logs_20042023/{log_name}'
    writer = SummaryWriter(log_dir)
    all_classes = np.arange(len(label_encoder.classes_))

    train_dataset = SlidingWindowDataset(train_features, train_labels)
    val_dataset = SlidingWindowDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=model_params['batch_size'], shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = AdvancedLSTMClassifier(input_size=2048, hidden_size=256, num_classes= len(np.unique(train_labels.numpy())), num_layers=2, bidirectional=True, dropout_rate=model_params['dropout_rate']).to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)  # Wrap model for 2 GPUS
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Implement early stopping logic
    min_val_loss = float('inf')
    patience = 3
    trigger_times = 0

    # Training loop with early stopping
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f'Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}')
        writer.add_scalar('Training Loss', avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_trues = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_trues.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        logger.info(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}')
        writer.add_scalar('Validation Loss', val_loss, epoch)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                logger.info('Early stopping!')
                break

    logger.info(f'all_trues: {all_trues},  all_preds: {all_preds}')
    report = classification_report(all_trues, all_preds,labels=all_classes, target_names=label_encoder.classes_, zero_division=0)
    logger.info('Classification Report:\n' + report)
    logger.info('Confusion Matrix:\n' + str(confusion_matrix(all_trues, all_preds)))

    del model, optimizer, criterion
    torch.cuda.empty_cache()
    writer.close()
    # Prepare results to return
    return {
        'all_trues': all_trues,
        'all_preds': all_preds,
        'labels': label_encoder.classes_.tolist()
    }


def main():
    # feature type, can be changed to "dino" or "w2v"
    feature_type = "both"
    features, labels, session_names, frame_numbers, label_encoder = load_data(feature_type)
    selected_sessions = select_sessions_for_training(
        [label_encoder.inverse_transform([lbl])[0] for lbl in labels.numpy()], session_names)
    print(selected_sessions)


    all_results = {}
    hyper_params = list(product(WINDOW_SIZES, STEP_SIZES, BATCH_SIZE, DROPOUT_RATES, LEARNING_RATES))
    logging.info(hyper_params)
    for session in selected_sessions:
        training_session_mask = session_names != session
        train_session_features, train_session_labels, train_session_frame_numbers = features[training_session_mask], labels[training_session_mask], \
        frame_numbers[training_session_mask]

        val_session_mask = session_names == session
        val_session_features, val_session_labels, val_session_frame_numbers = features[val_session_mask], \
        labels[val_session_mask], \
            frame_numbers[val_session_mask]


        for window_size, step_size, batch_size, dropout_rate, learning_rate in hyper_params:
            log_name = f'session_{session}_ws{window_size}_ss{step_size}_bs{batch_size}_dr{dropout_rate}_lr{learning_rate}'
            logger, log_dir = setup_logging(log_name)
            train_features, train_labels = create_sliding_windows(train_session_features, train_session_labels, train_session_frame_numbers, window_size, step_size)
            val_features, val_labels = create_sliding_windows(val_session_features, val_session_labels, val_session_frame_numbers, window_size, step_size)

            model_params = {
                'window_size': window_size,
                'step_size': step_size,
                'batch_size': batch_size,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate
            }
            results = train_and_evaluate(train_features, train_labels, val_features, val_labels, model_params,
                                         label_encoder, logger, log_name)
            all_results[
                f'session_{session}_ws{window_size}_ss{step_size}_bs{batch_size}_dr{dropout_rate}_lr{learning_rate}'] = results

    # # Save all results in one JSON file
    # with open('all_results.json', 'w') as f:
    #     json.dump(all_results, f)
    print(all_results)


    # Assuming all_results is convertible to a DataFrame
    log_name = f'all_results'
    logger_, log_dir_ = setup_logging(log_name)
    df_results = pd.DataFrame.from_dict(all_results)
    logger_.info(df_results)
    df_results.to_json('all_results.json')

    df = pd.DataFrame({
        'Session': list(all_results.keys()),
        'Results': [','.join(map(str, result)) if isinstance(result, list) else result for result in
                    all_results.values()]
    })

    # Save to CSV
    df.to_csv('all_results.csv', index=False)


if __name__ == '__main__':
    main()