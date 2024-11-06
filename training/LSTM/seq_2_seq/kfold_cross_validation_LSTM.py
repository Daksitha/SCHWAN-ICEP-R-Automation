import gc
from sklearn.utils.class_weight import compute_class_weight
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
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import re
from itertools import groupby
from tqdm import tqdm

# Constants
DATA_DIRECTORY = "C:/Users/withanda/Documents/Github/schwan_project_codebase/data"
FILE_NAME = "processed_unified_dataset_seamless.hdf5"
FILE_PATH = Path(DATA_DIRECTORY) / FILE_NAME

# debug mode or select less sessions
NUMBER_FRAMES = 0# one session have about 9000 frames
K_OUTTER = 10
K_INNER =4

IGNORED_LABELS = ['Garbage', 'NoAnno']
BATCH_SIZE = [128, 64]
EPOCHS = 100
HIDDEN_SIZE = [128]
DROPOUT_RATES = [0.25]
OPTIMIZER = [optim.SGD]
CRITERIA = [nn.CrossEntropyLoss]
BATCH_NORM = [False]
WEIGHT_DECAY = [0.0]
WINDOW_STRIDE_COMBINATION = [ (50, 25)]


ROLE = ["infant", "caretaker", ]
MODALITY = ["dino", "w2v", "both"]
def setup_logging(session_name, log_to_file=False):
    # Define the log directory based on global constants
    log_dir = f'logs/{ROLE}/{MODALITY}/NF{NUMBER_FRAMES}_WS_SS{WINDOW_STRIDE_COMBINATION}_BS{BATCH_SIZE}_HS{HIDDEN_SIZE}'
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(session_name)
    logger.setLevel(logging.INFO)

    # Check if logging to file is enabled
    if log_to_file:
        # File handler setup
        file_handler = logging.FileHandler(f'{log_dir}/{session_name}.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger, log_dir

class ChunkedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class Seq2SeqLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, bidirectional=True, dropout_rate=0.5, batch_normalise=False):
        super().__init__()
        self.num_classes = num_classes
        self.batch_normalise = batch_normalise

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional,
                            dropout=dropout_rate if num_layers > 1 else 0.0)

        # Batch normalization layer applied to the outputs of the LSTM across the feature dimension
        self.batch_norm = nn.BatchNorm1d(hidden_size * (2 if bidirectional else 1))

        # Fully connected layer that outputs the logits for each class
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        # Pass input through LSTM layer
        out, _ = self.lstm(x)

        if self.batch_normalise:
            # Reshape output for batch normalization
            batch_size, seq_length, features = out.shape
            out = out.contiguous().view(-1, features)  # Flatten output for batch norm
            out = self.batch_norm(out)
            out = out.view(batch_size, seq_length, -1)  # Reshape back to (batch, seq_length, features)

        # Pass through fully connected layer to get predictions
        out = self.fc(out)
        return out

def load_data(feature_type,role, number_frames=0):
    start_time = time.time()
    max_index = None
    if number_frames > 0:
        max_index = number_frames

    with h5py.File(FILE_PATH, "r") as f:
        if max_index is not None:
            labels = [label.decode('utf-8') for label in f[f'labels_{role}'][:max_index]]
            session_names = [session.decode('utf-8') for session in f['session_names'][:max_index]]
            frame_numbers = np.array(f['frame_numbers'][:max_index])
            valid_indices = [i for i, label in enumerate(labels) if label not in IGNORED_LABELS]
            labels = np.array(labels)[valid_indices]
            session_names = np.array(session_names)[valid_indices]
            frame_numbers = frame_numbers[valid_indices]

            if feature_type == "both":
                dino_features = torch.tensor(f[f'{role}_dino_features'][:max_index][valid_indices], dtype=torch.float32)
                w2v_features = torch.tensor(f['w2v_features'][:max_index][valid_indices], dtype=torch.float32)
                features = torch.cat((dino_features, w2v_features), dim=1)
            elif feature_type == "dino":
                features = torch.tensor(f[f'{role}_dino_features'][:max_index][valid_indices], dtype=torch.float32)
            elif feature_type == "w2v":
                features = torch.tensor(f['w2v_features'][:max_index][valid_indices], dtype=torch.float32)
        else:
            labels = [label.decode('utf-8') for label in f[f'labels_{role}'][:]]
            session_names = [session.decode('utf-8') for session in f['session_names'][:]]
            frame_numbers = np.array(f['frame_numbers'][:])
            valid_indices = [i for i, label in enumerate(labels) if label not in IGNORED_LABELS]
            labels = np.array(labels)[valid_indices]
            session_names = np.array(session_names)[valid_indices]
            frame_numbers = frame_numbers[valid_indices]

            if feature_type == "both":
                dino_features = torch.tensor(f[f'{role}_dino_features'][:][valid_indices], dtype=torch.float32)
                w2v_features = torch.tensor(f['w2v_features'][:][valid_indices], dtype=torch.float32)
                features = torch.cat((dino_features, w2v_features), dim=1)
            elif feature_type == "dino":
                features = torch.tensor(f[f'{role}_dino_features'][:][valid_indices], dtype=torch.float32)
            elif feature_type == "w2v":
                features = torch.tensor(f['w2v_features'][:][valid_indices], dtype=torch.float32)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = torch.tensor(labels, dtype=torch.long)
    elapsed_time = time.time() - start_time
    print(f"Data loaded for {role} {feature_type} and filtered in {elapsed_time:.2f}s. Feature type '{feature_type}' selected, shape {features.shape}")

    return features, labels, session_names, frame_numbers, label_encoder


def create_session_specific_chunks(features, labels, frame_numbers, sessions, window_size, stride):
    # Initialize containers for the chunked data
    chunked_features = []
    chunked_labels = []

    # Get the unique sessions to process each separately
    unique_sessions = np.unique(sessions)

    # Process each session individually to maintain continuity within sessions
    for session in unique_sessions:
        # Filter the data by session
        session_mask = (sessions == session)
        session_features = features[session_mask]
        session_labels = labels[session_mask]
        session_frame_numbers = frame_numbers[session_mask]

        # Create overlapping chunks within this session
        i = 0
        while i < len(session_frame_numbers) - window_size + 1:
            # Check for continuity in the frame numbers
            window_start_frame = session_frame_numbers[i]
            window_end_frame = session_frame_numbers[i + window_size - 1]

            if window_end_frame - window_start_frame + 1 == window_size:
                chunked_features.append(session_features[i:i + window_size])
                chunked_labels.append(session_labels[i:i + window_size])
                i += stride  # Move by the stride
            else:
                i += 1  # Move to the next frame and check again

    # Stack all the collected chunks if available
    if chunked_features and chunked_labels:
        chunked_features = torch.stack(chunked_features)
        chunked_labels = torch.stack(chunked_labels)
        return chunked_features, chunked_labels
    else:
        raise RuntimeError("Error in chunking the dataset")


def train_and_evaluate(model, train_loader,val_loader, device, criterion, optimizer,log_name, session,logger, role, modality,scheduler, train_writer, patience=5 ):

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_params = None
    inner_hyper_paramdict = {'best_val_loss': 0.0, 'best_epoc': 0}
    best_val_f1 = 0
    best_model_wts = None
    best_model = None
    best_model_per_train = None

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            if isinstance(model, torch.nn.DataParallel):
                num_classes = model.module.num_classes
            else:
                num_classes = model.num_classes
            loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_writer.add_scalar('Training Loss', avg_train_loss, epoch)
        logger.info(f'Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}')

        model.eval()
        total_val_loss = 0
        # with torch.no_grad():
        #     for features, labels in val_loader:
        #         features, labels = features.to(device), labels.to(device)
        #         outputs = model(features)
        #         if isinstance(model, torch.nn.DataParallel):
        #             num_classes = model.module.num_classes
        #         else:
        #             num_classes = model.num_classes
        #         loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
        #         total_val_loss += loss.item()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for val_features, val_labels in val_loader:
                val_features = val_features.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_features)

                if isinstance(model, torch.nn.DataParallel):
                    num_classes = model.module.num_classes
                else:
                    num_classes = model.num_classes
                loss = criterion(val_outputs.view(-1, num_classes), val_labels.view(-1))
                total_val_loss += loss.item()

                # Get the predicted class index for each timestep in each sequence
                _, preds = torch.max(val_outputs,
                                     dim=2)  # This changes the dimension for torch.max to 2, as outputs are [batch, seq_len, classes]

                # Flatten the predictions and labels to align them correctly
                # If outputs are [batch_size, sequence_length, num_classes], preds will be [batch_size, sequence_length]
                val_preds.extend(preds.cpu().numpy().flatten())
                val_targets.extend(val_labels.cpu().numpy().flatten())

        avg_val_loss = total_val_loss / len(val_loader)
        train_writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        logger.info(f'Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}')

        val_f1 = f1_score(val_targets, val_preds, average='macro')
        logger.info(f'Epoch {epoch + 1} Validation F1: {val_f1}')
        train_writer.add_scalar('Validation F1', val_f1, epoch)

        scheduler.step()

        if avg_val_loss < best_val_loss:
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                #best_model_wts = model.state_dict().copy()
                best_model = model
                best_opti = optimizer

            best_val_loss = avg_val_loss
            #best_model_params = model.state_dict().copy()  # Deep copy model params
            current_lr = optimizer.param_groups[0]['lr']
            best_model_per_train= {'best_val_loss': best_val_loss,
                        'best_epoch': epoch,
                        'best_lr':current_lr,
                        'best_val_macro_f1':best_val_f1,
                        'avg_train_loss':avg_train_loss}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                #print(f'Early stopping triggered after {epoch + 1} epochs.')
                break

    #final_metrics = {'hparam/best_val_loss': best_val_loss}
    hparams = {
        'role':role,
        'session':session,
        'optimizer': optimizer.__class__.__name__,
    }


    train_writer.add_hparams(
        hparam_dict=hparams,
        metric_dict=best_model_per_train
    )
    train_writer.close()
    return best_model,best_opti, best_model_per_train

def evaluate(model, loader, device, label_encoder, log_dir, session,log_name, val_writer, dict_loss,hparams ):

    #model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)

            # Get the predicted class index for each timestep in each sequence
            _, preds = torch.max(outputs, dim=2)  # This changes the dimension for torch.max to 2, as outputs are [batch, seq_len, classes]

            # Flatten the predictions and labels to align them correctly
            # If outputs are [batch_size, sequence_length, num_classes], preds will be [batch_size, sequence_length]
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    average_type='weighted'
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=average_type, zero_division=0)
    f1_unwei = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average=average_type, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=average_type, zero_division=0)

    metric_dict = {'test_accuracy': accuracy,
                   'test_weight_f1_score': f1,
                   'test_macro_f1_score': f1_unwei,
                   'test_precision': precision, 'test_recall': recall}
    final_metric = metric_dict | dict_loss

    val_writer.add_hparams(
        hparam_dict=hparams,
        metric_dict=final_metric
    )

    #print(classification_report(all_labels, all_preds))
    all_trues_dencoded = label_encoder.inverse_transform(all_labels)
    all_preds_dencoded = label_encoder.inverse_transform(all_preds)

    results_dict = {
       'all_trues': all_trues_dencoded,
       'all_preds': all_preds_dencoded,
    }
    df_results = pd.DataFrame.from_dict(results_dict)
    path_csv = Path(log_dir).parent / "json" / session
    path_csv.mkdir(exist_ok=True, parents=True)
    df_results.to_json(path_csv / f"{log_name}_trues_preds.json")
    # logger.info(f'all_trues: {all_trues_dencoded},  all_preds: {all_preds_dencoded}')
    # report = classification_report(all_trues_dencoded, all_preds_dencoded,
    #                               zero_division=0)
    # logger.info(f'Best Classification Report:{report}\n')
    # logger.info('Best Confusion Matrix:\n' + str(confusion_matrix(all_labels, all_preds)))

    return f1, accuracy, f1_unwei, precision, recall

def select_sessions_for_training(labels, sessions,target_labels, how_many_to_pick=10):
    session_label_counts = Counter()
    for label, session in zip(labels, sessions):
        if label in target_labels:
            session_label_counts[session] += 1
    selected_sessions = [session for session, count in session_label_counts.most_common(how_many_to_pick)]
    return selected_sessions

def prepare_datasets(features, labels, frame_numbers, session_names, train_sessions, val_sessions, window_size, step_size):
    # Filter data for training sessions
    train_idx = np.isin(session_names, train_sessions)
    train_features, train_labels = create_session_specific_chunks(
        features[train_idx], labels[train_idx], frame_numbers[train_idx], session_names[train_idx], window_size, step_size
    )

    # Filter data for validation sessions
    val_idx = np.isin(session_names, val_sessions)
    val_features, val_labels = create_session_specific_chunks(
        features[val_idx], labels[val_idx], frame_numbers[val_idx], session_names[val_idx], window_size, step_size
    )

    return train_features, train_labels, val_features, val_labels

def load_checkpoint(filepath, device,weight_decay ):
    """Loads a model checkpoint from a file."""
    checkpoint = torch.load(filepath, map_location=device)
    model = Seq2SeqLSTMClassifier(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hyperparams']['hidden_size'],
        num_classes=checkpoint['num_classes'],
        dropout_rate=checkpoint['hyperparams']['dropout_rate'],
        batch_normalise=checkpoint['hyperparams']['batch_norm']
    )
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint['hyperparams']

def parse_and_group_sessions(sessions):
    # Regular expression to parse typical session names like "NP001", "P011"
    pattern = re.compile(r'([a-zA-Z]+)(\d+)')

    # Parse session names
    parsed_sessions = [(match.group(1), int(match.group(2))) for session in sessions if (match := pattern.match(session))]

    # Group by prefix and sort by number
    grouped = {}
    for prefix, group in groupby(sorted(parsed_sessions, key=lambda x: (x[0], x[1])), key=lambda x: x[0]):
        numbers = sorted([g[1] for g in group])
        if numbers:
            # Check if numbers form a continuous sequence
            if max(numbers) - min(numbers) == len(numbers) - 1:
                range_str = f"{min(numbers)}-{max(numbers)}"
            else:
                range_str = '-'.join(map(str, numbers))  # List all numbers separated by commas
            grouped[prefix] = range_str

    # Construct a consolidated name
    consolidated_name = '_'.join([f"{prefix}{num_range}" for prefix, num_range in grouped.items()])
    return consolidated_name

def update_best_model(current_model,current_optimizer, best_model_info, epoch, f1_score, max_epochs, hparams,  num_classes, input_size_, best_model_dict, logger):
    """ Updates the best model based on the composite score. """
    def calculate_composite_score(f1_score, epoch):
        """ Normalize and weight the epoch and F1 score. """
        normalized_epoch = 1 - (epoch / max_epochs)  # Less epochs is better
        normalized_f1 = f1_score
        # Adjust weights as necessary. Here we use 70% weight for F1 and 30% for epoch
        return 0.5 * normalized_f1 + 0.5 * normalized_epoch

    current_score = calculate_composite_score(f1_score, epoch)
    if best_model_info is None or current_score > best_model_info['score']:
        print(f"found a best model with these parameters: {best_model_dict}, with current_score {current_score} ")
        logger.info(f"found a best model with these parameters: {best_model_dict}, with current_score {current_score} ")
        return {
            'state_dict': current_model.state_dict(),
            'optimizer_state': current_optimizer.state_dict(),
            'score': current_score,
            'epoch': epoch,
            'f1_score': f1_score,
            'hparams' : hparams,
            'num_classes':num_classes,
            'input_size':input_size_,
        }
    return best_model_info

def main():
    #features, labels, sessions, label_encoder = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for role in ROLE:
        features = None
        labels = None
        session_names= None
        label_encoder = None
        for feature_type in MODALITY:


            features, labels, session_names, frame_numbers, label_encoder = load_data(feature_type, role, NUMBER_FRAMES)
            sessions = np.array(session_names)
            unique_sessions = np.unique(sessions)

            results_file_path = f'{role}_{feature_type}_cross_validation_results_.json'
            with open(results_file_path, 'w') as file:
                json.dump([], file)
            hyper_params = list(
                product(WINDOW_STRIDE_COMBINATION, BATCH_SIZE, DROPOUT_RATES, HIDDEN_SIZE, OPTIMIZER, WEIGHT_DECAY,
                        BATCH_NORM))

            # Outer Loop (Test Sessions)
            kf_outer = KFold(n_splits=K_OUTTER, shuffle=True, random_state=42)


            for fold, (train_index, test_index) in enumerate(
                    tqdm(kf_outer.split(unique_sessions), total=kf_outer.n_splits, desc="Processing Outer K Folds")):

                # Initialize best_model_info for every test and training
                best_model_info = None

                train_sessions = unique_sessions[train_index]
                test_sessions = unique_sessions[test_index]
                test_session_name = parse_and_group_sessions(test_sessions)

                print(f"One session leave out {test_session_name}")

                # Inner Loop (Hyperparameter Tuning using Validation Set)
                kf_inner = KFold(n_splits=K_INNER, shuffle=True, random_state=42)


                for ((window_size, step_size), batch_size, dropout_rate, hidden_size, optimizer_class, weight_decay, isBatchNorm) in tqdm(hyper_params, desc="Evaluating Hyperparameters"):

                    hparams = {
                        'test_session_name':test_session_name,
                        'feature_type': feature_type,
                        'window_size': window_size,
                        'step_size': step_size,
                        'batch_size': batch_size,
                        'dropout_rate': dropout_rate,
                        'hidden_size': hidden_size,
                        'optimizer': optimizer_class.__name__,
                        'weight_decay': weight_decay,
                        'batch_norm': isBatchNorm
                    }


                    log_name = f'{test_session_name}_{role}_{feature_type}_ws{window_size}_ss{step_size}_bs{batch_size}_dr{dropout_rate}_hs{hidden_size}_WD{weight_decay}_BN_{isBatchNorm}'
                    logger, log_dir = setup_logging(log_name, log_to_file=True)
                    train_writer = SummaryWriter(f"runs/train/NF{NUMBER_FRAMES}/{role}/{feature_type}/{test_session_name}/{log_name}'")
                    val_writer = SummaryWriter(f"runs/val/NF{NUMBER_FRAMES}/{role}/{feature_type}/{test_session_name}/{log_name}'")

                    #avg_val_performance = 0
                    #count = 0
                    num_classes = len(np.unique(labels))
                    input_size_ = features.shape[1]

                    model = Seq2SeqLSTMClassifier(input_size=input_size_, hidden_size=hidden_size,
                                                  num_classes=num_classes, dropout_rate=dropout_rate,
                                                  batch_normalise=isBatchNorm)
                    model = nn.DataParallel(model)  # Enable multi-GPU
                    model.to(device)
                    model.train()

                    class_weights = compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(labels.cpu().numpy()),
                        y=labels.cpu().numpy()
                    )
                    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
                    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

                    if optimizer_class == optim.Adam:
                        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
                    elif optimizer_class == optim.SGD:
                        # used in dinov2 linear eval
                        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay)


                    #print(f"Hyper parameter training")

                    for inner_train_index, val_index in tqdm(kf_inner.split(train_sessions), total=kf_inner.n_splits, desc="Inner K-Fold Progress",  leave=True):
                        best_model_, best_opti, best_model_per_train_dict = None, None, None
                        train_features, train_labels = None, None
                        val_features, val_labels  = None, None

                        inner_train_sessions = train_sessions[inner_train_index]
                        val_sessions = train_sessions[val_index]

                        # Convert session names to indices for training and validation
                        train_idx = np.isin(session_names, inner_train_sessions)
                        val_idx = np.isin(session_names, val_sessions)

                        # Prepare training and validation datasets
                        train_features, train_labels = create_session_specific_chunks(
                            features[train_idx], labels[train_idx], frame_numbers[train_idx], session_names[train_idx],
                            window_size, step_size
                        )
                        val_features, val_labels = create_session_specific_chunks(
                            features[val_idx], labels[val_idx], frame_numbers[val_idx], session_names[val_idx],
                            window_size, step_size
                        )

                        train_dataset = ChunkedDataset(train_features, train_labels)
                        val_dataset = ChunkedDataset(val_features, val_labels)

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)


                        epoch_length = len(train_idx) // batch_size

                        max_iter = EPOCHS * epoch_length
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)


                        best_model_, best_opti,best_model_per_train_dict = train_and_evaluate(model, train_loader,val_loader, device, criterion, optimizer,log_name, test_session_name,logger, role,feature_type,scheduler,train_writer, patience=5)

                        best_model_info = update_best_model(best_model_,best_opti, best_model_info, best_model_per_train_dict['best_epoch'], best_model_per_train_dict['best_val_macro_f1'],EPOCHS, hparams, num_classes, input_size_, best_model_per_train_dict, logger)



                    del model, train_loader, val_loader, optimizer, best_model_, best_opti, best_model_per_train_dict
                    torch.cuda.empty_cache()
                    gc.collect()


                if best_model_info:


                    checkpoint = {
                        'model_state_dict': best_model_info['state_dict'],
                        'optimizer_state_dict': best_model_info['optimizer_state'],
                        'best_performance': best_model_info['f1_score'],
                        'hyperparams': best_model_info['hparams'],
                        'input_size' :best_model_info['input_size'],
                        'num_classes' : best_model_info['num_classes']
                    }
                    checkpoint_filename = os.path.join('model', f'best_model_{test_session_name}_{role}_{feature_type}.pth.tar')
                    torch.save(checkpoint, f=checkpoint_filename)
                    best_checkpoint_path = checkpoint_filename


                if best_checkpoint_path:
                    model, optimizer, hyperparams = load_checkpoint(best_checkpoint_path, device, class_weights)
                    #criterion = nn.CrossEntropyLoss()
                    test_idx = np.isin(session_names, test_sessions)
                    test_features, test_labels = create_session_specific_chunks(
                        features[test_idx], labels[test_idx], frame_numbers[test_idx], session_names[test_idx],
                        hyperparams['window_size'], hyperparams['step_size'],
                    )
                    test_dataset = ChunkedDataset(test_features, test_labels)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    best_inner_dict = {'best_performance': best_model_info['f1_score'],
                                      'score': best_model_info['score']}

                    test_f1, test_accuracy, test_f1_unwei, test_precision, test_recall = evaluate(model, test_loader, device, label_encoder, log_dir, test_session_name,log_name, val_writer, best_inner_dict, hyperparams)

                    # Write results to JSON file
                    with open(results_file_path, 'r+') as file:
                        results = json.load(file)
                        results.append({'test_session_name':test_session_name ,'fold': fold, 'accuracy': test_accuracy, 'f1_score_weighted': test_f1, 'precision': test_precision, 'test_f1_unwei':test_f1_unwei, 'test_recall':test_recall})
                        file.seek(0)
                        json.dump(results, file)

                    print(f"Fold {fold}: Accuracy = {test_accuracy}, F1 Score (W) = {test_f1}, Precision = {test_precision}")

                    val_writer.add_scalar('F1/Test', test_f1, 0)
                    val_writer.add_scalar('Accuracy/Test', test_accuracy, 0)
                    val_writer.add_scalar('Precision/Test', test_precision, 0)
                    val_writer.add_scalar('Recall/Test', test_recall, 0)
                    val_writer.close()
                #best_model_state = select_best_model(model_histories)



            del features, labels, session_names, frame_numbers, label_encoder
            torch.cuda.empty_cache()
            gc.collect()
if __name__ == '__main__':
    main()


