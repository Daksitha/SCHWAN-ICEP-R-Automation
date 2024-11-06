import json
import time
import numpy as np
from torchmetrics.classification import MulticlassAccuracy

import torch.nn as nn
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchmetrics import ConfusionMatrix
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score
import h5py
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from pathlib import Path
from tqdm import tqdm

DATA_DIRECTORY = "C:/Users/withanda/Documents/Github/schwan_project_codebase/data"
FILE_NAME = "processed_unified_dataset_seamless.hdf5"
FILE_PATH = Path(DATA_DIRECTORY) / FILE_NAME
IGNORED_LABELS = ['Garbage', 'NoAnno']
EPOCHS = 100


# Assuming FILE_PATH and IGNORED_LABELS are defined
ROLE = ["caretaker", "infant"]
MODALITY = ["dino", "w2v", "both"]
NUMBER_FRAMES = 0

class DinoW2VDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# class LitClassifier(LightningModule):
#     def __init__(self, input_size, hidden_size, learning_rate, num_classes):
#         super(LitClassifier, self).__init__()
#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, hidden_size),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_size, num_classes)
#         )
#         self.learning_rate = learning_rate
#         self.loss = torch.nn.CrossEntropyLoss()
#
#     def forward(self, x):
#         return self.model(x)
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.loss(logits, y)
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.loss(logits, y)
#         self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.loss(logits, y)
#         self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
#         return optimizer
# class LitClassifier(LightningModule):
#     def __init__(self, input_size, hidden_size, learning_rate, num_classes):
#         super(LitClassifier, self).__init__()
#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, hidden_size),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_size, num_classes)
#         )
#         self.learning_rate = learning_rate
#         self.loss = torch.nn.CrossEntropyLoss()
#         self.num_classes = num_classes
#         # Initializing metrics
#         self.accuracy = MulticlassAccuracy(num_classes=num_classes)
#         self.precision = MulticlassPrecision(num_classes=num_classes, average='macro')
#         self.recall = MulticlassRecall(num_classes=num_classes, average='macro')
#         self.f1_score = MulticlassF1Score(num_classes=num_classes, average='macro')
#
#     def forward(self, x):
#         return self.model(x)
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.loss(logits, y)
#         self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.loss(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         # Logging metrics
#         self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
#         self.log('val_accuracy', self.accuracy(preds, y), on_epoch=True, prog_bar=True, logger=True)
#         self.log('val_precision', self.precision(preds, y), on_epoch=True, prog_bar=True, logger=True)
#         self.log('val_recall', self.recall(preds, y), on_epoch=True, prog_bar=True, logger=True)
#         self.log('val_f1', self.f1_score(preds, y), on_epoch=True, prog_bar=True, logger=True)
#         return {'val_loss': loss, 'val_preds': preds, 'val_targets': y}
#
#
#     def test_step(self, batch, batch_idx):
#         outputs = self.validation_step(batch, batch_idx)
#         self.log_dict({f'test_{k}': v for k, v in outputs.items() if k != 'loss'}, on_epoch=True, prog_bar=True, logger=True)
#         return outputs
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
#         return optimizer
#
#     def test_epoch_end(self, outputs):
#         all_preds = torch.cat([x['preds'] for x in outputs], dim=0)
#         all_targets = torch.cat([x['targets'] for x in outputs], dim=0)
#         cm = confusion_matrix(all_targets.cpu(), all_preds.cpu(), labels=range(self.num_classes))
#         fig = plt.figure(figsize=(10, 8))
#         sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
#         plt.xlabel('Predictions')
#         plt.ylabel('Actuals')
#         plt.title('Confusion Matrix')
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         plt.close(fig)
#         buf.seek(0)
#         cm_image = torch.tensor(plt.imread(buf, format='png'))
#         self.logger.experiment.add_image("Confusion Matrix", cm_image, self.current_epoch)
# # def train_model(config, input_size, num_classes, fold_id, train_dataloader, val_dataloader):
# #     model = LitClassifier(input_size=input_size, hidden_size=config["hidden_size"], learning_rate=config["lr"], num_classes=num_classes)
# #     logger = TensorBoardLogger(save_dir="logs", name="", version=".")
# #     checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir, filename="{epoch}-{val_loss:.2f}", monitor="val_loss", mode="min", save_top_k=1)
# #     early_stopping = EarlyStopping(monitor='val_f1', mode='max', patience=10, verbose=True)
# #     trainer = Trainer(
# #         max_epochs=EPOCHS,
# #         check_val_every_n_epoch=1,
# #         callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end"), checkpoint_callback, early_stopping],
# #         logger=logger,
# #         accelerator="cuda"
# #     )
# #     trainer.fit(model, train_dataloader, val_dataloader)

class LitClassifier(LightningModule):
    def __init__(self, input_size, num_classes, config):
        super(LitClassifier, self).__init__()
        layers = []
        layer_sizes = [input_size] + [config["hidden_size"]] * config["num_layers"]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            if "dropout_rate" in config:
                layers.append(nn.Dropout(config["dropout_rate"]))
        layers.append(nn.Linear(layer_sizes[-1], num_classes))

        self.model = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.precision_weighted = MulticlassPrecision(num_classes=num_classes, average='weighted')
        self.recall = MulticlassRecall(num_classes=num_classes, average='macro')
        self.recall_weighted = MulticlassRecall(num_classes=num_classes, average='weighted')
        self.f1_score = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.f1_score_weighted = MulticlassF1Score(num_classes=num_classes, average='weighted')
        self.confmat = ConfusionMatrix(num_classes=num_classes, task='multiclass')
        self.save_hyperparameters(config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.accuracy(preds, y), on_epoch=True)
        self.log("val_precision", self.precision(preds, y), on_epoch=True)
        self.log("val_precision_weighted", self.precision_weighted(preds, y), on_epoch=True)
        self.log("val_recall", self.recall(preds, y), on_epoch=True)
        self.log("val_recall_weighted", self.recall_weighted(preds, y), on_epoch=True)
        self.log("val_f1", self.f1_score(preds, y), on_epoch=True)
        self.log("val_f1_weighted", self.f1_score_weighted(preds, y), on_epoch=True)
        self.confmat(preds, y)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        # Saving confusion matrix to the logs
        cm = self.confmat.compute()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm.cpu().numpy(), annot=True, fmt='g', cmap='Blues', ax=ax)
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Actuals')
        ax.set_title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close(fig)
        self.logger.experiment.add_image("Confusion Matrix", 'confusion_matrix.png', self.current_epoch)

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.hparams["learning_rate"], momentum=0.9)
        else:  # Default to Adam if not specified
            return torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
def train_model(config, input_size, num_classes, fold_id, role, feature_type, train_dataloader, val_dataloader):
    # Format the logger and checkpoint directory names
    model_name = f"{role}_{feature_type}_fold_{fold_id}"
    log_dir = f"logs/{model_name}"
    model_dir = f"models/{model_name}"

    # Initialize the model using the provided config which includes all hyperparameters
    model = LitClassifier(input_size=input_size, num_classes=num_classes, config=config)

    # Setup TensorBoard Logger to track experiments
    logger = TensorBoardLogger(save_dir=log_dir, name="training_logs")

    # Checkpointing to save model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    # Early stopping based on the weighted F1 score
    early_stopping = EarlyStopping(
        monitor='val_f1_weighted',  # Change this to the metric you have in your model that you wish to monitor
        mode='max',
        patience=10,
        verbose=True
    )

    # Trainer configuration
    trainer = Trainer(
        max_epochs=config.get("max_epochs", 100),  # Default to 100 if not specified
        check_val_every_n_epoch=1,
        callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end"), checkpoint_callback, early_stopping],
        logger=logger,
        accelerator="cuda",
        limit_val_batches=1.0  # Ensure validation happens if val_dataloader is small
    )

    # Fit model to data
    trainer.fit(model, train_dataloader, val_dataloader)

    # Optionally return something, e.g., path to best model, to integrate with further evaluation or analysis
    return checkpoint_callback.best_model_path

def hyperparameter_tuning(input_size, num_classes, fold_id, role, feature_type, train_dataloader, val_dataloader):
    config = {
        "hidden_size": tune.choice([64, 128, 256]),
        "learning_rate": tune.loguniform(1e-4, 1e-10),
        "num_layers": tune.choice([1, 2, 3]),
        "dropout_rate": tune.uniform(0.1, 0.5),
        "optimizer": tune.choice(["adam", "sgd"])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=1,
        reduction_factor=2
    )
    analysis = tune.run(
        tune.with_parameters(train_model, input_size=input_size, num_classes=num_classes, fold_id=fold_id, role=role, feature_type=feature_type, train_dataloader=train_dataloader, val_dataloader=val_dataloader),
        resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=10,
        scheduler=scheduler
    )
    best_trial = analysis.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    return best_trial.config

def final_evaluation(input_size, num_classes, best_config, train_val_dataloader, test_dataloader, fold_id, role, feature_type):
    model = LitClassifier(input_size=input_size, hidden_size=best_config["hidden_size"], learning_rate=best_config["lr"], num_classes=num_classes)
    model_name = f"{role}_{feature_type}_fold_{fold_id}_final"
    logger = TensorBoardLogger("tb_logs", name=model_name)
    trainer = Trainer(logger=logger, accelerator="cuda")
    trainer.fit(model, train_val_dataloader)
    results = trainer.test(model, test_dataloader)

    # Save results to JSON
    results_path = f"results/{model_name}.json"
    with open(results_path, 'w') as fp:
        json.dump(results, fp)

    print(f"Results for fold {fold_id} saved to {results_path}")



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

def prepare_dataloaders(features, labels, session_names, train_sessions, test_sessions):
    train_indices = np.isin(session_names, train_sessions)
    test_indices = np.isin(session_names, test_sessions)

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    # Split the test set into validation and final test sets
    val_features, final_test_features, val_labels, final_test_labels = train_test_split(test_features, test_labels, test_size=0.2, random_state=42)

    # Creating dataloaders for training, validation, and final testing
    train_dataloader = DataLoader(DinoW2VDataset(train_features, train_labels), batch_size=32, shuffle=True)
    val_dataloader = DataLoader(DinoW2VDataset(val_features, val_labels), batch_size=32, shuffle=False)
    final_test_dataloader = DataLoader(DinoW2VDataset(final_test_features, final_test_labels), batch_size=32, shuffle=False)

    return train_dataloader, val_dataloader, final_test_dataloader
def main():
    # Iterate over each role and modality
    for role in ROLE:
        for feature_type in MODALITY:
            features, labels, session_names, frame_numbers, label_encoder = load_data(feature_type, role, NUMBER_FRAMES)

            sessions = np.array(session_names)
            unique_sessions = np.unique(sessions)
            num_classes = len(np.unique(labels))
            kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)

            # Processing each fold
            for fold, (train_index, test_index) in enumerate(tqdm(kf_outer.split(unique_sessions), total=kf_outer.n_splits, desc=f"Processing {role} {feature_type}")):
                train_sessions = unique_sessions[train_index]
                test_sessions = unique_sessions[test_index]

                # Prepare data loaders for training, validation, and testing
                train_dataloader, val_dataloader, final_test_dataloader = prepare_dataloaders(
                    features, labels, session_names, train_sessions, test_sessions)

                input_size = features.shape[1]
                best_config = hyperparameter_tuning(
                    input_size, num_classes, fold, role, feature_type, train_dataloader, val_dataloader)

                # Perform the final evaluation with the best configuration
                final_evaluation(input_size, num_classes, best_config, train_dataloader, final_test_dataloader, fold, role, feature_type)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()

