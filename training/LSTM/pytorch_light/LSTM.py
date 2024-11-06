import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import f1, precision, recall
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import classification_report

# Define the PyTorch Lightning LightningModule
class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5, batch_norm=False, learning_rate=1e-3, optimizer='adam'):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_size) if batch_norm else None
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the last output of the sequence
        if self.batch_norm:
            out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        f1_score = f1(y_hat.argmax(dim=1), y, num_classes=self.num_classes, average='macro')
        precision_score = precision(y_hat.argmax(dim=1), y, num_classes=self.num_classes, average='macro')
        recall_score = recall(y_hat.argmax(dim=1), y, num_classes=self.num_classes, average='macro')
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1', f1_score, prog_bar=True)
        self.log('val_precision', precision_score, prog_bar=True)
        self.log('val_recall', recall_score, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Optimizer not supported")
        return optimizer

# Define the PyTorch Lightning DataModule
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_features, train_labels, val_features, val_labels, batch_size=64, num_workers=4):
        super().__init__()
        self.train_features = train_features
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_labels = val_labels
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = MyDataset(self.train_features, self.train_labels)
        self.val_dataset = MyDataset(self.val_features, self.val_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# Function to train a model with Ray Tune
def train_with_tune(config, train_loader, val_loader):
    model = LSTMClassifier(**config)
    trainer = pl.Trainer(
        max_epochs=10,
        gpus=1,
        progress_bar_refresh_rate=0,
        logger=None
    )
    trainer.fit(model, train_loader, val_loader)

# Function to tune hyperparameters
def tune_hyperparameters(train_features, train_labels, val_features, val_labels):
    config = {
        "input_size": tune.choice([64, 128, 256]),
        "hidden_size": tune.choice([64, 128, 256]),
        "num_classes": len(np.unique(train_labels)),
        "dropout_rate": tune.choice([0.0, 0.2, 0.5]),
        "batch_norm": tune.choice([True, False]),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "optimizer": tune.choice(['adam', 'sgd'])
    }

    train_loader = DataLoader(MyDataset(train_features, train_labels), batch_size=64, num_workers=4)
    val_loader = DataLoader(MyDataset(val_features, val_labels), batch_size=64, num_workers=4)

    scheduler = ASHAScheduler(max_t=10, grace_period=1)
    reporter = CLIReporter()

    analysis = tune.run(
        tune.with_parameters(train_with_tune, train_loader=train_loader, val_loader=val_loader),
        resources_per_trial={"gpu": 1},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = analysis.get_best_trial("val_loss", "min", "last")
    best_hyperparameters = best_trial.config
    return best_hyperparameters

# Function to train the model
def train_model(train_features, train_labels, val_features, val_labels, best_hyperparameters):
    train_loader = DataLoader(MyDataset(train_features, train_labels), batch_size=64, num_workers=4)
    val_loader = DataLoader(MyDataset(val_features, val_labels), batch_size=64, num_workers=4)

    model = LSTMClassifier(**best_hyperparameters)
    trainer = pl.Trainer(max_epochs=10, gpus=1)
    trainer.fit(model, train_loader, val_loader)
    return model

# Function to evaluate the model
def evaluate_model(model, test_features, test_labels):
    test_loader = DataLoader(MyDataset(test_features, test_labels), batch_size=64, num_workers=4)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            preds = model(features)
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds))

def main():
    # Load your data
    train_features, train_labels, val_features, val_labels, test_features, test_labels = load_data()

    # Create log directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Set precision for float32 matrix multiplication
    torch.set_default_dtype(torch.float32)

    # Tune hyperparameters
    best_hyperparameters = tune_hyperparameters(train_features, train_labels, val_features, val_labels)

    # Train the model with best hyperparameters
    model = train_model(train_features, train_labels, val_features, val_labels, best_hyperparameters)

    # Evaluate the model
    evaluate_model(model, test_features, test_labels)

if __name__ == "__main__":
    main()
