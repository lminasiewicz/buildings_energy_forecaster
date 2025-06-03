import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from ..model.transformer_model import EncoderOnlyTransformerModel
from .dataset_loader import EnergyDataset

TRAIN_PATH = "../data/processed_data/train_dataset.csv"
TEST_PATH = "../data/processed_data/test_dataset.csv"
STATIC_FEATURES_PATH = "../data/processed_data/static_features.csv"
MODEL_SAVE_DIR = "../models"

EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
SEQ_LENGTH = 48  # how many time steps per sample
INPUT_DIM = 11  # number of features (excluding target and building ID)
BUILDING_EMBED_DIM = 16 # Building ID embed size
STATIC_FEAT_DIM = 1 # Only surface area
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------

static_features = {}
with open(STATIC_FEATURES_PATH, "r") as file:
    rows = file.readlines()
    for row in rows:
        static_features[int(row[0])] = float(row[1])

train_dataset = EnergyDataset(TRAIN_PATH, SEQ_LENGTH, static_features)
test_dataset = EnergyDataset(TEST_PATH, SEQ_LENGTH, static_features)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # shuffle=True is highly questionable but it's arguably worse without it
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


model = EncoderOnlyTransformerModel(
    input_dim=INPUT_DIM,
    building_count=max(train_dataset.building_id) + 1,
    building_embed_dim=BUILDING_EMBED_DIM,
    static_feat_dim=STATIC_FEAT_DIM,
    d_model=64,
    nhead=4,
    num_layers=2,
    dropout=0.1,
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training
train_losses = []
test_losses = []

best_loss = float('inf')
best_model_state = None
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0
    for building_id, feats, targets in train_loader:
        feats, targets = feats.to(DEVICE), targets.to(DEVICE)
        building_id = building_id.to(DEVICE)
        optimizer.zero_grad()
        output = model(building_id, feats)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * feats.size(0)

    avg_train_loss = epoch_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    model.eval()
    epoch_test_loss = 0
    with torch.no_grad():
        for building_id, feats, targets in test_loader:
            feats, targets = feats.to(DEVICE), targets.to(DEVICE)
            building_id = building_id.to(DEVICE)
            output = model(building_id, feats)
            loss = criterion(output, targets)
            epoch_test_loss += loss.item() * feats.size(0)

    avg_test_loss = epoch_test_loss / len(test_loader.dataset)
    test_losses.append(avg_test_loss)

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_model_state = model.state_dict()

    elapsed = (time.time() - start_time) / (epoch + 1)
    print(f"\rEpoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Avg Time/Epoch: {elapsed:.2f}s", end="")

# Save best model
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
save_path = os.path.join(MODEL_SAVE_DIR, timestamp)
os.makedirs(save_path, exist_ok=True)
torch.save(best_model_state, os.path.join(save_path, "model.pt"))

# Plot loss
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid()
plt.savefig(os.path.join(save_path, "loss.png"))
plt.show()
