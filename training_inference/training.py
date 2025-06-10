import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")
from dataset_loader import EnergyDataset
from buildings_energy_forecaster.model.transformer_model import EncoderOnlyTransformerModel


TRAIN_PATH = "../data/processed_data/train_dataset.csv"
TEST_PATH = "../data/processed_data/test_dataset.csv"
STATIC_FEATURES_PATH = "../data/processed_data/static_features.csv"
MODEL_SAVE_DIR = "../models"

EPOCHS = 20
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
SEQ_LENGTH = 48  # how many time steps per sample
INPUT_DIM = 10  # number of features (excluding target and building ID)
BUILDING_EMBED_DIM = 16 # Building ID embed size
STATIC_FEAT_DIM = 1 # Only surface area
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------

static_features = {}
with open(STATIC_FEATURES_PATH, "r") as file:
    rows = file.readlines()
    for row in rows:
        split_row = row.split(";")
        static_features[int(split_row[0])] = float(split_row[1])

train_dataset = EnergyDataset(TRAIN_PATH, SEQ_LENGTH, static_features)
test_dataset = EnergyDataset(TEST_PATH, SEQ_LENGTH, static_features)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # shuffle=True is highly questionable but it's arguably worse without it
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


model = EncoderOnlyTransformerModel(
    input_dim=INPUT_DIM,
    building_count=max(train_dataset.building_id) + 1,
    building_embed_dim=BUILDING_EMBED_DIM,
    static_feat_dim=STATIC_FEAT_DIM,
    d_model=32,
    nhead=2,
    num_layers=1,
    dropout=0.5,
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Training
train_losses = []
test_losses = []

best_loss = float('inf')
best_model_state = None
start_time = time.time()
len_train = len(train_loader.dataset)
len_test = len(test_loader.dataset)
print()

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0
    i = 1
    for building_id, feats, static, targets in train_loader:
        feats, targets = feats.to(DEVICE), targets.to(DEVICE)
        building_id = building_id.to(DEVICE)
        static = static.to(DEVICE)
        optimizer.zero_grad()
        output = model(feats, building_id, static)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item() * feats.size(0)
        print(f"\rTrain: {i}/{len_train}", end="")
        i += 1

    print("\r                               ", end="")
    avg_train_loss = epoch_train_loss / len_train
    train_losses.append(avg_train_loss)

    model.eval()
    epoch_test_loss = 0
    i = 1
    with torch.no_grad():
        for building_id, feats, static, targets in test_loader:
            feats, targets = feats.to(DEVICE), targets.to(DEVICE)
            building_id = building_id.to(DEVICE)
            static = static.to(DEVICE)
            output = model(feats, building_id, static)
            loss = criterion(output, targets)
            epoch_test_loss += loss.item() * feats.size(0)
            print(f"\rTest: {i}/{len_test}", end="")
            i += 1

    avg_test_loss = epoch_test_loss / len_test
    test_losses.append(avg_test_loss)

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_model_state = model.state_dict()

    elapsed = (time.time() - start_time) / (epoch + 1)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Avg Time/Epoch: {elapsed:.2f}s", end="")
    print()

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
