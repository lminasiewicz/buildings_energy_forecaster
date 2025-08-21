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


TRAIN_PATH = "../data/processed_data/alt_train_dataset.csv" # ALT!!!!!!!!!!!!!!!!!!!! WATCH OUT!!!!!!!!!!!!!
TEST_PATH = "../data/processed_data/alt_test_dataset.csv"
STATIC_FEATURES_PATH = "../data/processed_data/static_features.csv"
MODEL_SAVE_DIR = "../models"

EPOCHS = 30
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
SEQ_LENGTH = 72  # how many time steps per sample
WEIGHT_DECAY = 1e-4 # float, 0 if no weight decay
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

builcount = max(train_dataset.building_id) if (max(train_dataset.building_id) > max(test_dataset.building_id)) else max(test_dataset.building_id)

model = EncoderOnlyTransformerModel(
    input_dim=INPUT_DIM,
    building_count=builcount + 1,
    building_embed_dim=BUILDING_EMBED_DIM,
    static_feat_dim=STATIC_FEAT_DIM,
    d_model=256,
    nhead=8,
    num_layers=4,
    dropout=0.35
).to(DEVICE)

criterion = nn.MSELoss()
criterion2 = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    

# Training
train_losses = []
test_losses = []
train_losses2 = []
test_losses2 = []

best_loss = float('inf')
best_model_state = None
start_time = time.time()
len_train_batches = len(train_loader)
len_test_batches = len(test_loader)
len_train_samples = len(train_loader.dataset)
len_test_samples = len(test_loader.dataset)
print()

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0
    epoch_train_loss2 = 0
    i = 1
    for building_id, feats, static, targets in train_loader:
        feats, targets = feats.to(DEVICE), targets.to(DEVICE)
        building_id = building_id.to(DEVICE)
        static = static.to(DEVICE)
        optimizer.zero_grad()
        output = model(feats, building_id, static)
        loss = criterion(output, targets); loss2 = criterion2(output, targets)
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item() * feats.size(0); epoch_train_loss2 += loss2.item() * feats.size(0)
        print(f"\rTrain: {i}/{len_train_batches}", end="")
        i += 1
    print("\r                               ", end="")
    avg_train_loss = epoch_train_loss / len_train_samples
    avg_train_loss2 = epoch_train_loss2 / len_train_samples
    train_losses.append(avg_train_loss)
    train_losses2.append(avg_train_loss2)

    model.eval()
    epoch_test_loss = 0
    epoch_test_loss2 = 0
    i = 1
    with torch.no_grad():
        for building_id, feats, static, targets in test_loader:
            feats, targets = feats.to(DEVICE), targets.to(DEVICE)
            building_id = building_id.to(DEVICE)
            static = static.to(DEVICE)
            output = model(feats, building_id, static)
            loss = criterion(output, targets); loss2 = criterion2(output, targets)
            epoch_test_loss += loss.item() * feats.size(0); epoch_test_loss2 += loss2.item() * feats.size(0)
            print(f"\rTest: {i}/{len_test_batches}", end="")
            i += 1

    avg_test_loss = epoch_test_loss / len_test_samples
    avg_test_loss2 = epoch_test_loss2 / len_test_samples
    test_losses.append(avg_test_loss)
    test_losses2.append(avg_test_loss2)

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_model_state = model.state_dict()
    
    elapsed = (time.time() - start_time) / (epoch + 1)
    print()
    print(avg_train_loss, avg_test_loss, best_loss, epoch_test_loss, epoch_train_loss)
    print(avg_train_loss2, avg_test_loss2)
    print(train_losses2, test_losses2)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss2:.4f} | Test Loss: {avg_test_loss2:.4f} | Avg Time/Epoch: {elapsed:.2f}s")

# Save best model
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
save_path = os.path.join(MODEL_SAVE_DIR, timestamp)
os.makedirs(save_path, exist_ok=True)
torch.save(best_model_state, os.path.join(save_path, "model.pt"))
with open(f"{save_path}/params.txt", "w") as params_file:
    params_file.write(f"EPOCHS = {EPOCHS}\nLEARNING_RATE = {LEARNING_RATE}\nBATCH_SIZE = {BATCH_SIZE}\nSEQ_LENGTH = {SEQ_LENGTH}\nWEIGHT_DECAY = {WEIGHT_DECAY}\n\td_model={model.d_model}\n\tnhead={model.nhead}\n\tnum_layers={model.num_layers}\n\tdropout={model.dropout}")

# Plot loss
plt.figure()
plt.plot(train_losses, label="Train MSE")
plt.plot(test_losses, label="Test MSE")
plt.legend()
plt.title("MSE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid()
plt.savefig(os.path.join(save_path, "loss.png"))
plt.show()

plt.figure()
plt.plot(train_losses2, label="Train MAE")
plt.plot(test_losses2, label="Test MAE")
plt.legend()
plt.title("MAE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.grid()
plt.savefig(os.path.join(save_path, "loss2.png"))
plt.show()