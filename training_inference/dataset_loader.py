from torch.utils.data import Dataset
import torch
import numpy as np


class EnergyDataset(Dataset):
    def __init__(self, filepath: str, seq_length: int, static_lookup: dict[int, float]) -> None:
        raw = np.loadtxt(filepath, delimiter=';', dtype=str)
        self.building_id = raw[:, 0].astype(int)
        self.target = raw[:, 1].astype(np.float32)  # energy
        self.features = np.column_stack((raw[:, 2:-1].astype(np.float32), (raw[:, -1] == "True").astype(np.float32)))
        self.static_lookup = static_lookup

        self.samples = []
        for i in range(len(self.target) - seq_length):
            feat_seq = self.features[i:i + seq_length]
            target_seq = self.target[i:i + seq_length]
            building_seq = self.building_id[i:i + seq_length]
            if np.all(building_seq == building_seq[0]):
                self.samples.append((building_seq[0], feat_seq, target_seq))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        building_id, feat_seq, target_seq = self.samples[idx]
        static_feat = self.static_lookup[building_id]
        
        return (
            torch.tensor(building_id, dtype=torch.long), # Scalar tensor (for embedding, apparently good practice)
            torch.tensor(feat_seq, dtype=torch.float32), # (seq_len, feat_dim)
            torch.tensor(static_feat, dtype=torch.float32), # (static_feat_dim,)
            torch.tensor(target_seq, dtype=torch.float32), # (seq_len,)
        )