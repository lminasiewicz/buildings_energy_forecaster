import torch
import torch.nn as nn
from .postitional_encoding import PositionalEncoder


class EncoderOnlyTransformerModel(nn.Module):
    def __init__(self,
                 input_dim,               # number of numerical input features per timestep
                 building_count,          # total number of unique buildings
                 building_embed_dim,      # size of building ID embedding
                 static_feat_dim,         # number of continuous static features (e.g., dist_to_station, surface_area)
                 d_model=128,             # transformer model dimension
                 nhead=4,
                 num_layers=4,
                 dropout=0.1,
                 ) -> None:
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim

        # Embedding for building ID
        self.building_embedding = nn.Embedding(building_count, building_embed_dim)

        # Project full input (dynamic + embedding + static) to d_model
        self.input_projection = nn.Linear(input_dim + building_embed_dim + static_feat_dim, d_model)
        self.positional_encoder = PositionalEncoder(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(), # Add nonlinearity (apparently that's a good practice)
            nn.Linear(d_model // 2, 1)  # Output one step at a time
        )

        # Only for params.txt
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x: torch.Tensor, building_ids: torch.Tensor, static_feats: torch.Tensor, future_steps: int = 1) -> torch.Tensor:
        # x: [seq_len, batch_size, input_dim] (excluding building/static info)
        # building_ids: [batch_size]
        # static_feats: [batch_size, static_feat_dim]

        batch_size, seq_len, _ = x.shape
        building_embed = self.building_embedding(building_ids)  # [batch_size, embed_dim]

        if static_feats.dim() == 1:
            static_feats = static_feats.unsqueeze(-1) # future-proof for the case when we have more static features than 1

        # Repeat across sequence
        building_embed_seq = building_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, input_dim]
        static_feats_seq = static_feats.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, input_dim]

        x = torch.cat([x, building_embed_seq, static_feats_seq], dim=-1)  # [batch_size, seq_len, input_dim + emb + static]
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        x = self.positional_encoder(x)
        x = self.encoder(x)  # [batch_size, seq_len, d_model]

        if future_steps == 1:
            out = self.regressor(x)  # [batch_size, seq_len, 1]
            return out.squeeze(-1)   # [batch_size, seq_len]

        # Rolling forecast FIX THIS SOON!
        predictions = []
        current_step = last_step
        for _ in range(future_steps):
            pred = self.regressor(current_step)  # [batch_size, 1]
            predictions.append(pred)

            # Inject prediction into dynamic features at appropriate index (assumed index 0)
            pred_feat = torch.zeros((1, batch_size, self.input_dim), device=pred.device)
            pred_feat[0, :, 0] = pred.squeeze(-1)  # inject predicted value

            building_embed_step = building_embed.unsqueeze(0)  # [1, batch_size, embed_dim]
            static_feats_step = static_feats.unsqueeze(0)      # [1, batch_size, static_feat_dim]

            x_step_combined = torch.cat([pred_feat, building_embed_step, static_feats_step], dim=-1)  # [1, batch_size, input_dim + emb + static]
            x_step_proj = self.input_projection(x_step_combined)  # [1, batch_size, d_model]
            x_step_encoded = self.positional_encoder(x_step_proj)
            current_step = self.encoder(x_step_encoded)[-1]  # [batch_size, d_model]

        return torch.stack(predictions, dim=0).squeeze(-1)  # [future_steps, batch_size]
