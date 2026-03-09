import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Encoder-Decoder forecaster:
    - encodes past window -> latent z
    - decodes a fixed-length future horizon
    """
    def __init__(self, n_features: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)

        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, n_features, batch_first=True)

    def forward(self, x_past: torch.Tensor, pred_horizon: int) -> torch.Tensor:
        # x_past: [B, IW, F]
        enc_out, _ = self.encoder(x_past)
        h_last = enc_out[:, -1, :]            # [B, H]
        z = self.to_latent(h_last)            # [B, Z]

        h0 = self.from_latent(z).unsqueeze(1) # [B, 1, H]
        hrep = h0.repeat(1, pred_horizon, 1)  # [B, PH, H]
        y_pred, _ = self.decoder(hrep)        # [B, PH, F]
        return y_pred