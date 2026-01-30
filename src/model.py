import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    Lightweight LSTM autoencoder for multivariate time series reconstruction.

    Encoder:
      - LSTM over [W, F] to produce hidden states
      - Linear projection of the last hidden state to latent vector z

    Decoder:
      - Expand z back to hidden dimension, repeat over time
      - LSTM decoder produces reconstruction [W, F]

    Notes
    -----
    This is intentionally simple. It works surprisingly well as a baseline when
    anomalies are "regime changes" rather than single spikes.
    """
    def __init__(self, n_features: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)

        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, n_features, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, F]
        enc_out, _ = self.encoder(x)
        h_last = enc_out[:, -1, :]          # [B, H]
        z = self.to_latent(h_last)          # [B, Z]

        h0 = self.from_latent(z).unsqueeze(1)   # [B, 1, H]
        hrep = h0.repeat(1, x.size(1), 1)       # [B, W, H]
        dec_out, _ = self.decoder(hrep)         # [B, W, F]
        return dec_out
