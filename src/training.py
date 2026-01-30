import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def reconstruction_errors(model: torch.nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """
    Compute per-window reconstruction MSE.

    Returns
    -------
    np.ndarray
        Error per batch element (one scalar per window).
    """
    model.eval()
    errs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=(1, 2))
            errs.append(mse.detach().cpu().numpy())

    return np.concatenate(errs, axis=0) if errs else np.array([], dtype=np.float32)


def train_with_early_stopping(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    patience: int,
    min_delta: float,
    use_early_stopping: bool = True,
) -> dict:
    """
    Train a model with optional early stopping based on validation MSE.

    Returns
    -------
    dict
        Training summary:
        - best_val_mse
        - epochs_ran
    """
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    epochs_ran = 0

    for epoch in range(1, epochs + 1):
        epochs_ran = epoch
        model.train()
        losses = []

        for batch in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)

            # If this happens, it's almost always data related (NaN/Inf).
            if torch.isnan(loss):
                raise RuntimeError("Loss became NaN during training (check data preprocessing).")

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        val_errs = reconstruction_errors(model, val_loader, device)
        val_mse = float(np.mean(val_errs)) if len(val_errs) else float("inf")

        # Tiny "human" log: helps when scanning long runs
        train_loss = float(np.mean(losses)) if len(losses) else float("inf")
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_mse={val_mse:.6f}")

        if use_early_stopping:
            if (best_val - val_mse) > min_delta:
                best_val = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping: no improvement > {min_delta} for {patience} epochs.")
                    break

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_mse": best_val, "epochs_ran": epochs_ran}
