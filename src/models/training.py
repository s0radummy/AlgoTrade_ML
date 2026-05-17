import os
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.logger import setup_logger
from src.models.tft_model import TemporalFusionTransformer
from config.settings import settings

logger = setup_logger(__name__)

LOG_EVERY = 500   # print progress every N batches


class QuantileLoss(nn.Module):
    """Pinball (quantile) loss for multi-quantile regression."""

    def __init__(self, quantiles: list):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        predictions : (B, steps, num_quantiles)
        targets     : (B, steps, 1)
        """
        errors = targets - predictions
        losses = torch.where(
            errors >= 0,
            self.quantiles * errors,
            (self.quantiles - 1) * errors,
        )
        return losses.mean()


class TFTTrainer:
    """Training pipeline for the TFT model."""

    def __init__(
        self,
        model: TemporalFusionTransformer,
        learning_rate: float = 1e-3,
        quantiles: list = [0.1, 0.3, 0.5, 0.7, 0.9],
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = QuantileLoss(quantiles).to(self.device)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        t0 = time.time()

        for i, (static_cov, past_inputs, future_inputs, targets) in enumerate(train_loader, 1):
            static_cov    = static_cov.to(self.device)
            past_inputs   = past_inputs.to(self.device)
            future_inputs = future_inputs.to(self.device)
            targets       = targets.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(static_cov, past_inputs, future_inputs)
            loss = self.criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if i % LOG_EVERY == 0:
                elapsed  = time.time() - t0
                batches  = len(train_loader)
                eta_sec  = elapsed / i * (batches - i)
                avg_so_far = total_loss / i
                print(
                    f"  epoch {epoch}  [{i:>6}/{batches}]  "
                    f"loss={avg_so_far:.6f}  "
                    f"eta={eta_sec/60:.1f}min",
                    flush=True,
                )

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for static_cov, past_inputs, future_inputs, targets in val_loader:
                static_cov    = static_cov.to(self.device)
                past_inputs   = past_inputs.to(self.device)
                future_inputs = future_inputs.to(self.device)
                targets       = targets.to(self.device)

                predictions = self.model(static_cov, past_inputs, future_inputs)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int = 50,
        patience:     int = 10,
    ):
        """Train with early stopping, LR decay, and best-model checkpointing."""
        os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)

        best_val_loss    = float("inf")
        patience_counter = 0

        print(f"Device: {self.device}")
        print(f"Train batches/epoch: {len(train_loader):,}   Val batches: {len(val_loader):,}")
        print(f"Model path: {settings.model_path}\n")

        for epoch in range(1, epochs + 1):
            t_epoch = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss   = self.validate(val_loader)
            epoch_min  = (time.time() - t_epoch) / 60

            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"train={train_loss:.6f}  val={val_loss:.6f}  "
                f"({epoch_min:.1f}min)",
                flush=True,
            )
            logger.info(
                "epoch_end",
                extra=dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss),
            )

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "epoch":                epoch,
                        "model_state_dict":     self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss":             val_loss,
                    },
                    settings.model_path,
                )
                print(f"  → best model saved  (val={val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch} epochs (no improvement for {patience}).")
                    break

        return self.train_losses, self.val_losses


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.dataset import create_dataloaders

    print("Loading dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir="data/historical",
        batch_size=1024,
        num_workers=0,
    )

    model = TemporalFusionTransformer()

    # Warm-start from existing checkpoint if available
    if os.path.exists(settings.model_path):
        ckpt = torch.load(settings.model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded weights from {settings.model_path} (epoch {ckpt['epoch']}, val={ckpt['val_loss']:.6f})")
    else:
        print("No checkpoint found — training from scratch.")

    trainer = TFTTrainer(model, learning_rate=1e-3)
    trainer.fit(train_loader, val_loader, epochs=25, patience=5)
