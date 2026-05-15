import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
import numpy as np
from src.utils.logger import setup_logger
from src.models.tft_model import TemporalFusionTransformer
from config.settings import settings

logger = setup_logger(__name__)

class QuantileLoss(nn.Module):
    """Quantile loss (pinball loss) for multi-quantile regression."""

    def __init__(self, quantiles: list):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.
        predictions: (batch, num_steps, num_quantiles)
        targets: (batch, num_steps, 1)
        """
        errors = targets - predictions  # (batch, num_steps, num_quantiles)
        losses = torch.where(
            errors >= 0,
            self.quantiles * errors,
            (self.quantiles - 1) * errors
        )
        return losses.mean()

class TFTTrainer:
    """Training pipeline for TFT model."""

    def __init__(
        self,
        model: TemporalFusionTransformer,
        learning_rate: float = 0.001,
        quantiles: list = [0.1, 0.3, 0.5, 0.7, 0.9],
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = QuantileLoss(quantiles)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for static_cov, past_inputs, future_inputs, targets in train_loader:
            static_cov = static_cov.to(self.device)
            past_inputs = past_inputs.to(self.device)
            future_inputs = future_inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(static_cov, past_inputs, future_inputs)
            loss = self.criterion(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for static_cov, past_inputs, future_inputs, targets in val_loader:
                static_cov = static_cov.to(self.device)
                past_inputs = past_inputs.to(self.device)
                future_inputs = future_inputs.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(static_cov, past_inputs, future_inputs)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
    ):
        """Train model with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": val_loss,
                    },
                    f"{settings.model_path}",
                )
                logger.info(f"Best model saved with val_loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break

        return self.train_losses, self.val_losses

def create_dummy_data(
    num_samples: int = 1000,
    batch_size: int = 32,
    static_dim: int = 32,
    past_seq_len: int = 60,
    num_future_steps: int = 5,
    future_dim: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create dummy data for testing/development."""
    # Generate random tensors
    static_covariates = torch.randn(num_samples, static_dim)
    past_inputs = torch.randn(num_samples, past_seq_len, 7)  # 7 features
    future_inputs = torch.randn(num_samples, num_future_steps, future_dim)
    targets = torch.randn(num_samples, num_future_steps, 5)  # 5 quantiles

    dataset = TensorDataset(static_covariates, past_inputs, future_inputs, targets)

    # Split: 80% train, 20% val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
