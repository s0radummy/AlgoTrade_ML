import argparse
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

LOG_EVERY = 500


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
        learning_rate: float = 3e-4,
        quantiles: list = [0.1, 0.3, 0.5, 0.7, 0.9],
        max_epochs: int = 50,
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = QuantileLoss(quantiles).to(self.device)

        # CosineAnnealingLR prevents the scheduler from being blind to epoch-1 collapse
        # (ReduceLROnPlateau with patience=5 never triggered before epoch-1 collapse)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=1e-6
        )

        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self.target_stats: dict = {}   # populated before fit() by the entry point

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

            # clip_grad_norm_ returns the pre-clip total norm
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if i % LOG_EVERY == 0:
                elapsed    = time.time() - t0
                batches    = len(train_loader)
                eta_sec    = elapsed / i * (batches - i)
                avg_so_far = total_loss / i
                print(
                    f"  epoch {epoch}  [{i:>6}/{batches}]  "
                    f"loss={avg_so_far:.6f}  "
                    f"grad_norm={grad_norm:.4f}  "
                    f"eta={eta_sec/60:.1f}min",
                    flush=True,
                )

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> tuple[float, float]:
        """Returns (avg_val_loss, p50_std). p50_std tracks prediction variance collapse."""
        self.model.eval()
        total_loss = 0.0
        all_p50: list[torch.Tensor] = []

        with torch.no_grad():
            for static_cov, past_inputs, future_inputs, targets in val_loader:
                static_cov    = static_cov.to(self.device)
                past_inputs   = past_inputs.to(self.device)
                future_inputs = future_inputs.to(self.device)
                targets       = targets.to(self.device)

                predictions = self.model(static_cov, past_inputs, future_inputs)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
                all_p50.append(predictions[:, :, 2].cpu())  # P50 index = 2

        avg_loss = total_loss / len(val_loader)
        p50_std  = torch.cat(all_p50).std().item()

        self.val_losses.append(avg_loss)
        return avg_loss, p50_std

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int = 50,
        patience:     int = 10,
    ):
        """Train with early stopping, cosine LR, variance monitoring, and loss curve CSV."""
        os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)

        # Loss curve CSV — primary diagnostic: shows whether collapse happens at epoch 1
        loss_log_path = os.path.splitext(settings.model_path)[0] + "_loss_curve.csv"
        if not os.path.exists(loss_log_path):
            with open(loss_log_path, "w", newline="") as f:
                f.write("epoch,train_loss,val_loss,p50_std,lr\n")

        best_val_loss    = float("inf")
        patience_counter = 0

        print(f"Device: {self.device}")
        print(f"Train batches/epoch: {len(train_loader):,}   Val batches: {len(val_loader):,}")
        print(f"Model path: {settings.model_path}\n")

        for epoch in range(1, epochs + 1):
            t_epoch    = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, p50_std = self.validate(val_loader)
            epoch_min  = (time.time() - t_epoch) / 60

            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"train={train_loss:.6f}  val={val_loss:.6f}  "
                f"p50_std={p50_std:.6f}  "
                f"({epoch_min:.1f}min)",
                flush=True,
            )

            # Collapse alarm: p50_std << 1 on z-scored targets means model predicts near-constant
            if p50_std < 1e-4:
                print(
                    "  WARNING: P50 prediction std collapsed — model predicts near-constant values! "
                    "Check target standardization and learning rate.",
                    flush=True,
                )

            logger.info(
                "epoch_end",
                extra=dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, p50_std=p50_std),
            )

            # Append to loss curve CSV
            current_lr = self.optimizer.param_groups[0]["lr"]
            with open(loss_log_path, "a", newline="") as f:
                f.write(f"{epoch},{train_loss:.8f},{val_loss:.8f},{p50_std:.8f},{current_lr:.8f}\n")

            self.scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "epoch":                epoch,
                        "model_state_dict":     self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss":             val_loss,
                        "target_stats":         self.target_stats,
                    },
                    settings.model_path,
                )
                print(f"  -> best model saved  (val={val_loss:.6f}  p50_std={p50_std:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch} epochs (no improvement for {patience}).")
                    break

        return self.train_losses, self.val_losses


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.dataset import create_dataloaders

    parser = argparse.ArgumentParser(description="Train TFT model on Nifty 50 historical data.")
    parser.add_argument(
        "--diagnostic", action="store_true",
        help="Quick 3-stock test (RELIANCE, INFY, HDFCBANK) to verify the model learns.",
    )
    parser.add_argument("--epochs",     type=int,   default=25)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int,   default=512)
    args = parser.parse_args()

    DIAG_SYMBOLS = ["RELIANCE", "INFY", "HDFCBANK"]  # Energy / IT / Finance — diverse
    symbols    = DIAG_SYMBOLS if args.diagnostic else None
    batch_size = 256        if args.diagnostic else args.batch_size
    epochs     = 10         if args.diagnostic else args.epochs

    if args.diagnostic:
        print("=== DIAGNOSTIC MODE: 3 stocks, 10 epochs ===")
        print(f"Stocks: {DIAG_SYMBOLS}\n")

    print("Loading dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir="data/historical",
        batch_size=batch_size,
        num_workers=0,
        symbols=symbols,
    )

    model = TemporalFusionTransformer()

    # Warm-start from existing checkpoint when available.
    # A RuntimeError here means the checkpoint was built with a different past_dim
    # (e.g., old 7-feature checkpoint vs new 10-feature model) — train from scratch.
    if os.path.exists(settings.model_path):
        try:
            ckpt = torch.load(settings.model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded weights from {settings.model_path} "
                  f"(epoch {ckpt['epoch']}, val={ckpt['val_loss']:.6f})")
        except RuntimeError as e:
            print(
                f"Checkpoint incompatible (likely past_dim mismatch after feature expansion) "
                f"— training from scratch.\n  {e}"
            )
    else:
        print("No checkpoint found — training from scratch.")

    # Collect per-stock target normalization stats so inference can denormalize predictions
    target_stats = {
        s["symbol"]: {"mean": float(s["target_mean"]), "std": float(s["target_std"])}
        for s in train_loader.dataset._stocks
    }
    print(f"Collected target_stats for {len(target_stats)} stocks.\n")

    trainer = TFTTrainer(model, learning_rate=args.lr, max_epochs=epochs)
    trainer.target_stats = target_stats
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=5)
