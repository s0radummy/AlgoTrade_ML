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
        pinball = losses.mean()

        # Soft diversity penalty: push P90−P10 spread to stay above 0.05 z-score units.
        # Weight 0.01 keeps this well below the primary pinball signal (~5% at collapse).
        spread = predictions[..., -1] - predictions[..., 0]
        spread_penalty = torch.relu(0.05 - spread).mean()
        return pinball + 0.01 * spread_penalty


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

        # AdamW (decoupled weight decay) instead of Adam+L2: prevents the adaptive
        # amplification feedback loop where small output-head gradients cause
        # effective L2 → large → weights → 0 → p50_std collapse.
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
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

            # clip_grad_norm_ returns the pre-clip total norm.
            # 0.1 matches pytorch-forecasting reference; 1.0 is too high for attention
            # architectures and can trigger attention entropy collapse.
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
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
        self._recent_p50: list[float] = []
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

            # Collapse alarm 1: absolute threshold
            if p50_std < 1e-4:
                print(
                    "  WARNING: p50_std collapsed below 1e-4 — model predicts near-constant values!",
                    flush=True,
                )

            # Collapse alarm 2: monotonic decline over 3+ consecutive epochs fires early,
            # before the absolute threshold, saving hours of wasted training.
            self._recent_p50.append(p50_std)
            self._recent_p50 = self._recent_p50[-4:]
            if len(self._recent_p50) >= 3 and all(
                self._recent_p50[i] > self._recent_p50[i + 1]
                for i in range(len(self._recent_p50) - 1)
            ):
                print(
                    "  WARNING: p50_std declining for 3+ consecutive epochs — "
                    "collapse in progress. Consider killing this run.",
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
                        "p50_std":              p50_std,
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
        help="Quick 10-stock test to verify the model learns before committing to a full run.",
    )
    parser.add_argument("--epochs",        type=int,   default=25)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--batch_size",    type=int,   default=512)
    parser.add_argument(
        "--no-warmstart", action="store_true",
        help="Ignore any existing checkpoint and train from scratch.",
    )
    args = parser.parse_args()

    DIAG_SYMBOLS = [
        "RELIANCE",   # Energy      — mega-cap
        "HDFCBANK",   # Finance     — mega-cap
        "INFY",       # IT          — large-cap
        "ITC",        # Consumer    — large-cap
        "TATASTEEL",  # Materials   — mid-cap
        "APOLLOHOSP", # Healthcare  — mid-cap
        "BEL",        # Industrials — small-cap
        "BHARTIARTL", # Telecom     — large-cap
        "MARUTI",     # Auto        — large-cap
        "NTPC",       # Utilities   — large-cap
    ]
    symbols    = DIAG_SYMBOLS if args.diagnostic else None
    batch_size = 256        if args.diagnostic else args.batch_size
    epochs     = 10         if args.diagnostic else args.epochs

    if args.diagnostic:
        print("=== DIAGNOSTIC MODE: 10 stocks, 10 epochs ===")
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
    # Skipped if --no-warmstart is set or if the checkpoint has collapsed predictions
    # (p50_std < 1e-3 means the saved weights are useless for inference).
    if os.path.exists(settings.model_path) and not args.no_warmstart:
        try:
            ckpt = torch.load(settings.model_path, map_location="cpu", weights_only=False)
            saved_p50_std = ckpt.get("p50_std", 1.0)
            if saved_p50_std < 1e-3:
                print(
                    f"Checkpoint has collapsed predictions (p50_std={saved_p50_std:.6f}) "
                    f"— skipping warm-start, training from scratch."
                )
            else:
                model.load_state_dict(ckpt["model_state_dict"])
                print(
                    f"Loaded weights from {settings.model_path} "
                    f"(epoch {ckpt['epoch']}, val={ckpt['val_loss']:.6f}, "
                    f"p50_std={saved_p50_std:.6f})"
                )
        except RuntimeError as e:
            print(
                f"Checkpoint incompatible (likely past_dim mismatch after feature expansion) "
                f"— training from scratch.\n  {e}"
            )
    else:
        if args.no_warmstart:
            print("--no-warmstart set — training from scratch.")
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
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=15)
