"""
Script to generate and save a dummy TFT model for testing.
"""

import torch
import os
from src.models.tft_model import TemporalFusionTransformer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def generate_model(output_path: str = "models/tft_v1.pth"):
    """Generate and save a dummy model."""
    logger.info("Generating dummy TFT model for testing...")

    # Create models directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize model
    model = TemporalFusionTransformer()

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "version": "v1",
            "created_at": str(__import__('datetime').datetime.utcnow()),
            "note": "Dummy model for testing - NOT trained"
        }
    }

    torch.save(checkpoint, output_path)
    logger.info(f"Model saved to {output_path}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    generate_model()
