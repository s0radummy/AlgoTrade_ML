import pytest
import torch
from src.models.tft_model import TemporalFusionTransformer

def test_tft_model_initialization():
    """Test TFT model initialization."""
    model = TemporalFusionTransformer()
    assert model is not None
    assert model.hidden_dim == 64

def test_tft_forward_pass():
    """Test TFT model forward pass."""
    model = TemporalFusionTransformer()
    model.eval()

    batch_size = 2
    static_cov = torch.randn(batch_size, 32)
    past_inputs = torch.randn(batch_size, 60, 7)
    future_inputs = torch.randn(batch_size, 5, 4)

    with torch.no_grad():
        output = model(static_cov, past_inputs, future_inputs)

    assert output.shape == (batch_size, 5, 5)  # (batch, future_steps, quantiles)
    assert not torch.isnan(output).any()

def test_tft_device():
    """Test TFT model on different devices."""
    model = TemporalFusionTransformer()

    # Test on CPU
    model = model.cpu()
    static_cov = torch.randn(1, 32)
    past_inputs = torch.randn(1, 60, 7)
    future_inputs = torch.randn(1, 5, 4)

    with torch.no_grad():
        output = model(static_cov, past_inputs, future_inputs)

    assert output.shape == (1, 5, 5)
