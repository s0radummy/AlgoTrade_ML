import torch
import torch.nn as nn
from typing import Tuple

class TemporalFusionTransformer(nn.Module):
    """
    PyTorch implementation of Temporal Fusion Transformer for stock price prediction.
    Predicts multi-step quantile forecasts of log returns.
    """

    def __init__(
        self,
        static_dim: int = 32,  # Static covariate dimension
        past_dim: int = 7,     # Past input features (log_return, volume, OHLC)
        past_seq_len: int = 60,  # History window
        future_dim: int = 4,   # Future known inputs (time features)
        num_future_steps: int = 5,  # Prediction horizon
        num_quantiles: int = 5,  # Output quantiles (P10, P30, P50, P70, P90)
        hidden_dim: int = 64,  # Hidden dimension
        num_heads: int = 4,    # Attention heads
        num_layers: int = 2,   # Transformer layers
        dropout: float = 0.1,
    ):
        super().__init__()

        self.static_dim = static_dim
        self.past_dim = past_dim
        self.past_seq_len = past_seq_len
        self.future_dim = future_dim
        self.num_future_steps = num_future_steps
        self.num_quantiles = num_quantiles
        self.hidden_dim = hidden_dim

        # Static covariate processing
        self.static_embedding = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Past input projection
        self.past_linear = nn.Linear(past_dim, hidden_dim)

        # Future input projection
        self.future_linear = nn.Linear(future_dim, hidden_dim)

        # Temporal Fusion Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Temporal Fusion Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Multi-step quantile regression head
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_quantiles),
        )

        # Output projection
        self.output_linear = nn.Linear(
            hidden_dim, num_future_steps * num_quantiles
        )

    def forward(
        self,
        static_covariates: torch.Tensor,  # (batch, static_dim)
        past_inputs: torch.Tensor,        # (batch, past_seq_len, past_dim)
        future_inputs: torch.Tensor,      # (batch, num_future_steps, future_dim)
    ) -> torch.Tensor:
        """
        Forward pass of TFT model.

        Args:
            static_covariates: Static features (stock_id_emb, sector_emb, market_cap_norm)
            past_inputs: Historical sequence of past variables
            future_inputs: Known future features (time, day of week, etc.)

        Returns:
            Quantile forecasts: (batch, num_future_steps, num_quantiles)
        """
        batch_size = past_inputs.shape[0]

        # Process static covariates
        static_embed = self.static_embedding(static_covariates)  # (batch, hidden_dim)

        # Process past inputs
        past_embed = self.past_linear(past_inputs)  # (batch, past_seq_len, hidden_dim)

        # Add static embedding to past sequence
        static_expanded = static_embed.unsqueeze(1).expand(
            batch_size, self.past_seq_len, self.hidden_dim
        )
        past_combined = past_embed + static_expanded

        # Transformer encoder on past sequence
        encoder_output = self.transformer_encoder(past_combined)

        # Process future inputs
        future_embed = self.future_linear(future_inputs)  # (batch, num_future_steps, hidden_dim)

        # Add static embedding to future sequence
        static_expanded_future = static_embed.unsqueeze(1).expand(
            batch_size, self.num_future_steps, self.hidden_dim
        )
        future_combined = future_embed + static_expanded_future

        # Transformer decoder for future predictions
        decoder_output = self.transformer_decoder(
            future_combined, encoder_output
        )  # (batch, num_future_steps, hidden_dim)

        # Project to quantiles
        output = self.output_linear(decoder_output)  # (batch, num_future_steps, num_quantiles)
        output = output.reshape(
            batch_size, self.num_future_steps, self.num_quantiles
        )

        return output
