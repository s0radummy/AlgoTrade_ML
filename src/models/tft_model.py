"""
Temporal Fusion Transformer — proper implementation following:
  Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting"

Key components implemented from scratch:
  GRN  — Gated Residual Network (core building block)
  VSN  — Variable Selection Network
  TFT  — full model with LSTM encoder-decoder + temporal self-attention

Input shapes (match src/data/dataset.py):
  static_covariates : (B, 32)
  past_inputs       : (B, 60, 7)
  future_inputs     : (B,  5, 4)

Output shape:
  (B, 5, num_quantiles)   default 5 quantiles → (B, 5, 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class GRN(nn.Module):
    """
    Gated Residual Network.
    GRN(a, c) = LayerNorm(a + GLU(ELU(W₂·a + W₃·c + b₂)))

    Optional context vector c is added before the ELU activation.
    Skip connection projects input to output_dim when they differ.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: int = None,
    ):
        super().__init__()
        self.fc1         = nn.Linear(input_dim, hidden_dim)
        self.fc2         = nn.Linear(hidden_dim, output_dim * 2)   # *2 for GLU split
        self.dropout     = nn.Dropout(dropout)
        self.norm        = nn.LayerNorm(output_dim)
        self.context_fc  = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.skip        = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else None

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = self.skip(x) if self.skip is not None else x

        h = self.fc1(x)
        if context is not None and self.context_fc is not None:
            c = self.context_fc(context)
            # Broadcast: (B, d) → (B, 1, d) → (B, T, d) as needed
            while c.dim() < h.dim():
                c = c.unsqueeze(-2)
            h = h + c
        h = F.elu(h)

        h = self.dropout(self.fc2(h))
        gate, value = h.chunk(2, dim=-1)
        return self.norm(value * torch.sigmoid(gate) + residual)


class VSN(nn.Module):
    """
    Variable Selection Network.
    Learns soft per-variable importances and returns a weighted combination
    of per-variable GRN outputs.

    Expects input already projected to hidden_dim per variable:
      x : (..., num_vars, hidden_dim)
    """

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: int = None,
    ):
        super().__init__()
        self.num_vars = num_vars

        # One GRN per variable
        self.var_grns = nn.ModuleList([
            GRN(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_vars)
        ])

        # Softmax weight GRN: flattened input → num_vars importance scores
        self.weight_grn = GRN(
            num_vars * hidden_dim,
            hidden_dim,
            num_vars,
            dropout,
            context_dim=context_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          selected : (..., hidden_dim)   weighted combination
          weights  : (..., num_vars)     softmax variable importances
        """
        # Per-variable GRN outputs: (..., num_vars, hidden_dim)
        var_out = torch.stack(
            [self.var_grns[i](x[..., i, :]) for i in range(self.num_vars)],
            dim=-2,
        )

        # Flatten variables for weight GRN: (..., num_vars * hidden_dim)
        flat    = x.flatten(start_dim=-2)
        weights = F.softmax(self.weight_grn(flat, context), dim=-1)  # (..., num_vars)

        # Weighted sum across variable dimension
        selected = (weights.unsqueeze(-1) * var_out).sum(dim=-2)     # (..., hidden_dim)
        return selected, weights


# ── Main model ─────────────────────────────────────────────────────────────────

class TemporalFusionTransformer(nn.Module):
    """
    Full TFT model for multi-step quantile forecasting of stock log-returns.

    Architecture (follows the paper):
      1. Project each scalar variable to hidden_dim
      2. Static covariate encoders → 4 context vectors (cs, ch, cc, ce)
      3. Variable Selection Networks for past and future inputs
      4. LSTM encoder (past) + decoder (future), initialised from static context
      5. Post-LSTM gating + skip connection
      6. Static enrichment (GRN with ce)
      7. Temporal self-attention over full past+future sequence
      8. Post-attention gating + skip connection
      9. Position-wise GRN feed-forward
      10. Pre-output gating + skip connection
      11. Linear output head → quantile forecasts at future positions
    """

    def __init__(
        self,
        static_dim: int      = 32,
        past_dim: int        = 7,
        past_seq_len: int    = 60,
        future_dim: int      = 4,
        num_future_steps: int = 5,
        num_quantiles: int   = 5,
        hidden_dim: int      = 64,
        num_heads: int       = 4,
        num_lstm_layers: int = 1,
        dropout: float       = 0.1,
        # Legacy alias kept for backward compat with old checkpoints
        num_layers: int      = None,
    ):
        super().__init__()

        self.past_seq_len     = past_seq_len
        self.num_future_steps = num_future_steps
        self.num_quantiles    = num_quantiles
        self.hidden_dim       = hidden_dim
        self.num_past_vars    = past_dim
        self.num_future_vars  = future_dim

        d = hidden_dim

        # ── 1. Input projections: scalar → d ─────────────────────────────────
        self.static_proj  = nn.Linear(static_dim, d)
        self.past_projs   = nn.ModuleList([nn.Linear(1, d) for _ in range(past_dim)])
        self.future_projs = nn.ModuleList([nn.Linear(1, d) for _ in range(future_dim)])

        # ── 2. Static covariate encoders → 4 context vectors ─────────────────
        self.static_cs = GRN(d, d, d, dropout)   # variable selection context
        self.static_ch = GRN(d, d, d, dropout)   # LSTM initial hidden state
        self.static_cc = GRN(d, d, d, dropout)   # LSTM initial cell state
        self.static_ce = GRN(d, d, d, dropout)   # static enrichment context

        # ── 3. Variable Selection Networks ────────────────────────────────────
        self.past_vsn   = VSN(past_dim,   d, dropout, context_dim=d)
        self.future_vsn = VSN(future_dim, d, dropout, context_dim=d)

        # ── 4. LSTM encoder / decoder ─────────────────────────────────────────
        lstm_kw = dict(
            input_size=d, hidden_size=d,
            num_layers=num_lstm_layers, batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.lstm_encoder = nn.LSTM(**lstm_kw)
        self.lstm_decoder = nn.LSTM(**lstm_kw)
        self._num_lstm_layers = num_lstm_layers

        # ── 5. Post-LSTM gating ───────────────────────────────────────────────
        self.post_lstm_gate = nn.Linear(d, d * 2)
        self.post_lstm_norm = nn.LayerNorm(d)

        # ── 6. Static enrichment ──────────────────────────────────────────────
        self.static_enrichment = GRN(d, d, d, dropout, context_dim=d)

        # ── 7. Temporal self-attention ────────────────────────────────────────
        self.attention      = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.post_attn_gate = nn.Linear(d, d * 2)
        self.post_attn_norm = nn.LayerNorm(d)

        # ── 9 & 10. Position-wise FF + pre-output gating ──────────────────────
        self.pos_ff        = GRN(d, d, d, dropout)
        self.pre_out_gate  = nn.Linear(d, d * 2)
        self.pre_out_norm  = nn.LayerNorm(d)

        # ── 11. Quantile output head ──────────────────────────────────────────
        self.output_fc = nn.Linear(d, num_quantiles)

        self.dropout = nn.Dropout(dropout)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _glu(x: torch.Tensor) -> torch.Tensor:
        gate, value = x.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)

    def _init_lstm_state(
        self, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand (B, d) context vectors to (num_layers, B, d) LSTM states."""
        h = h.unsqueeze(0).expand(self._num_lstm_layers, -1, -1).contiguous()
        c = c.unsqueeze(0).expand(self._num_lstm_layers, -1, -1).contiguous()
        return h, c

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        static_covariates: torch.Tensor,   # (B, 32)
        past_inputs: torch.Tensor,         # (B, 60, 7)
        future_inputs: torch.Tensor,       # (B,  5, 4)
    ) -> torch.Tensor:                     # (B,  5, num_quantiles)

        T_p = past_inputs.shape[1]

        # ── Static embedding + 4 context vectors ─────────────────────────────
        static_emb = self.static_proj(static_covariates)   # (B, d)
        cs = self.static_cs(static_emb)                    # variable selection
        ch = self.static_ch(static_emb)                    # LSTM h0
        cc = self.static_cc(static_emb)                    # LSTM c0
        ce = self.static_ce(static_emb)                    # enrichment

        # ── Per-variable projections ──────────────────────────────────────────
        # (B, T, V) → (B, T, V, d)
        past_emb = torch.stack(
            [self.past_projs[i](past_inputs[..., i:i+1]) for i in range(self.num_past_vars)],
            dim=2,
        )
        future_emb = torch.stack(
            [self.future_projs[i](future_inputs[..., i:i+1]) for i in range(self.num_future_vars)],
            dim=2,
        )

        # ── Variable selection ────────────────────────────────────────────────
        past_sel,   _ = self.past_vsn(past_emb,   cs)    # (B, 60, d)
        future_sel, _ = self.future_vsn(future_emb, cs)  # (B,  5, d)

        # ── LSTM encoder → decoder ────────────────────────────────────────────
        h0, c0            = self._init_lstm_state(ch, cc)
        enc_out, (hn, cn) = self.lstm_encoder(past_sel,   (h0, c0))   # (B, 60, d)
        dec_out, _        = self.lstm_decoder(future_sel, (hn, cn))   # (B,  5, d)

        lstm_out = torch.cat([enc_out, dec_out], dim=1)                 # (B, 65, d)
        lstm_in  = torch.cat([past_sel, future_sel], dim=1)            # (B, 65, d)

        # ── Post-LSTM gating + skip ───────────────────────────────────────────
        gated = self.post_lstm_norm(
            self._glu(self.post_lstm_gate(lstm_out)) + lstm_in
        )   # (B, 65, d)

        # ── Static enrichment ─────────────────────────────────────────────────
        enriched = self.static_enrichment(gated, ce)    # (B, 65, d)

        # ── Temporal self-attention ───────────────────────────────────────────
        attn_out, _ = self.attention(enriched, enriched, enriched)
        attn_out    = self.dropout(attn_out)

        attended = self.post_attn_norm(
            self._glu(self.post_attn_gate(attn_out)) + enriched
        )   # (B, 65, d)

        # ── Position-wise FF + pre-output gating + skip ───────────────────────
        ff_out = self.pos_ff(attended)
        output = self.pre_out_norm(
            self._glu(self.pre_out_gate(ff_out)) + attended
        )   # (B, 65, d)

        # ── Quantile output at future positions only ──────────────────────────
        return self.output_fc(output[:, T_p:, :])   # (B, 5, num_quantiles)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = TemporalFusionTransformer()
    model.eval()

    B = 4
    static = torch.randn(B, 32)
    past   = torch.randn(B, 60, 7)
    future = torch.randn(B,  5, 4)

    with torch.no_grad():
        out = model(static, past, future)

    print(f"Output shape : {tuple(out.shape)}")          # (4, 5, 5)
    print(f"NaN in output: {out.isnan().any().item()}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {n_params:,}")
