"""
LTI-stable input injection for recurrent-depth transformers.

Follows the Parcae (arxiv 2604.12946) stability formulation exactly:

    A_continuous = Diag(-exp(log_A))       ← negative diagonal, eigenvalues < 0
    A_discrete   = exp(Δt · A_continuous)   ← ZOH discretization, ρ(A) < 1
    B_discrete   = Δt · B                  ← Euler discretization

The update rule inside the recurrent block:
    h_{t+1} = (A - I)·h_t + B·e + Transformer(h_t)

where:
    h_t     = current hidden state (input to recurrent block)
    e       = encoded input from Prelude (injected every loop to prevent drift)
    A, B    = discretized injection parameters (A has eigenvalues in (0, 1))
    I       = identity matrix (from transformer internal residual connections)

Why (A - I): every decoder layer adds h back via its residual connection
(h = h + attn + ffn), so Transformer(h_t) = h_t + Σ_deltas.  Adding A·h_t on
top would double-count h_t.  Subtracting I from A cancels the implicit
identity matrix inside the residuals, giving the pure Parcae update:
(A - I)·h_t + B·e + (h_t + Σ_deltas) = A·h_t + B·e + Σ_deltas.

ρ(A) < 1 is guaranteed by construction for any learned values of log_A and log_dt.
This is what makes training robust; the spectral radius can never drift ≥ 1.
"""

import torch
import torch.nn as nn


class LTIInjection(nn.Module):
    """
    Parcae stability: ZOH-discretized negative-diagonal A with Euler B.

    The continuous parameters (log_A, log_dt, B_raw) are learned.
    Discretization happens in get_A() / get_B() each forward pass.

    Args:
        dim: Hidden state dimension (1536 for Gemma E2B, 3840 for 12B)
    """

    def __init__(self, dim: int):
        super().__init__()
        # log_A: learned vector -> A_continuous = Diag(-exp(log_A))
        self.log_A = nn.Parameter(torch.zeros(dim))

        # log_dt: learned PER-DIMENSION step size -> A_discrete = exp(dt * A_continuous)
        # Each dimension has its own time constant, allowing the model to learn
        # which channels are more recurrent (small dt) vs feedforward (large dt)
        self.log_dt = nn.Parameter(torch.full((dim,), -3.0))

        # B_raw: learned input injection matrix (full, not constrained)
        # Initialized at 2.0 for strong injection signal from step 0
        self.B_raw = nn.Parameter(torch.randn(dim) * 2.0)

        # C: output projection (removed; unused, 14.7M dead params)

        # prelude_norm: LN on e before injection (Parcae Section 4.1, Eq. 3)
        self.prelude_norm = nn.LayerNorm(dim, eps=1e-6)

    def get_A(self) -> torch.Tensor:
        """
        Compute the discrete diagonal A matrix via ZOH discretization.

        A_continuous = -exp(log_A)              [negative diagonal]
        A_discrete   = exp(dt * A_continuous)    [element-wise, values in (0, 1)]

        All values are strictly in (0, 1), guaranteeing ρ(A) < 1.

        The clamp on log_dt + log_A keeps the product safe in float32:
            dt * A_c = exp(log_dt) * exp(log_A) = exp(log_dt + log_A)
            If log_dt + log_A > ~88, exp() overflows. Clamping to 20
            means A_discrete ≤ exp(-exp(20)) ≈ 0, which is fine; it just
            means that channel decays to zero instantly.
        """
        log_product = (self.log_dt + self.log_A).clamp(-20, 20)
        return torch.exp(-torch.exp(log_product))

    def get_B(self) -> torch.Tensor:
        """
        Compute the discrete B vector via Euler discretization.

        B_discrete = dt * B_raw = exp(log_dt) * B_raw

        Returns 1-D tensor of shape (dim,).
        """
        dt = torch.exp(self.log_dt)
        return dt * self.B_raw

    def forward(
        self,
        h: torch.Tensor,           # (B, T, dim); hidden state BEFORE recurrent block
        e: torch.Tensor,           # (B, T, dim); encoded prelude output, frozen
        transformer_out: torch.Tensor,  # (B, T, dim); output AFTER recurrent block
    ) -> torch.Tensor:
        """
        Compute h_{t+1} = (A - I)·h_t + B·norm(e) + transformer_out.

        transformer_out = h_t + Σ_deltas  (includes residual identity).
        Subtracting I·h_t from A·h_t cancels the implicit identity, giving
        the pure Parcae update: A·h_t + B·norm(e) + Σ_deltas.

        At initialization A≈0.95, so (A-I)≈-0.05 — a tiny dampening rather
        than the 1.95× growth the uncorrected formula would produce.
        """
        norm_dtype = self.prelude_norm.weight.dtype
        e_norm = self.prelude_norm(e.to(norm_dtype)).to(h.dtype)
        A = self.get_A().to(h.dtype)
        B = self.get_B().to(h.dtype)
        # (A - 1) accounts for the identity matrix inside transformer_out's
        # residual path.  Without this, h_t would be counted twice — once
        # by A·h and once by the residual I·h inside transformer_out.
        return (A - 1.0) * h + B * e_norm + transformer_out

    def compute_spectral_radius(self) -> torch.Tensor:
        """Return ρ(A); must be < 1 for stability."""
        with torch.no_grad():
            return self.get_A().abs().max()
