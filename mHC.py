"""
Manifold-Constrained Hyper-Connections (mHC) Implementation
Based on arXiv:2512.24880 - DeepSeek-AI

mHC extends the residual connection paradigm by:
1. Expanding the residual stream width by factor n
2. Using learnable mappings (H_pre, H_post, H_res) for connection
3. Constraining H_res to doubly stochastic matrices via Sinkhorn-Knopp algorithm
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class mHCConfig:
    """Configuration for mHC connections.
    
    Attributes:
        hidden_size: Model hidden dimension (C)
        n_streams: Number of parallel streams (n), default 4
        sinkhorn_iterations: Iterations for Sinkhorn-Knopp algorithm
        alpha_init: Initial value for gating factors (small for stability)
    """
    hidden_size: int
    n_streams: int = 4
    sinkhorn_iterations: int = 20
    alpha_init: float = 0.01


def sinkhorn_knopp(M: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """
    Project matrix M onto the Birkhoff polytope (doubly stochastic matrices).
    
    A doubly stochastic matrix has:
    - All non-negative entries
    - Row sums = 1
    - Column sums = 1
    
    This ensures:
    - Spectral norm <= 1 (non-expansive, prevents gradient explosion)
    - Compositional closure (product of doubly stochastic matrices is doubly stochastic)
    - Feature mean conservation across layers
    
    Args:
        M: Input matrix [..., n, n]
        iterations: Number of Sinkhorn-Knopp iterations
        
    Returns:
        Doubly stochastic matrix [..., n, n]
    """
    # Make all elements positive via exp
    M = torch.exp(M)
    
    for _ in range(iterations):
        # Row normalization: each row sums to 1
        M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
        # Column normalization: each column sums to 1
        M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
    
    return M


class mHCConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection module.
    
    Implements the mHC formula (Paper Eq. 3):
        x_{l+1} = H^res @ x_l + H^post^T @ F(H^pre @ x_l)
    
    Where:
        - H^pre: Aggregates n streams into 1 for layer input
        - H^post: Distributes layer output to n streams
        - H^res: Mixes information within n streams (doubly stochastic)
    
    The mappings are computed dynamically based on input (Paper Eq. 7, 8).
    """
    
    def __init__(self, config: mHCConfig):
        super().__init__()
        self.config = config
        self.n = config.n_streams
        self.C = config.hidden_size
        self.sinkhorn_iterations = config.sinkhorn_iterations
        
        n, C = self.n, self.C
        
        # ============================================================
        # Dynamic mapping projections (phi in Paper Eq. 7)
        # These project the flattened input to compute dynamic coefficients
        # ============================================================
        self.phi_pre = nn.Parameter(torch.zeros(n * C, n))
        self.phi_post = nn.Parameter(torch.zeros(n * C, n))
        self.phi_res = nn.Parameter(torch.zeros(n * C, n * n))
        
        # ============================================================
        # Gating factors (alpha in Paper Eq. 7)
        # Initialized small (0.01) for training stability
        # ============================================================
        self.alpha_pre = nn.Parameter(torch.tensor(config.alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(config.alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(config.alpha_init))
        
        # ============================================================
        # Static biases (b in Paper Eq. 7)
        # These provide the base mapping when dynamic contribution is small
        # ============================================================
        # Initialize b_pre with uniform weights (1/n) for balanced aggregation
        self.b_pre = nn.Parameter(torch.full((1, n), 1.0 / n))
        # Initialize b_post with ones for full contribution to all streams
        self.b_post = nn.Parameter(torch.ones(1, n))
        # Initialize b_res as identity-like for stability
        self.b_res = nn.Parameter(torch.eye(n))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights with small random values."""
        nn.init.normal_(self.phi_pre, std=0.01)
        nn.init.normal_(self.phi_post, std=0.01)
        nn.init.normal_(self.phi_res, std=0.01)
    
    def _compute_mappings(self, x: torch.Tensor):
        """
        Compute H_pre, H_post, H_res from input hidden states.
        
        Args:
            x: Input tensor [B, S, n, C]
            
        Returns:
            H_pre: [B, S, n] - aggregation weights
            H_post: [B, S, n] - distribution weights
            H_res: [B, S, n, n] - residual mixing matrix (doubly stochastic)
        """
        B, S, n, C = x.shape
        
        # Flatten n-stream to single vector: [B, S, n*C]
        x_flat = x.reshape(B, S, -1)
        
        # RMSNorm on flattened input (Paper Eq. 7)
        # ||x||_2 / sqrt(n*C) normalization
        rms = torch.sqrt((x_flat ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        x_norm = x_flat / rms
        
        # ============================================================
        # Compute H_tilde (unnormalized mappings) - Paper Eq. 7
        # H_tilde = alpha * (x_norm @ phi) + b
        # ============================================================
        
        # [B, S, n*C] @ [n*C, n] -> [B, S, n]
        H_tilde_pre = self.alpha_pre * (x_norm @ self.phi_pre) + self.b_pre
        H_tilde_post = self.alpha_post * (x_norm @ self.phi_post) + self.b_post
        
        # [B, S, n*C] @ [n*C, n*n] -> [B, S, n*n] -> [B, S, n, n]
        H_tilde_res = self.alpha_res * (x_norm @ self.phi_res).view(B, S, n, n) + self.b_res
        
        # ============================================================
        # Manifold projection - Paper Eq. 8
        # H_pre, H_post: sigmoid for non-negativity
        # H_res: Sinkhorn-Knopp for doubly stochastic constraint
        # ============================================================
        H_pre = torch.sigmoid(H_tilde_pre)           # [B, S, n], values in (0, 1)
        H_post = 2 * torch.sigmoid(H_tilde_post)     # [B, S, n], values in (0, 2)
        H_res = sinkhorn_knopp(H_tilde_res, self.sinkhorn_iterations)  # [B, S, n, n]
        
        return H_pre, H_post, H_res
    
    def aggregate_for_layer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate n-stream hidden states into single input for layer function.
        
        Applies H_pre to combine n streams: H_pre @ x
        
        Args:
            x: [B, S, n, C] - n-stream hidden states
            
        Returns:
            [B, S, C] - aggregated hidden states for layer input
        """
        H_pre, _, _ = self._compute_mappings(x)
        
        # H_pre: [B, S, n], x: [B, S, n, C]
        # Weighted sum across streams: sum_i(H_pre_i * x_i)
        # einsum: 'bsi,bsic->bsc'
        aggregated = torch.einsum('bsi,bsic->bsc', H_pre, x)
        
        return aggregated
    
    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        """
        Apply mHC connection.
        
        Implements: x_{l+1} = H^res @ x_l + H^post^T @ layer_output
        
        Args:
            x: [B, S, n, C] - current n-stream hidden states
            layer_output: [B, S, C] - output from layer function F(H_pre @ x)
            
        Returns:
            [B, S, n, C] - updated n-stream hidden states
        """
        _, H_post, H_res = self._compute_mappings(x)
        
        # ============================================================
        # Apply residual mixing: H_res @ x
        # H_res: [B, S, n, n], x: [B, S, n, C]
        # Result: [B, S, n, C]
        # ============================================================
        # einsum: 'bsij,bsjc->bsic' means for each (b,s), matrix multiply [n,n] @ [n,C]
        x_mixed = torch.einsum('bsij,bsjc->bsic', H_res, x)
        
        # ============================================================
        # Distribute layer output to streams: H_post^T @ layer_output
        # H_post: [B, S, n], layer_output: [B, S, C]
        # Result: [B, S, n, C] (broadcast layer_output weighted by H_post)
        # ============================================================
        # einsum: 'bsi,bsc->bsic' means multiply each stream weight with output
        post_contribution = torch.einsum('bsi,bsc->bsic', H_post, layer_output)
        
        # Combine: residual mixing + layer contribution
        return x_mixed + post_contribution


class mHCBlock(nn.Module):
    """
    Convenience wrapper that handles both Attention and MLP residual connections
    with mHC for a single transformer block.
    
    This replaces the standard:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    
    With mHC connections for both sub-layers.
    """
    
    def __init__(self, config: mHCConfig):
        super().__init__()
        self.mhc_attn = mHCConnection(config)
        self.mhc_mlp = mHCConnection(config)
    
    def aggregate_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate n-stream for attention layer input."""
        return self.mhc_attn.aggregate_for_layer(x)
    
    def apply_attention_residual(self, x: torch.Tensor, attn_output: torch.Tensor) -> torch.Tensor:
        """Apply mHC after attention."""
        return self.mhc_attn(x, attn_output)
    
    def aggregate_for_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate n-stream for MLP layer input."""
        return self.mhc_mlp.aggregate_for_layer(x)
    
    def apply_mlp_residual(self, x: torch.Tensor, mlp_output: torch.Tensor) -> torch.Tensor:
        """Apply mHC after MLP."""
        return self.mhc_mlp(x, mlp_output)


def expand_to_streams(x: torch.Tensor, n_streams: int) -> torch.Tensor:
    """
    Expand single-stream hidden states to n-stream format.
    
    Used at the start of the model to initialize the n-stream residual.
    
    Args:
        x: [B, S, C] - single-stream hidden states
        n_streams: number of streams (n)
        
    Returns:
        [B, S, n, C] - n-stream hidden states (replicated)
    """
    # Add stream dimension and replicate
    return x.unsqueeze(2).expand(-1, -1, n_streams, -1).contiguous()


def collapse_from_streams(x: torch.Tensor) -> torch.Tensor:
    """
    Collapse n-stream hidden states back to single-stream.
    
    Used at the end of the model before final layer norm and LM head.
    Simple mean across streams preserves scale.
    
    Args:
        x: [B, S, n, C] - n-stream hidden states
        
    Returns:
        [B, S, C] - collapsed hidden states
    """
    return x.mean(dim=2)
