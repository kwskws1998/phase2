"""
Heteroscedastic Monte Carlo Sampling Utilities.

This module contains shared functions for computing log probabilities
with heteroscedastic Monte Carlo sampling, used by both loss.py and GDPO.py.

Supports both parallel (default, faster) and sequential (lower memory) modes.
"""

import torch


def compute_heteroscedastic_log_probs(logits, labels, T=3, sequential=False, debug=False, return_sigma=False):
    """
    Compute log probabilities with heteroscedastic Monte Carlo sampling.
    
    Formula: log_prob = log( (1/T) * Σ_t softmax(f + σ*ε_t)[c] )
    
    Where:
        f = logits (model output)
        σ = std(f) across vocab dimension (detached for memory efficiency)
        ε_t ~ N(0, 1)
        c = ground truth token index
    
    Args:
        logits: (batch, seq_len, vocab_size) - raw logits f_i
        labels: (batch, seq_len) - ground truth token ids (may contain -100)
        T: Number of Monte Carlo samples (default: 3)
        sequential: If True, use sequential (memory-efficient) mode. 
                   If False (default), use parallel (faster) mode.
        debug: If True, return debug_info dict along with log_probs
        return_sigma: If True, return sigma statistics (for logging)
    
    Returns:
        log_probs: (batch, seq_len) - Monte Carlo averaged log probabilities
                   (invalid positions have undefined values - caller handles masking)
        sigma_stats: (if return_sigma=True) dict with sigma statistics
        debug_info: (if debug=True) dict with sigma and avg_prob statistics
    """
    if sequential:
        return _compute_heteroscedastic_sequential(logits, labels, T, debug, return_sigma)
    else:
        return _compute_heteroscedastic_parallel(logits, labels, T, debug, return_sigma)


def _compute_heteroscedastic_parallel(logits, labels, T, debug, return_sigma):
    """
    Parallel Monte Carlo sampling - processes all T samples simultaneously.
    Faster but uses T times more memory.
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # σ_i = std(f_i) across vocab dimension
    sigma = logits.std(dim=-1, keepdim=True).detach()  # (batch, seq_len, 1)
    
    # Handle -100 labels for gather (temporary replacement)
    labels_safe = labels.clone()
    labels_safe[labels == -100] = 0
    labels_idx = labels_safe.unsqueeze(-1)  # (batch, seq_len, 1)
    
    # Generate all T noise samples at once: (T, batch, seq_len, vocab_size)
    with torch.no_grad():
        epsilon = torch.randn(T, batch_size, seq_len, vocab_size,
                              device=logits.device, dtype=logits.dtype)
    
    # Expand logits and sigma for broadcasting: (1, batch, seq_len, vocab_size)
    logits_expanded = logits.unsqueeze(0)  # (1, batch, seq_len, vocab_size)
    sigma_expanded = sigma.unsqueeze(0)    # (1, batch, seq_len, 1)
    
    # x_hat = f + σ*ε for all T samples: (T, batch, seq_len, vocab_size)
    x_hat = logits_expanded + sigma_expanded * epsilon
    del epsilon
    
    # Compute softmax probabilities for the correct class
    # logsumexp for numerical stability: (T, batch, seq_len, 1)
    log_sum_exp = torch.logsumexp(x_hat, dim=-1, keepdim=True)
    
    # Expand labels_idx for T samples: (T, batch, seq_len, 1)
    labels_idx_expanded = labels_idx.unsqueeze(0).expand(T, -1, -1, -1)
    x_c = x_hat.gather(dim=-1, index=labels_idx_expanded)  # (T, batch, seq_len, 1)
    del x_hat
    
    # prob_c = softmax(x_hat)_c: (T, batch, seq_len)
    prob_c = torch.exp(x_c - log_sum_exp).squeeze(-1)
    del x_c, log_sum_exp
    
    # Average over T samples: (batch, seq_len)
    avg_prob = prob_c.mean(dim=0)
    del prob_c
    
    # log(avg_prob) with numerical stability
    log_probs = torch.log(avg_prob.clamp(min=1e-10))
    
    # Compute sigma statistics for logging
    sigma_stats = {
        "sigma_mean": sigma.mean().item(),
        "sigma_std": sigma.std().item(),
        "sigma_min": sigma.min().item(),
        "sigma_max": sigma.max().item(),
    }
    
    if debug:
        debug_info = {
            **sigma_stats,
            "avg_prob_mean": log_probs.exp().mean().item(),
            "avg_prob_std": log_probs.exp().std().item(),
        }
        return log_probs, debug_info
    
    if return_sigma:
        return log_probs, sigma_stats
    
    return log_probs


def _compute_heteroscedastic_sequential(logits, labels, T, debug, return_sigma):
    """
    Sequential Monte Carlo sampling - processes one sample at a time.
    Memory-efficient but slower. Uses single-sample gradient estimation.
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # σ_i = std(f_i) across vocab dimension
    sigma = logits.std(dim=-1, keepdim=True).detach()  # (batch, seq_len, 1)
    
    # Handle -100 labels for gather (temporary replacement)
    labels_safe = labels.clone()
    labels_safe[labels == -100] = 0
    labels_idx = labels_safe.unsqueeze(-1)  # (batch, seq_len, 1)
    
    # Monte Carlo sampling: Σ_t softmax(f + σ*ε)[c]
    prob_sum = torch.zeros(batch_size, seq_len, device=logits.device, dtype=logits.dtype)
    
    for t in range(T):
        # Generate noise: ε_t is per-vocab (independent noise for each class)
        with torch.no_grad():
            epsilon_t = torch.randn(batch_size, seq_len, vocab_size,
                                    device=logits.device, dtype=logits.dtype)
        
        # x_hat = f + σ*ε_t
        x_hat = logits + sigma * epsilon_t
        del epsilon_t
        
        # Use logsumexp for memory efficiency
        log_sum_exp = torch.logsumexp(x_hat, dim=-1, keepdim=True)
        x_c = x_hat.gather(dim=-1, index=labels_idx)
        del x_hat
        
        # prob_c = softmax(x_hat)_c
        prob_c = torch.exp(x_c - log_sum_exp).squeeze(-1)
        del x_c, log_sum_exp
        
        # Accumulate - detach all but last sample for memory efficiency
        if t < T - 1:
            prob_sum = prob_sum + prob_c.detach()
        else:
            prob_sum = prob_sum + prob_c  # Last sample keeps gradient
        del prob_c
        
        # Clear CUDA cache periodically
        if t < T - 1 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # (1/T) * Σ_t p_{i,t,c}
    avg_prob = prob_sum / T
    del prob_sum
    
    # log(avg_prob) with numerical stability
    log_probs = torch.log(avg_prob.clamp(min=1e-10))
    
    # Compute sigma statistics for logging
    sigma_stats = {
        "sigma_mean": sigma.mean().item(),
        "sigma_std": sigma.std().item(),
        "sigma_min": sigma.min().item(),
        "sigma_max": sigma.max().item(),
    }
    
    if debug:
        debug_info = {
            **sigma_stats,
            "avg_prob_mean": log_probs.exp().mean().item(),
            "avg_prob_std": log_probs.exp().std().item(),
        }
        return log_probs, debug_info
    
    if return_sigma:
        return log_probs, sigma_stats
    
    return log_probs


def compute_learned_heteroscedastic_log_probs(logits, log_variance, labels, T=3, sequential=False):
    """
    Compute log probabilities with LEARNED heteroscedastic Monte Carlo sampling.
    
    Formula (Kendall & Gal, 2017 Eq. 12):
        x̂_{i,t} = f_i + σ_i * ε_t,  ε_t ~ N(0, I)
        L = log( (1/T) * Σ_t exp(x̂_{i,t,c} - log Σ_{c'} exp(x̂_{i,t,c'})) )
    
    Key difference from compute_heteroscedastic_log_probs:
        - σ is LEARNED from model output (log_variance), not computed from logits.std()
        - σ = exp(log_variance / 2)  where log_variance = s = log(σ²)
    
    Args:
        logits: (batch, seq_len, vocab_size) - raw logits f_i
        log_variance: (batch, seq_len, 1) - learned log(σ²) from model
        labels: (batch, seq_len) - ground truth token ids (may contain -100)
        T: Number of Monte Carlo samples
        sequential: If True, use sequential (memory-efficient) mode.
                   If False (default), use parallel (faster) mode.
    
    Returns:
        log_probs: (batch, seq_len) - Monte Carlo averaged log probabilities
        sigma_stats: dict with sigma statistics for logging
    """
    if sequential:
        return _compute_learned_heteroscedastic_sequential(logits, log_variance, labels, T)
    else:
        return _compute_learned_heteroscedastic_parallel(logits, log_variance, labels, T)


def _compute_learned_heteroscedastic_parallel(logits, log_variance, labels, T):
    """
    Parallel Monte Carlo sampling for learned sigma - processes all T samples simultaneously.
    Faster but uses T times more memory.
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # σ_i = exp(s_i / 2) where s_i = log(σ²_i) is learned from model
    sigma = torch.exp(log_variance / 2)  # (batch, seq_len, 1)
    
    # Handle -100 labels for gather (temporary replacement)
    labels_safe = labels.clone()
    labels_safe[labels == -100] = 0
    labels_idx = labels_safe.unsqueeze(-1)  # (batch, seq_len, 1)
    
    # Generate all T noise samples at once: (T, batch, seq_len, vocab_size)
    with torch.no_grad():
        epsilon = torch.randn(T, batch_size, seq_len, vocab_size,
                              device=logits.device, dtype=logits.dtype)
    
    # Expand logits and sigma for broadcasting
    logits_expanded = logits.unsqueeze(0)  # (1, batch, seq_len, vocab_size)
    sigma_expanded = sigma.unsqueeze(0)    # (1, batch, seq_len, 1)
    
    # x_hat = f + σ*ε for all T samples: (T, batch, seq_len, vocab_size)
    # Gradients flow through BOTH logits AND sigma (log_variance)
    x_hat = logits_expanded + sigma_expanded * epsilon
    del epsilon
    
    # Compute softmax probabilities for the correct class
    log_sum_exp = torch.logsumexp(x_hat, dim=-1, keepdim=True)
    
    # Expand labels_idx for T samples
    labels_idx_expanded = labels_idx.unsqueeze(0).expand(T, -1, -1, -1)
    x_c = x_hat.gather(dim=-1, index=labels_idx_expanded)
    del x_hat
    
    # prob_c = softmax(x_hat)_c: (T, batch, seq_len)
    prob_c = torch.exp(x_c - log_sum_exp).squeeze(-1)
    del x_c, log_sum_exp
    
    # Average over T samples: (batch, seq_len)
    avg_prob = prob_c.mean(dim=0)
    del prob_c
    
    # log(avg_prob) with numerical stability
    log_probs = torch.log(avg_prob.clamp(min=1e-10))
    
    # Compute sigma statistics for logging
    sigma_squeezed = sigma.squeeze(-1)
    variance = torch.exp(log_variance.squeeze(-1))  # σ²
    
    sigma_stats = {
        "sigma_mean": sigma_squeezed.mean().item(),
        "sigma_std": sigma_squeezed.std().item(),
        "sigma_min": sigma_squeezed.min().item(),
        "sigma_max": sigma_squeezed.max().item(),
        "variance_mean": variance.mean().item(),
        "log_variance_mean": log_variance.squeeze(-1).mean().item(),
    }
    
    return log_probs, sigma_stats


def _compute_learned_heteroscedastic_sequential(logits, log_variance, labels, T):
    """
    Sequential Monte Carlo sampling for learned sigma - processes one sample at a time.
    Memory-efficient but slower. Uses single-sample gradient estimation.
    """
    # Ensure consistent dtype from model (prevents float32 OOM with large vocab)
    compute_dtype = logits.dtype  # 모델에서 나온 dtype (bf16/fp16 등)
    logits = logits.to(compute_dtype)
    log_variance = log_variance.to(compute_dtype)

    batch_size, seq_len, vocab_size = logits.shape

    # σ_i = exp(s_i / 2) where s_i = log(σ²_i) is learned from model
    sigma = torch.exp(log_variance / 2)  # (batch, seq_len, 1)

    # Handle -100 labels for gather (temporary replacement)
    labels_safe = labels.clone()
    labels_safe[labels == -100] = 0
    labels_idx = labels_safe.unsqueeze(-1)  # (batch, seq_len, 1)

    # Monte Carlo sampling: Σ_t softmax(f + σ*ε)[c]
    prob_sum = torch.zeros(batch_size, seq_len, device=logits.device, dtype=compute_dtype)

    for t in range(T):
        # Generate noise: ε_t is per-vocab (independent noise for each class)
        with torch.no_grad():
            epsilon_t = torch.randn(batch_size, seq_len, vocab_size,
                                    device=logits.device, dtype=compute_dtype)
        
        # x_hat = f + σ*ε_t
        # Gradients flow through BOTH logits AND sigma (log_variance)
        x_hat = logits + sigma * epsilon_t
        del epsilon_t
        
        # Use logsumexp for memory efficiency
        log_sum_exp = torch.logsumexp(x_hat, dim=-1, keepdim=True)
        x_c = x_hat.gather(dim=-1, index=labels_idx)
        del x_hat
        
        # prob_c = softmax(x_hat)_c
        prob_c = torch.exp(x_c - log_sum_exp).squeeze(-1)
        del x_c, log_sum_exp
        
        # Accumulate - detach all but last sample for memory efficiency
        if t < T - 1:
            prob_sum = prob_sum + prob_c.detach()
        else:
            prob_sum = prob_sum + prob_c  # Last sample keeps gradient
        del prob_c
        
        if t < T - 1 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # (1/T) * Σ_t p_{i,t,c}
    avg_prob = prob_sum / T
    del prob_sum
    
    # log(avg_prob) with numerical stability
    log_probs = torch.log(avg_prob.clamp(min=1e-10))
    
    # Compute sigma statistics for logging
    sigma_squeezed = sigma.squeeze(-1)
    variance = torch.exp(log_variance.squeeze(-1))  # σ²
    
    sigma_stats = {
        "sigma_mean": sigma_squeezed.mean().item(),
        "sigma_std": sigma_squeezed.std().item(),
        "sigma_min": sigma_squeezed.min().item(),
        "sigma_max": sigma_squeezed.max().item(),
        "variance_mean": variance.mean().item(),
        "log_variance_mean": log_variance.squeeze(-1).mean().item(),
    }
    
    return log_probs, sigma_stats
