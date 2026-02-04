import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from training_logger import LossResult
from architectures.ministral_3_3b_instruct import Ministral3TokenConfig

# Import will be done lazily to avoid circular import
# from loss import compute_heteroscedastic_log_probs


def condition_reward(easy_reward, hard_reward, threshold=1.0):
    """
    Condition easier reward on harder reward (Paper Eq. 8).
    
    The easier reward is only given if the harder reward meets or exceeds the threshold.
    This forces the model to prioritize the harder objective first.
    
    Args:
        easy_reward: The easier reward value (e.g., format, length)
        hard_reward: The harder reward value (e.g., accuracy)
        threshold: Minimum hard_reward value required to receive easy_reward
        
    Returns:
        easy_reward if hard_reward >= threshold, else 0.0
    """
    if hard_reward >= threshold:
        return easy_reward
    return 0.0


def compute_rewards(sequences, tokenizer, references=None, reward_config=None, 
                    token_config=None, num_objectives=3):
    """
    Computes rewards based on GDPO paper objectives:
    1. Format Reward: Checks for [THINK]...[/THINK]<SPECIAL_36>...<SPECIAL_37> structure.
    2. Length Penalty: Penalizes excessive length (Length Explosion).
    3. Accuracy Reward: Checks against ground truth sequences (if provided).
    
    Args:
        sequences: Generated token sequences
        tokenizer: Tokenizer for decoding
        references: Ground truth references for accuracy check
        reward_config: Configuration dict with keys:
            - use_conditioned_rewards (bool): Whether to condition easier rewards on accuracy
            - condition_threshold (float): Threshold for accuracy to receive other rewards
            - target_length (int): Target length for length penalty calculation
        token_config: Token configuration class (default: Ministral3TokenConfig)
        num_objectives: Number of reward objectives (default 3)
    
    Returns: (Batch * Group, num_objectives)
    """
    batch_size = sequences.shape[0]
    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    
    rewards = torch.zeros(batch_size, num_objectives)
    
    # Token config 기본값 설정
    if token_config is None:
        token_config = Ministral3TokenConfig
    
    # THINK 토큰 필수 검증
    if not token_config.THINK_START or not token_config.THINK_END:
        raise ValueError(
            f"GDPO requires THINK tokens. "
            f"THINK_START={token_config.THINK_START}, THINK_END={token_config.THINK_END}"
        )
    
    # Parse reward_config
    if reward_config is None:
        reward_config = {}
    use_conditioned_rewards = reward_config.get("use_conditioned_rewards", False)
    condition_threshold = reward_config.get("condition_threshold", 1.0)
    target_len = reward_config.get("target_length", 1024)
    
    # Format reward용 정규식 패턴 생성
    think_pattern = rf"{re.escape(token_config.THINK_START)}\s*\S+.*?{re.escape(token_config.THINK_END)}"
    
    if token_config.ANSWER_START and token_config.ANSWER_END:
        answer_pattern = rf"\s*{re.escape(token_config.ANSWER_START)}\s*\S+.*?{re.escape(token_config.ANSWER_END)}"
        format_pattern = think_pattern + answer_pattern
    else:
        format_pattern = think_pattern
    
    format_regex = re.compile(format_pattern, re.DOTALL)
    
    for i, text in enumerate(texts):
        # 1. Format Reward
        # Strict adherence to the tag structure
        if format_regex.search(text):
            format_score = 1.0
        else:
            format_score = 0.0
            
        # 2. Length Penalty
        # Paper suggests penalizing very long sequences to prevent "Length Explosion".
        # We can use a soft penalty or a hard cap.
        # Let's use a soft penalty: reward decreases as length exceeds a threshold.
        seq_len = len(sequences[i])
        if seq_len > target_len:
            # Decay reward beyond target length
            length_score = max(0.0, 1.0 - ((seq_len - target_len) / target_len))
        else:
            length_score = 1.0
        
        # 3. Accuracy Reward (Hard reward - computed first as it may condition others)
        # If references (ground truth texts) are provided, we check for containment or exact match.
        # For now, we assume simple containment of the reference answer in the generated <answer> block.
        acc_score = 0.0
        if references and i < len(references):
            ref_text = references[i]  # This might need alignment with Batch*Group
            # In a real scenario, we'd parse the <answer> part and compare.
            # Simplified: Check if reference is inside generation.
            if ref_text in text:
                acc_score = 1.0
        
        # 4. Apply Conditioned Rewards (Paper Eq. 8)
        # Easier rewards (format, length) are conditioned on harder reward (accuracy)
        if use_conditioned_rewards:
            format_score = condition_reward(format_score, acc_score, condition_threshold)
            length_score = condition_reward(length_score, acc_score, condition_threshold)
        
        rewards[i, 0] = format_score
        rewards[i, 1] = length_score
        rewards[i, 2] = acc_score
        
    return rewards

def compute_gdpo_loss(model, inputs, trainer, ref_model=None):
    """
    Implements Group reward-Decoupled Normalization Policy Optimization (GDPO).
    """
    # Use processing_class (new) or tokenizer (deprecated) for backward compatibility
    tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Try to get ref_model from trainer if not provided
    if ref_model is None and hasattr(trainer, "ref_model"):
        ref_model = trainer.ref_model
    
    # If labels present, use them for accuracy checking reference
    references = None
    if "labels" in inputs:
         # We'd need to decode labels to get ground truth text
         # Note: labels usually have -100 for ignored parts.
         valid_labels = inputs["labels"].clone()
         valid_labels[valid_labels == -100] = tokenizer.pad_token_id
         references_raw = tokenizer.batch_decode(valid_labels, skip_special_tokens=True)
         # Assign references for use in reward calculation
         references = references_raw

    B = input_ids.shape[0]
    n = 3 # Number of objectives (Format, Length, Accuracy)
    
    # Get GDPO config from trainer (with defaults for backward compatibility)
    gdpo_config = getattr(trainer, "gdpo_config", {})
    G = gdpo_config.get("group_size", 4)
    max_new_tokens = gdpo_config.get("max_new_tokens", 128)
    kl_coef = gdpo_config.get("kl_coef", 0.01)
    temperature = gdpo_config.get("temperature", 1.0)
    
    # Priority Variation config reward_weights
    reward_weights = gdpo_config.get("reward_weights", {
        "format": 1.0,
        "length": 1.0,
        "accuracy": 1.0
    })
    use_conditioned_rewards = gdpo_config.get("use_conditioned_rewards", False)
    condition_threshold = gdpo_config.get("condition_threshold", 1.0)
    target_length = gdpo_config.get("target_length", 1024)
    
    # Build reward_config for compute_rewards
    reward_config = {
        "use_conditioned_rewards": use_conditioned_rewards,
        "condition_threshold": condition_threshold,
        "target_length": target_length,
    }
    
    # 1. Rollout
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,  # Controls generation diversity
            "top_p": 0.95,  # Slightly higher for more diverse samples
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "num_return_sequences": G, 
            "use_cache": False,
        }
        generated_sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )
    
    # Prepare references for the expanded batch if available
    expanded_refs = None
    if references:
        expanded_refs = []
        for ref in references_raw:
            expanded_refs.extend([ref] * G)

    # 2. Multi-Objective Rewards
    rewards_flat = compute_rewards(
        generated_sequences, tokenizer, 
        references=expanded_refs, 
        reward_config=reward_config,
        token_config=Ministral3TokenConfig,
        num_objectives=n
    ).to(model.device)
    rewards = rewards_flat.view(B, G, n)
    
    # 3. GDPO Advantage Calculation (Decoupled Normalization)
    eps = 1e-8
    
    # Normalize each objective INDEPENDENTLY within the group
    means = rewards.mean(dim=1, keepdim=True) # (B, 1, n)
    stds = rewards.std(dim=1, keepdim=True)   # (B, 1, n)
    advantages_k = (rewards - means) / (stds + eps) # (B, G, n)
    
    # Weighted Sum of Advantages (Paper Eq. 7)
    # A_sum = w1*A1 + w2*A2 + ... + wn*An
    weights_tensor = torch.tensor([
        reward_weights.get("format", 1.0),
        reward_weights.get("length", 1.0),
        reward_weights.get("accuracy", 1.0)
    ], device=advantages_k.device, dtype=advantages_k.dtype)
    advantages_sum = (advantages_k * weights_tensor).sum(dim=2)  # (B, G)
    
    # Batch-wise advantage normalization (Eq. 6 in GDPO paper)
    # This ensures stable numerical range regardless of reward count
    bn_mean = advantages_sum.mean()
    bn_std = advantages_sum.std()
    final_advantages = (advantages_sum - bn_mean) / (bn_std + eps)
    final_advantages_flat = final_advantages.view(-1)
    
    # 4. KL Divergence Penalty (Optional but critical for stability)
    # We need log_probs of generated sequences under current model AND reference model.
    # Current model log_probs will be computed during the forward pass for loss.
    # Reference model log_probs need a separate forward pass if ref_model exists.
    
    # Forward Pass on Generated Sequences (Current Policy)
    # We treat the generated sequences as input for training.
    train_input_ids = generated_sequences
    train_attention_mask = (train_input_ids != tokenizer.pad_token_id).long()
    
    # Labels: Mask prompt.
    labels = train_input_ids.clone()
    # Mask prompt part. Assuming right padding, prompts are at start.
    input_len = input_ids.shape[1]
    labels[:, :input_len] = -100 
    
    # Current Model Forward
    outputs = model(input_ids=train_input_ids, attention_mask=train_attention_mask)
    logits = outputs.logits # [B*G, SeqLen, Vocab]
    
    # Get log_prob of the actual tokens selected
    # Gather: [B*G, SeqLen-1]
    # Shift for next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # We need to gather the log prob of the chosen action
    # shift_labels has -100, we need to handle that.
    
    # Create valid mask
    valid_mask = (shift_labels != -100).float()
    
    # Gather log_p from logits
    # We need to use tokens from input_ids (shifted) as indices
    # shift_input_ids = train_input_ids[..., 1:].contiguous()
    # log_probs_shifted = F.log_softmax(shift_logits, dim=-1)
    # token_log_probs = torch.gather(log_probs_shifted, 2, shift_input_ids.unsqueeze(-1)).squeeze(-1)
    
    # Simpler retrieval using CrossEntropy (which is -log_p)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_nll = token_nll.view(B*G, -1)
    token_log_probs = -token_nll
    
    # Reference Model Forward (KL)
    kl_penalty = 0.0
    if ref_model:
        with torch.no_grad():
            ref_outputs = ref_model(input_ids=train_input_ids, attention_mask=train_attention_mask)
            ref_logits = ref_outputs.logits
            
            # Shift
            ref_shift_logits = ref_logits[..., :-1, :].contiguous()
            
            # Ref Log Probs
            ref_loss_fct = nn.CrossEntropyLoss(reduction='none')
            ref_token_nll = ref_loss_fct(ref_shift_logits.view(-1, ref_shift_logits.size(-1)), shift_labels.view(-1))
            ref_token_nll = ref_token_nll.view(B*G, -1)
            ref_token_log_probs = -ref_token_nll
            
            # KL ~ log(pi) - log(ref) = token_log_probs - ref_token_log_probs
            # Penalize: R = R' - beta * KL
            # Here we apply it to the loss directly or advantage?
            # Standard PPO: Loss = - (Advantage) * log_prob + beta * KL
            # Or usually KL is subtracted from Reward.
            # GDPO paper: Likely subtracts KL from rewards/advantages or adds KL term to loss.
            # We will add KL term to validation/logging but apply penalty to Loss.
            
            kl = token_log_probs - ref_token_log_probs # (B*G, SeqLen-1)
            
            # Apply KL penalty to rewards? 
            # In GRPO/GDPO, usually KL is part of the reward structure before normalization
            # OR added as a separate loss term.
            # Let's add it as a loss term for stability: + beta * KL
            kl_penalty = kl_coef * (kl * valid_mask).sum() / (valid_mask.sum() + eps)

    # 5. Policy Gradient Loss
    # Loss = - (Advantage * log_prob)
    # Advantage is per sequence [B*G].
    # log_prob sum per sequence.
    
    # Sum log_probs per sequence
    seq_log_prob = (token_log_probs * valid_mask).sum(dim=1) 
    
    # Advantage is detached
    # pg_loss = - (final_advantages_flat.detach() * seq_log_prob).mean()
    
    # However, standard PPO uses ratio. Here likely vanilla PG or PPO-clip.
    # GDPO paper uses vanilla PG on the group? 
    # "The objective function of GDPO" -> E [ A * log pi ]
    
    pg_loss = - (final_advantages_flat.detach() * seq_log_prob).mean()
    
    total_loss = pg_loss + kl_penalty
    
    # ===== GDPO Logging =====
    # Log reward statistics per objective
    reward_means = rewards.mean(dim=(0, 1))  # (n,)
    reward_stds = rewards.std(dim=(0, 1))    # (n,)
    
    # Log weights and conditioning settings
    w_fmt = reward_weights.get("format", 1.0)
    w_len = reward_weights.get("length", 1.0)
    w_acc = reward_weights.get("accuracy", 1.0)
    cond_str = f"(conditioned, t={condition_threshold})" if use_conditioned_rewards else ""
    
    print(f"[GDPO] Rewards - Format: {reward_means[0].item():.3f} (±{reward_stds[0].item():.3f}, w={w_fmt}), "
          f"Length: {reward_means[1].item():.3f} (±{reward_stds[1].item():.3f}, w={w_len}), "
          f"Accuracy: {reward_means[2].item():.3f} (±{reward_stds[2].item():.3f}, w={w_acc}) {cond_str}")
    
    # Log advantage statistics
    adv_mean = final_advantages.mean().item()
    adv_std = final_advantages.std().item()
    print(f"[GDPO] Advantages - Mean: {adv_mean:.4f}, Std: {adv_std:.4f}")
    
    # Log KL divergence if computed
    if ref_model and isinstance(kl_penalty, torch.Tensor):
        kl_mean = (kl * valid_mask).sum() / (valid_mask.sum() + eps)
        print(f"[GDPO] KL Divergence - Mean: {kl_mean.item():.4f}, Penalty: {kl_penalty.item():.4f}")
    
    # Log loss components
    print(f"[GDPO] Loss - PG: {pg_loss.item():.4f}, KL Penalty: {kl_penalty if isinstance(kl_penalty, float) else kl_penalty.item():.4f}, Total: {total_loss.item():.4f}")
    
    outputs["gdpo_advantages"] = final_advantages.detach()
    
    # Build LossResult with all components for CSV logging
    kl_penalty_val = kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else kl_penalty
    loss_result = LossResult(
        total_loss=total_loss,
        components={
            "pg_loss": pg_loss.item(),
            "kl_penalty": kl_penalty_val,
            "reward_format": reward_means[0].item(),
            "reward_length": reward_means[1].item(),
            "reward_accuracy": reward_means[2].item(),
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
        },
        outputs=outputs
    )
    
    # Memory cleanup to prevent accumulation across steps
    del generated_sequences, train_input_ids, train_attention_mask
    del logits, shift_logits, shift_labels, token_nll, token_log_probs
    del rewards_flat, rewards, advantages_k, advantages_sum
    if ref_model:
        del ref_logits, ref_shift_logits, ref_token_nll, ref_token_log_probs, kl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return loss_result


def compute_heteroscedastic_gdpo_loss(model, inputs, trainer, ref_model=None):
    """
    GDPO with Heteroscedastic Uncertainty.
    
    Uses Monte Carlo sampling to compute log probabilities with uncertainty,
    applied to both policy and reference model.
    """
    # Lazy import to avoid circular dependency
    from loss import compute_heteroscedastic_log_probs
    
    # Get heteroscedastic T from trainer
    T = getattr(trainer, 'heteroscedastic_T', 10)
    debug = getattr(trainer, 'debug', False)
    
    # Use processing_class (new) or tokenizer (deprecated) for backward compatibility
    tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Try to get ref_model from trainer if not provided
    if ref_model is None and hasattr(trainer, "ref_model"):
        ref_model = trainer.ref_model
    
    # If labels present, use them for accuracy checking reference
    references = None
    if "labels" in inputs:
        valid_labels = inputs["labels"].clone()
        valid_labels[valid_labels == -100] = tokenizer.pad_token_id
        references_raw = tokenizer.batch_decode(valid_labels, skip_special_tokens=True)
        references = references_raw

    B = input_ids.shape[0]
    n = 3  # Number of objectives (Format, Length, Accuracy)
    
    # Get GDPO config from trainer (with defaults for backward compatibility)
    gdpo_config = getattr(trainer, "gdpo_config", {})
    G = gdpo_config.get("group_size", 4)
    max_new_tokens = gdpo_config.get("max_new_tokens", 128)
    kl_coef = gdpo_config.get("kl_coef", 0.01)
    temperature = gdpo_config.get("temperature", 1.0)
    
    reward_weights = gdpo_config.get("reward_weights", {
        "format": 1.0,
        "length": 1.0,
        "accuracy": 1.0
    })
    use_conditioned_rewards = gdpo_config.get("use_conditioned_rewards", True)
    condition_threshold = gdpo_config.get("condition_threshold", 1.0)
    target_length = gdpo_config.get("target_length", 1024)
    
    reward_config = {
        "use_conditioned_rewards": use_conditioned_rewards,
        "condition_threshold": condition_threshold,
        "target_length": target_length,
    }
    
    # 1. Rollout
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "num_return_sequences": G,
            "use_cache": False,
        }
        generated_sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )
    
    # Prepare references for the expanded batch if available
    expanded_refs = None
    if references:
        expanded_refs = []
        for ref in references_raw:
            expanded_refs.extend([ref] * G)

    # 2. Multi-Objective Rewards
    rewards_flat = compute_rewards(
        generated_sequences, tokenizer,
        references=expanded_refs,
        reward_config=reward_config,
        token_config=Ministral3TokenConfig,
        num_objectives=n
    ).to(model.device)
    rewards = rewards_flat.view(B, G, n)
    
    # 3. GDPO Advantage Calculation (Decoupled Normalization)
    eps = 1e-8
    
    means = rewards.mean(dim=1, keepdim=True)
    stds = rewards.std(dim=1, keepdim=True)
    advantages_k = (rewards - means) / (stds + eps)
    
    weights_tensor = torch.tensor([
        reward_weights.get("format", 1.0),
        reward_weights.get("length", 1.0),
        reward_weights.get("accuracy", 1.0)
    ], device=advantages_k.device, dtype=advantages_k.dtype)
    advantages_sum = (advantages_k * weights_tensor).sum(dim=2)
    
    bn_mean = advantages_sum.mean()
    bn_std = advantages_sum.std()
    final_advantages = (advantages_sum - bn_mean) / (bn_std + eps)
    final_advantages_flat = final_advantages.view(-1)
    
    # 4. Forward Pass on Generated Sequences
    train_input_ids = generated_sequences
    train_attention_mask = (train_input_ids != tokenizer.pad_token_id).long()
    
    labels = train_input_ids.clone()
    input_len = input_ids.shape[1]
    labels[:, :input_len] = -100
    
    # Current Model Forward
    outputs = model(input_ids=train_input_ids, attention_mask=train_attention_mask)
    logits = outputs.logits
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    valid_mask = (shift_labels != -100).float()
    
    # ===== 1. Standard log_probs for PG and KL (theoretically correct) =====
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_log_probs = -token_nll.view(B*G, -1)
    
    # ===== 2. Heteroscedastic loss as separate regularization term =====
    hetero_log_probs, sigma_stats = compute_heteroscedastic_log_probs(
        shift_logits, shift_labels, T=T, return_sigma=True
    )
    hetero_ce = -(hetero_log_probs * valid_mask).sum() / (valid_mask.sum() + eps)
    
    # Get heteroscedastic weight from config
    hetero_weight = gdpo_config.get("heteroscedastic_weight", 0.1)
    
    # ===== 3. Reference Model Forward (KL) with standard log_probs =====
    kl_penalty = 0.0
    kl = None
    if ref_model:
        with torch.no_grad():
            ref_outputs = ref_model(input_ids=train_input_ids, attention_mask=train_attention_mask)
            ref_logits = ref_outputs.logits
            ref_shift_logits = ref_logits[..., :-1, :].contiguous()
            
            # Standard log_probs for reference (not heteroscedastic)
            ref_token_nll = loss_fct(ref_shift_logits.view(-1, ref_shift_logits.size(-1)), shift_labels.view(-1))
            ref_token_log_probs = -ref_token_nll.view(B*G, -1)
            
            kl = token_log_probs - ref_token_log_probs
            kl_penalty = kl_coef * (kl * valid_mask).sum() / (valid_mask.sum() + eps)

    # ===== 4. Policy Gradient Loss (standard log_probs) =====
    seq_log_prob = (token_log_probs * valid_mask).sum(dim=1)
    pg_loss = -(final_advantages_flat.detach() * seq_log_prob).mean()
    
    # ===== 5. Total Loss: GDPO + λ * Heteroscedastic =====
    total_loss = pg_loss + kl_penalty + hetero_weight * hetero_ce
    
    # ===== Logging (conditional on debug flag) =====
    reward_means = rewards.mean(dim=(0, 1))
    reward_stds = rewards.std(dim=(0, 1))
    adv_mean = final_advantages.mean().item()
    adv_std = final_advantages.std().item()
    
    if debug:
        w_fmt = reward_weights.get("format", 1.0)
        w_len = reward_weights.get("length", 1.0)
        w_acc = reward_weights.get("accuracy", 1.0)
        cond_str = f"(conditioned, t={condition_threshold})" if use_conditioned_rewards else ""
        
        print(f"[DEBUG] Heteroscedastic GDPO - T={T}, weight={hetero_weight}")
        print(f"[DEBUG] Rewards - Format: {reward_means[0].item():.3f} (±{reward_stds[0].item():.3f}, w={w_fmt}), "
              f"Length: {reward_means[1].item():.3f} (±{reward_stds[1].item():.3f}, w={w_len}), "
              f"Accuracy: {reward_means[2].item():.3f} (±{reward_stds[2].item():.3f}, w={w_acc}) {cond_str}")
        print(f"[DEBUG] Advantages - Mean: {adv_mean:.4f}, Std: {adv_std:.4f}")
        
        if ref_model and isinstance(kl_penalty, torch.Tensor):
            kl_mean = (kl * valid_mask).sum() / (valid_mask.sum() + eps)
            print(f"[DEBUG] KL Divergence - Mean: {kl_mean.item():.4f}, Penalty: {kl_penalty.item():.4f}")
        
        kl_val = kl_penalty if isinstance(kl_penalty, float) else kl_penalty.item()
        print(f"[DEBUG] Loss - PG: {pg_loss.item():.4f}, KL: {kl_val:.4f}, Hetero CE: {hetero_ce.item():.4f}, Total: {total_loss.item():.4f}")
    
    outputs["gdpo_advantages"] = final_advantages.detach()
    
    kl_penalty_val = kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else kl_penalty
    loss_result = LossResult(
        total_loss=total_loss,
        components={
            "pg_loss": pg_loss.item(),
            "kl_penalty": kl_penalty_val,
            "hetero_ce": hetero_ce.item(),
            "hetero_weight": hetero_weight,
            "reward_format": reward_means[0].item(),
            "reward_length": reward_means[1].item(),
            "reward_accuracy": reward_means[2].item(),
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
            "heteroscedastic_T": T,
            "sigma_mean": sigma_stats["sigma_mean"],
            "sigma_std": sigma_stats["sigma_std"],
            "sigma_min": sigma_stats["sigma_min"],
            "sigma_max": sigma_stats["sigma_max"],
        },
        outputs=outputs
    )
    
    # Memory cleanup
    del generated_sequences, train_input_ids, train_attention_mask
    del logits, shift_logits, shift_labels, token_log_probs
    del hetero_log_probs
    del rewards_flat, rewards, advantages_k, advantages_sum
    if ref_model:
        del ref_logits, ref_shift_logits, ref_token_log_probs
        if kl is not None:
            del kl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return loss_result
