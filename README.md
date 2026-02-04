# Ministral 3 Fine-Tuning Framework

A comprehensive fine-tuning framework for Mistral AI's Ministral 3 3B model family, featuring GDPO (Group reward-Decoupled Policy Optimization), mHC (Manifold-Constrained Hyper-Connections) architecture, and Chain-of-Thought (CoT) training capabilities.

## Ministral 3 Model Family

| Model | Type | Precision | CoT Support | Context | Use Case |
|-------|------|-----------|-------------|---------|----------|
| Ministral-3-3B-Base-2512 | Base | BF16 | No | 256K | Pre-training base |
| Ministral-3-3B-Instruct-2512 | Instruct | FP8 | No (requires fine-tuning) | 256K | General chat, instructions |
| Ministral-3-3B-Reasoning-2512 | Reasoning | BF16 | Yes (`[THINK]`/`[/THINK]`) | 256K | Math, coding, STEM |

### Key Notes

- **Instruct** version does NOT have built-in Chain-of-Thought capability. This project enables training it to use `[THINK]` tokens for reasoning.
- **Reasoning** version has native CoT support with `reasoning_content` output field.
- FP8 models require dequantization during weight loading (handled automatically).

## Features

- **Custom FP8 Weight Loading**: Automatic dequantization of FP8 weights to BF16
- **Partial Layer Freezing**: Memory-efficient training by freezing early layers
- **GDPO Loss**: Group reward-Decoupled Policy Optimization with Conditioning Easier Reward
- **Heteroscedastic Loss**: Monte Carlo sampling-based uncertainty loss (`heteroscedastic_cross_entropy`, `heteroscedastic_gdpo`)
- **mHC Architecture**: Manifold-Constrained Hyper-Connections variant for enhanced residual connections
- **KV Cache Inference**: O(1) per-token generation with Key-Value caching for fast long-sequence generation
- **Refusal Mechanism**: Uncertainty-based token regeneration with temperature decay
- **Streaming Inference**: Real-time token-by-token output during generation
- **CSV Training Logs**: Automatic logging of predictions, labels, and loss components
- **Chat Log Viewer**: Browser-based viewer for inference conversation logs

## Project Structure

```
pre-aiffel/
├── architectures/              # Model architecture definitions
│   ├── ministral_3_3b_instruct.py       # Base Ministral 3 implementation
│   └── ministral_3_3b_instruct_mHC.py   # mHC variant
├── data/                       # Training data
│   └── sample.json
├── model/                      # Model weights storage
│   └── ministral_3_3b_instruct/
├── result/                     # Training logs (CSV)
├── inference_log/              # Inference conversation logs (JSON)
├── training.py                 # Main training script
├── inference.py                # Main inference script (with KV cache & refusal mechanism)
├── chat_viewer.py              # Browser-based chat log viewer
├── GDPO.py                     # GDPO loss implementation
├── mHC.py                      # mHC module (generic)
├── loss.py                     # Loss function handlers (incl. heteroscedastic)
├── dataset.py                  # Dataset loader
├── model.py                    # Model loader and argument parser
├── model_loader.py             # Weight loading utilities
├── tokenizer.py                # Tokenizer manager
├── utils.py                    # Utility functions (FileConfig loader)
├── training_logger.py          # CSV logging system
├── environment.py              # Environment setup script
└── pyproject.toml              # Project dependencies
```

## Requirements

- Python >= 3.11
- CUDA-enabled GPU (recommended: 24GB VRAM for full fine-tuning)
- transformers >= 5.0.0
- PyTorch with CUDA support

## Installation

### Option 1: Automatic Setup (Recommended)

```bash
python environment.py
```

This will:
1. Install `uv` if not present
2. Create a virtual environment with Python 3.11
3. Install all dependencies from `pyproject.toml`

### Option 2: Manual Setup

```bash
# Using uv (fast)
uv venv --python 3.11 .venv
uv pip install -e .

# Or using pip
python -m venv .venv
pip install -e .
```

### Activate Virtual Environment

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

## Usage

### Training

Basic training with cross-entropy loss:

```bash
python training.py \
  --model_type ministral_3_3b_instruct \
  --data_path data/sample.json \
  --loss_type cross_entropy \
  --epochs 3 \
  --batch_size 1 \
  --learning_rate 2e-5
```

Training with partial layer freezing (memory efficient):

```bash
python training.py \
  --model_type ministral_3_3b_instruct \
  --data_path data/sample.json \
  --loss_type cross_entropy \
  --epochs 3 \
  --freeze_until_layer 20
```

Training with GDPO loss:

```bash
python training.py \
  --model_type ministral_3_3b_instruct \
  --data_path data/sample.json \
  --loss_type gdpo \
  --epochs 3 \
  --gdpo_group_size 4 \
  --gdpo_kl_coef 0.1
```

Training with heteroscedastic uncertainty loss:

```bash
python training.py \
  --model_type ministral_3_3b_instruct \
  --data_path data/sample.json \
  --loss_type heteroscedastic_cross_entropy \
  --heteroscedastic_T 5 \
  --epochs 3
```

#### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--loss_type` | cross_entropy | Loss function (`cross_entropy`, `gdpo`, `heteroscedastic_cross_entropy`, `heteroscedastic_gdpo`) |
| `--freeze_until_layer` | 0 | Freeze layers 0 to N (0 = no freezing) |
| `--heteroscedastic_T` | 3 | Monte Carlo samples for heteroscedastic loss |
| `--random_seed` | -1 | Random seed (-1 for random) |
| `--debug` | False | Enable debug output |

### Inference

Interactive chat with the model:

```bash
python inference.py --model ministral_3_3b_instruct
```

Chat with a trained model:

```bash
python inference.py --model train/your-trained-model-folder
```

#### Inference Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max_length` | 32768 | Maximum tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--top_k` | 50 | Top-k sampling |
| `--random_seed` | -1 | Random seed (-1 for random) |
| `--no_refusal` | False | Disable refusal mechanism |
| `--refusal_threshold` | 3.0 | Uncertainty threshold for refusal |
| `--refusal_max_retries` | 3 | Max retries per token |
| `--debug` | False | Enable debug output |

#### Refusal Mechanism

The inference engine includes a refusal mechanism that regenerates tokens with high uncertainty:

1. Calculate uncertainty (std of logits) for each generated token
2. If uncertainty > threshold, "refuse" the token
3. Retry with lower temperature and different seed
4. Accept after max retries

This improves generation stability for uncertain predictions.

#### KV Cache

Inference uses KV (Key-Value) caching for efficient long-sequence generation:
- O(1) computation per token (vs O(n) without cache)
- Proper position_ids handling for RoPE
- Automatic attention_mask management

#### Chat Commands

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history |
| `/history` | Show conversation history |
| `/save` | Save conversation to file |
| `/exit` | Save and quit |

### Chat Log Viewer

View saved inference conversations in the browser:

```bash
python chat_viewer.py
```

- Arrow keys (↑/↓) to navigate log files
- Enter to open in browser
- `q` to quit

Logs are stored in `inference_log/` as JSON files.

### mHC Model Training

Train the mHC variant with base weights:

```bash
python training.py \
  --model_type ministral_3_3b_instruct_mHC \
  --base_model_path model/ministral_3_3b_instruct \
  --freeze_until_layer -1 \
  --data_path data/sample.json \
  --loss_type cross_entropy \
  --epochs 5
```

- `--base_model_path`: Load matching weights from the base model
- `--freeze_until_layer -1`: Freeze ALL base layers (only train mHC parameters)

## Data Format

Training data should be in JSON format with `instruction` and `output` fields:

```json
[
    {
        "instruction": "What is 15 + 27?",
        "output": "[THINK]I need to add 15 and 27. 15 + 27 = 42.[/THINK]42"
    },
    {
        "instruction": "What is the capital of France?",
        "output": "[THINK]France is a country in Europe. Its capital city is Paris.[/THINK]Paris"
    }
]
```

The model learns to:
1. Receive user instructions
2. Generate reasoning within `[THINK]...[/THINK]` tags
3. Provide the final answer after the closing tag

## Special Tokens

| Token | Purpose |
|-------|---------|
| `[THINK]` | Start of reasoning section |
| `[/THINK]` | End of reasoning section |
| `<SPECIAL_36>` | Alternative answer start marker |
| `<SPECIAL_37>` | Alternative answer end marker |
| `[INST]` | Instruction start (chat template) |
| `[/INST]` | Instruction end (chat template) |

## GDPO Configuration

GDPO (Group reward-Decoupled Policy Optimization) supports multiple reward objectives:

| Reward | Description |
|--------|-------------|
| Format Reward | Checks for `[THINK]...[/THINK]` structure |
| Length Penalty | Penalizes excessive output length |
| Accuracy Reward | Checks against ground truth |

### Conditioning Easier Reward

Easier rewards (format, length) are conditioned on harder rewards (accuracy):

```bash
--gdpo_use_conditioned_rewards True \
--gdpo_condition_threshold 0.5
```

### Reward Weighting

Individual reward weights can be configured:

```bash
--gdpo_reward_weights 1.0,0.5,0.3
```

## Heteroscedastic Loss

Heteroscedastic uncertainty loss incorporates prediction uncertainty into training using Monte Carlo sampling:

$$\mathcal{L} = -\sum_{i} \log \frac{1}{T} \sum_{t=1}^{T} \text{softmax}(\hat{x}_{i,t})_c$$

Where:
- $\hat{x}_{i,t} = f_i + \sigma_i \cdot \epsilon_t$
- $f_i$ = logits at position $i$
- $\sigma_i$ = standard deviation of logits
- $\epsilon_t \sim \mathcal{N}(0, 1)$ (vocab-sized noise)
- $c$ = ground truth token

### Loss Types

| Loss Type | Description |
|-----------|-------------|
| `heteroscedastic_cross_entropy` | Cross-entropy with uncertainty sampling |
| `heteroscedastic_gdpo` | GDPO with uncertainty-aware log probabilities |

### Configuration

```bash
--loss_type heteroscedastic_cross_entropy \
--heteroscedastic_T 5  # Monte Carlo samples (default: 3)
```

## Training Logs

Training metrics are automatically saved to `result/` folder with naming format:

```
{model_type}-{freeze_layer}-{params}-{epochs}ep-{save_info}-{date}-{time}.csv
```

CSV columns include:
- `step`: Global training step
- `epoch`: Current epoch (1-based integer)
- `total_loss`: Combined loss value
- `predict`: Model prediction (decoded)
- `label`: Ground truth (decoded)
- Loss components (e.g., `cross_entropy`, `format_reward`, etc.)

## Memory Requirements

| Configuration | Approximate VRAM |
|---------------|------------------|
| Full fine-tune (BF16) | ~20GB |
| Freeze 20 layers | ~14GB |
| Freeze 24 layers | ~10GB |
| Inference only | ~7GB |

## References

- **Ministral 3 Paper**: [arXiv:2601.08584](https://arxiv.org/abs/2601.08584)
- **mHC Paper**: [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)
- **HuggingFace Models**: 
  - [Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)
  - [Ministral-3-3B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512)

## License

This project is for educational and research purposes. Model weights are subject to Mistral AI's Apache 2.0 license.
