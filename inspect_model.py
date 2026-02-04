
import os
import torch
import model as model_module
from tokenizer import TokenizerManager
from architectures.ministral_3_3b_instruct import Mistral3Config, Mistral3ForConditionalGeneration

import argparse

def inspect_system(model_type):
    print("[DEBUG] " + "="*50)
    print(f"[DEBUG]  INSPECTING MODEL ARCHITECTURE & TOKENIZER: {model_type}")
    print("[DEBUG] " + "="*50)

    # 1. Load Tokenizer
    print("\n[DEBUG] [1] Loading Tokenizer...")
    tok_manager = TokenizerManager(model_type=model_type, tokenizer_base_dir="model")
    tokenizer = tok_manager.load_tokenizer()
    
    print(f"[DEBUG] Tokenizer Class: {type(tokenizer)}")
    print(f"[DEBUG] Chat Template Present: {tokenizer.chat_template is not None}")
    if tokenizer.chat_template:
        print(f"[DEBUG] Template Snippet: {tokenizer.chat_template[:100]}...")

    # Test "System Prompt" application via apply_chat_template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("\n[DEBUG] [Test] Applying Chat Template with System Prompt:")
        print("[DEBUG] " + "-" * 30)
        print(f"[DEBUG] {formatted}")
        print("[DEBUG] " + "-" * 30)
    except Exception as e:
        print(f"\n[DEBUG] [Test] Failed to apply chat template: {e}")

    # 2. Inspect Model Structure
    print(f"\n[DEBUG] [2] Loading Model Structure (Config Only) for {model_type}...")
    
    # Dynamic Import to support different model types if needed
    import importlib
    import inspect
    from transformers import PretrainedConfig, PreTrainedModel
    try:
        mod = importlib.import_module(f"architectures.{model_type}")
    except ImportError:
        print(f"[DEBUG] Error: Could not import module 'architectures.{model_type}'")
        return

    config_class = None
    model_class = None
    
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, PretrainedConfig) and obj is not PretrainedConfig:
             if "Config" in name and "Mistral" in name: config_class = obj
        if issubclass(obj, PreTrainedModel) and obj is not PreTrainedModel:
             if "ForConditionalGeneration" in name: model_class = obj

    if not config_class or not model_class:
        print("[DEBUG] Could not automatically find Config or Model class in module.")
        return

    config = config_class()
    # Init model on CPU/Meta device to save memory
    with torch.device("meta"):
        model = model_class(config)
    
    print("\n[DEBUG] [3] Model Layer Hierarchy:")
    print("[DEBUG] " + "-" * 30)
    print(f"[DEBUG] {model}")
    print("[DEBUG] " + "-" * 30)

    # Count Layers
    try:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
             layers = model.model.layers
             num_layers = len(layers)
        elif hasattr(model, "layers"):
             layers = model.layers
             num_layers = len(layers)
        else:
             layers = []
             num_layers = 0
        print(f"\n[DEBUG] Verified Number of Decoder Layers: {num_layers}")
        
        if num_layers > 0:
            print("\n[DEBUG] [4] Partial Fine-Tuning Memory Calculator (Estimated)")
            print("[DEBUG] " + "-" * 80)
            print(f"[DEBUG] {'Freeze Until':<15} | {'Trainable Params':<18} | {'Opt State (FP32)':<16} | {'Total Training VRAM (Appx)':<25}")
            print("[DEBUG] " + "-" * 80)
            
            # Constants
            BYTES_PER_PARAM_BF16 = 2
            BYTES_PER_PARAM_FP32 = 4
            OPTIMIZER_STATES = 2 # Momentum + Variance
            BYTES_PER_OPT_STATE = 8 # 2 * 4 bytes
            
            # Base Static Memory (Model Weights + Gradients for Trainable)
            # Note: We assume full model loaded in BF16 (7GB)
            # Gradients only exist for trainable params (BF16 = 2 bytes)
            
            total_params = sum(p.numel() for p in model.parameters())
            
            # Embeddings often at start/end. We'll simplify: 
            # Accumulate params layer by layer.
            
            # 1. Embeddings (Start)
            embed_params = sum(p.numel() for n, p in model.named_parameters() if "embed" in n)
            
            # 2. Per Layer params
            layer_params = sum(p.numel() for p in layers[0].parameters())
            
            # 3. Head params (End - often shared but we count if separate)
            head_params = sum(p.numel() for n, p in model.named_parameters() if "head" in n)
            
            cumulative_frozen = embed_params # Start assuming embeddings frozen
            
            # Baseline: Full Finetune
            trainable_full = total_params
            opt_mem_full = trainable_full * BYTES_PER_OPT_STATE / 1024**3
            grad_mem_full = trainable_full * BYTES_PER_PARAM_BF16 / 1024**3
            model_mem = total_params * BYTES_PER_PARAM_BF16 / 1024**3
            
            # GDPO Overhead (Ref Model)
            gdpo_ref_mem = model_mem # Frozen copy
            
            total_full = model_mem + gdpo_ref_mem + grad_mem_full + opt_mem_full
            print(f"[DEBUG] {'None (Full)':<15} | {trainable_full/1e9:.2f} B{'':<11} | {opt_mem_full:.2f} GB{'':<9} | {total_full:.2f} GB (OOM)")
            
            # Loop layers
            for i in range(1, num_layers + 1):
                # Freezing up to layer i (means 0 to i-1 are frozen)
                # Frozen = Embeddings + (i * Layer Params)
                current_frozen = embed_params + (i * layer_params)
                current_trainable = total_params - current_frozen
                
                # Optimizer Mem (only for trainable)
                opt_mem = current_trainable * BYTES_PER_OPT_STATE / 1024**3
                
                # Gradient Mem (only for trainable)
                grad_mem = current_trainable * BYTES_PER_PARAM_BF16 / 1024**3
                
                # Total = Fixed Model + Fixed Ref + Gradients + Opt
                total_est = model_mem + gdpo_ref_mem + grad_mem + opt_mem
                
                # Print every 4 layers or last
                if i % 4 == 0 or i == num_layers or i == num_layers - 2:
                     print(f"[DEBUG] Layer {i:<9} | {current_trainable/1e9:.2f} B{'':<11} | {opt_mem:.2f} GB{'':<9} | {total_est:.2f} GB")

    except Exception as e:
        print(f"[DEBUG] Error calculating memory: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Model Architecture")
    parser.add_argument("--model_type", type=str, default="ministral_3_3b_instruct", help="Model type string (filename)")
    args = parser.parse_args()
    
    inspect_system(args.model_type)
