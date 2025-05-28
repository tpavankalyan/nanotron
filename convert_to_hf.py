#!/usr/bin/env python3
"""
Script to convert Nanotron checkpoint to Hugging Face format.

Usage:
    python convert_nanotron_to_hf.py --tp 1 --nanotron-checkpoint-path /path/to/nanotron/checkpoint --hugging-face-checkpoint-path /path/to/output/hf/checkpoint
"""
import argparse
import os
import json
import torch
import safetensors.torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from huggingface_hub import HfApi, HfFolder

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Nanotron Model")
    group.add_argument(
        "--nanotron-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory with a Nanotron Checkpoint",
    )

    group = parser.add_argument_group(title="Nanotron Parallelism")
    group.add_argument("--tp", type=int, required=True, help="Tensor Parallelism Degree of the Nanotron Checkpoint")

    group = parser.add_argument_group(title="HuggingFace Model")
    group.add_argument(
        "--hugging-face-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory to store the converted checkpoint",
    )

    parser.add_argument(
        "--repo-name",
        type=str,
        default=None,
        help="Repository name on Hugging Face Hub to push the model to (e.g., your-username/model-name)",
    )

    args = parser.parse_args()
    return args

def load_nanotron_config(checkpoint_path):
    """Load Nanotron config from checkpoint directory."""
    # Load model config from model_config.json
    with open(os.path.join(checkpoint_path, "model_config.json"), "r") as f:
        model_config = json.load(f)
    
    # Load YAML config if available
    config_yaml_path = os.path.join(checkpoint_path, "config.yaml")
    config_data = {}
    if os.path.exists(config_yaml_path):
        import yaml
        with open(config_yaml_path, "r") as f:
            config_data = yaml.safe_load(f)
    
    return model_config, config_data

def load_safetensors_weights(checkpoint_path):
    """Load all safetensors weights from checkpoint directory."""
    weights = {}
    
    # Walk through the directory structure to find all safetensors files
    for root, _, files in os.walk(os.path.join(checkpoint_path, "model")):
        for file in files:
            if file.endswith(".safetensors"):
                file_path = os.path.join(root, file)
                # Get relative path from checkpoint_path/model
                rel_path = os.path.relpath(file_path, os.path.join(checkpoint_path, "model"))
                # Load the tensor
                tensor_dict = safetensors.torch.load_file(file_path)
                for key, tensor in tensor_dict.items():
                    # Use the relative path and key as the full key
                    full_key = f"{os.path.dirname(rel_path)}/{key}"
                    weights[full_key] = tensor
    
    return weights

def create_hf_config(model_config, config_data):
    """Create a Hugging Face LlamaConfig from Nanotron config."""
    # Extract model config parameters
    if isinstance(model_config, dict) and "model_config" in model_config:
        model_config = model_config["model_config"]
    
    # Try to get values from config_data if available
    if config_data and "model" in config_data and "model_config" in config_data["model"]:
        config_values = config_data["model"]["model_config"]
    else:
        config_values = model_config
    
    # Create LlamaConfig with parameters from Nanotron config
    hf_config = LlamaConfig(
        vocab_size=config_values.get("vocab_size", 32000),
        hidden_size=config_values.get("hidden_size", 4096),
        intermediate_size=config_values.get("intermediate_size", 11008),
        num_hidden_layers=config_values.get("num_hidden_layers", 32),
        num_attention_heads=config_values.get("num_attention_heads", 32),
        num_key_value_heads=config_values.get("num_key_value_heads", 32),
        hidden_act=config_values.get("hidden_act", "silu"),
        max_position_embeddings=config_values.get("max_position_embeddings", 2048),
        initializer_range=config_values.get("initializer_range", 0.02),
        rms_norm_eps=config_values.get("rms_norm_eps", 1e-6),
        use_cache=config_values.get("use_cache", True),
        pad_token_id=config_values.get("pad_token_id", None),
        bos_token_id=config_values.get("bos_token_id", 1),
        eos_token_id=config_values.get("eos_token_id", 2),
        tie_word_embeddings=config_values.get("tie_word_embeddings", False),
        rope_theta=config_values.get("rope_theta", 10000.0),
    )
    
    return hf_config

def map_nanotron_to_hf_weights(nanotron_weights, hf_model):
    """Map Nanotron weights to Hugging Face model weights."""
    # Create a mapping dictionary for weight names
    mapping = {}
    
    # Identify patterns in the Nanotron weights
    for key in nanotron_weights.keys():
        if "token_position_embeddings/pp_block/token_embedding/model_weight" in key:
            mapping[key] = "model.embed_tokens.weight"
        elif "final_layer_norm/pp_block/model_weight" in key:
            mapping[key] = "model.norm.weight"
        elif "lm_head/pp_block/model_weight" in key:
            mapping[key] = "lm_head.weight"
        elif "/decoder/" in key:
            # Extract layer number
            parts = key.split("/")
            for i, part in enumerate(parts):
                if part == "decoder":
                    layer_num = int(parts[i+1])
                    break
            
            # Map different layer components
            if "input_layernorm/model_weight" in key:
                mapping[key] = f"model.layers.{layer_num}.input_layernorm.weight"
            elif "post_attention_layernorm/model_weight" in key:
                mapping[key] = f"model.layers.{layer_num}.post_attention_layernorm.weight"
            elif "attn/qkv_proj/model_weight" in key:
                # This will be handled specially
                mapping[key] = f"model.layers.{layer_num}.self_attn.qkv_proj.weight"
            elif "attn/o_proj/model_weight" in key:
                mapping[key] = f"model.layers.{layer_num}.self_attn.o_proj.weight"
            elif "mlp/gate_up_proj/model_weight" in key:
                # This will be handled specially
                mapping[key] = f"model.layers.{layer_num}.mlp.gate_up_proj.weight"
            elif "mlp/down_proj/model_weight" in key:
                mapping[key] = f"model.layers.{layer_num}.mlp.down_proj.weight"
    
    # Copy weights to HF model
    with torch.no_grad():
        for nanotron_key, hf_key in mapping.items():
            nanotron_weight = nanotron_weights[nanotron_key]
            
            # Handle special cases
            if "qkv_proj" in hf_key:
                layer_num = int(hf_key.split(".")[2])
                # Split QKV into separate Q, K, V
                hidden_size = hf_model.config.hidden_size
                num_heads = hf_model.config.num_attention_heads
                num_kv_heads = hf_model.config.num_key_value_heads
                head_dim = hidden_size // num_heads
                
                q_size = num_heads * head_dim
                k_size = num_kv_heads * head_dim
                v_size = num_kv_heads * head_dim
                
                q, k, v = torch.split(nanotron_weight, [q_size, k_size, v_size])
                
                # Set the weights
                hf_model.model.layers[layer_num].self_attn.q_proj.weight.copy_(q)
                hf_model.model.layers[layer_num].self_attn.k_proj.weight.copy_(k)
                hf_model.model.layers[layer_num].self_attn.v_proj.weight.copy_(v)
            
            elif "gate_up_proj" in hf_key:
                layer_num = int(hf_key.split(".")[2])
                # Split gate_up into separate gate and up
                intermediate_size = hf_model.config.intermediate_size
                
                gate_proj, up_proj = torch.split(nanotron_weight, [intermediate_size, intermediate_size])
                
                # Set the weights
                hf_model.model.layers[layer_num].mlp.gate_proj.weight.copy_(gate_proj)
                hf_model.model.layers[layer_num].mlp.up_proj.weight.copy_(up_proj)
            
            else:
                # Get the target parameter
                param = hf_model
                for attr in hf_key.split("."):
                    param = getattr(param, attr)
                
                # Copy the weight
                param.copy_(nanotron_weight)
    
    return hf_model

def push_model_to_hub(model_path, repo_name, commit_message="Upload converted model", private=False):
    print(f"Pushing model to Hugging Face Hub: {repo_name}")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)

    hf_model.push_to_hub(repo_name, commit_message=commit_message, private=private)
    hf_tokenizer.push_to_hub(repo_name, commit_message=commit_message)
    print(f"Successfully pushed model and tokenizer to https://huggingface.co/{repo_name}")


def main(args):
    print(f"Loading Nanotron checkpoint from: {args.nanotron_checkpoint_path}")
    
    # Load Nanotron config
    model_config, config_data = load_nanotron_config(args.nanotron_checkpoint_path)
    
    # Create HF config
    hf_config = create_hf_config(model_config, config_data)
    
    print(f"Creating HF model with config: {hf_config}")
    
    # Create HF model
    hf_model = AutoModelForCausalLM.from_config(
        config=hf_config,
        torch_dtype=torch.float16,  # Use float16 for compatibility
    )
    
    # Load Nanotron weights
    print("Loading Nanotron weights...")
    nanotron_weights = load_safetensors_weights(args.nanotron_checkpoint_path)
    
    # Map weights to HF model
    print("Mapping weights to HF model...")
    hf_model = map_nanotron_to_hf_weights(nanotron_weights, hf_model)
    
    # Save HF model
    print(f"Saving HF model to: {args.hugging_face_checkpoint_path}")
    os.makedirs(args.hugging_face_checkpoint_path, exist_ok=True)
    hf_model.save_pretrained(args.hugging_face_checkpoint_path)
    
    # Try to save tokenizer if available
    if config_data and "tokenizer" in config_data and "tokenizer_name_or_path" in config_data["tokenizer"]:
        tokenizer_name = config_data["tokenizer"]["tokenizer_name_or_path"]
        try:
            print(f"Loading and saving tokenizer from: {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            tokenizer.save_pretrained(args.hugging_face_checkpoint_path)
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {tokenizer_name}: {e}")
            print("You may need to manually copy the tokenizer files.")
    
    print(f"Successfully converted Nanotron checkpoint to Hugging Face format.")
    print(f"Saved to: {args.hugging_face_checkpoint_path}")

    if args.repo_name:
        push_model_to_hub(args.hugging_face_checkpoint_path, args.repo_name)


if __name__ == "__main__":
    _args = get_args()
    main(_args)
