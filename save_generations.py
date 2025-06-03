"""
Nanotron Inference Script

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 save_generations.py --ckpt-path /datadrive/pavan/CurLL/nanotron/checkpoints/stages_0_1_mix/3500 --dataset-name Pavankalyan/stage1_instruct_split --dataset-split test
```
"""

import argparse
import os
from pathlib import Path
import json
import pickle # Added for saving results as pickle

import torch
import torch.nn.functional as F # Added for perplexity calculation
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    GenerationArgs,
    LoggingArgs,
    ParallelismArgs,
    get_config_from_file,
)
from nanotron.generation.decode import (
    GenerationInput,
    TokenizerConfig,
    decode_text,
    decode_tokenized,
)
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import (
    OneForwardOneBackwardPipelineEngine,
)
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.random import (
    RandomStates,
    get_current_random_state,
    get_synced_random_state,
    set_random_seed,
)
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters

try:
    from transformers import AutoTokenizer
    from datasets import load_dataset
except ImportError:
    AutoTokenizer = None
    load_dataset = None # Handle case where datasets is not installed

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--dp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=0)
    parser.add_argument("--tp", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--use-cache", action="store_true", help="Use KV cache to speed up generation")
    parser.add_argument("--dataset-name", type=str, default="Pavankalyan/stage1_instruct_split", help="Hugging Face dataset name to evaluate on")
    parser.add_argument("--dataset-split", type=str, default="test", help="Dataset split to use (e.g., 'test' or 'validation')")
    return parser.parse_args()


def main():
    args = get_args()
    sampler_tech = "top_k"
    max_test_samples = 10   # Limit the number of samples for testing

    assert args.ckpt_path.exists(), f"Checkpoint path {args.ckpt_path} does not exist"

    config = get_config_from_file((args.ckpt_path / "config.yaml").as_posix())
    model_config = config.model.model_config
    tokenizer_path = config.tokenizer.tokenizer_name_or_path

    parallel_config = ParallelismArgs(
        dp=args.dp or config.parallelism.dp,
        pp=args.pp or config.parallelism.pp,
        tp=args.tp or config.parallelism.tp,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )

    # Initialise all process groups
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    # Set log levels
    logging_config = LoggingArgs(
        log_level="info",
        log_level_replica="info",
    )

    # Set log levels
    set_ranks_logging_level(parallel_context=parallel_context, logging_config=logging_config)

    log_rank(f"model_config: {model_config}", logger=logger, level=logging.INFO, rank=0)
    log_rank(f"tokenizer_path: {tokenizer_path}", logger=logger, level=logging.INFO, rank=0)

    dtype = torch.bfloat16

    # Set random states
    set_random_seed(42)

    model_config_cls = model_config.__class__.__name__
    if model_config_cls not in CONFIG_TO_MODEL_CLASS:
        raise ValueError(
            f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"
        )

    # Get synchronized random states
    if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        random_states = RandomStates(
            {"tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=parallel_context.tp_pg)}
        )
    else:
        # We don't need to sync across TP when using sequence parallel (REDUCE_SCATTER)
        random_states = RandomStates({})

    model = build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        dtype=dtype,
        parallel_context=parallel_context,
    )

    # Mark some parameters as tied
    mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)

    # Sanity check model
    sanity_check(root_module=model)

    # Load checkpoint
    checkpoint_path = args.ckpt_path
    log_rank(
        f"Loading checkpoint from {checkpoint_path}:",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    load_weights(model=model, parallel_context=parallel_context, root_folder=checkpoint_path)

    model.eval()
    if AutoTokenizer is not None and load_dataset is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif getattr(model.config, "pad_token_id", None) is not None:
                tokenizer.pad_token_id = int(model.config.pad_token_id)
            elif getattr(model.config, "eos_token_id", None) is not None:
                tokenizer.pad_token_id = int(model.config.eos_token_id)
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        # Load the dataset
        log_rank(f"Loading dataset: {args.dataset_name}, split: {args.dataset_split}", logger=logger, level=logging.INFO, rank=0)
        try:
            dataset = load_dataset(args.dataset_name, split=args.dataset_split).select(range(max_test_samples))
        except Exception as e:
            log_rank(f"Failed to load dataset '{args.dataset_name}' with split '{args.dataset_split}': {e}", logger=logger, level=logging.ERROR, rank=0)
            dist.barrier()
            return

        # Prepare inputs for batch inference
        # We store the original dataset item along with the prompt for later saving.
        input_prompts_with_data = []
        for item in dataset:
            input_prompts_with_data.append({
                "prompt_text": f"<|user|>{item['instruction']}<|assistant|>",
                "original_data": item # Store the entire original dataset item
            })

        log_rank(f"Prepared {len(input_prompts_with_data)} prompts from the dataset.", logger=logger, level=logging.INFO, rank=0)

        outputs = decode_text(
            input_iter=(GenerationInput(text=data["prompt_text"]) for data in input_prompts_with_data),
            tokenizer=tokenizer,
            model=model.model,
            parallel_context=parallel_context,
            max_new_tokens=args.max_new_tokens,
            max_micro_batch_size=256, # You can adjust this based on your GPU memory
            generation_config=GenerationArgs(sampler=sampler_tech, use_cache=args.use_cache),
            tokenizer_config=TokenizerConfig(max_input_length=None),
            is_bench=os.environ.get("USE_BENCH", "0") == "1",
            # returns_logits=True, # Set to True to get logits for perplexity/LM loss calculation
        )

        # Store generated outputs
        all_generated_results = []

        # Iterate through outputs and original data simultaneously
        for i, (output, original_data_item) in enumerate(zip(outputs, input_prompts_with_data)):
            input_ids = output.input_ids
            generated_ids = output.generation_ids
            logits = output.return_logits # Get the logits for the generated tokens

            # If using pipeline parallelism, input_ids, generated_ids, and logits might be TensorPointer
            # and only the last rank in the pipeline will have the actual tensors.
            # We skip if they are TensorPointer as the actual tensors are not available on all ranks.
            if isinstance(input_ids, TensorPointer):
                assert isinstance(generated_ids, TensorPointer)
                assert isinstance(logits, TensorPointer) # Logits will also be TensorPointer
                continue
            assert isinstance(generated_ids, torch.Tensor)
            assert isinstance(logits, torch.Tensor) # Ensure logits are tensors here

            original_input_text = tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)
            generated_text = tokenizer.decode(generated_ids[len(input_ids) :], clean_up_tokenization_spaces=False)

            # Calculate LM Loss and Perplexity
            lm_loss = float('nan') # Initialize with NaN
            perplexity = float('nan') # Initialize with NaN

            if logits is not None and generated_ids is not None:
                # Handle both 1D and 2D tensors for generated_ids
                if generated_ids.dim() == 1:
                    # 1D tensor case - just slice from input length
                    generated_tokens_only = generated_ids[len(input_ids):]
                elif generated_ids.dim() == 2:
                    # 2D tensor case - slice along sequence dimension
                    generated_tokens_only = generated_ids[:, len(input_ids):]
                else:
                    log_rank(f"Warning: Unexpected generated_ids dimensions: {generated_ids.dim()}", logger=logger, level=logging.WARNING, rank=0)
                    generated_tokens_only = None

                if generated_tokens_only is not None:
                    # Handle logits dimensions similarly
                    if logits.dim() == 2:
                        # 2D logits (seq_len, vocab_size)
                        current_logits = logits
                        current_generated_tokens = generated_tokens_only
                    elif logits.dim() == 3:
                        # 3D logits (batch_size, seq_len, vocab_size)
                        current_logits = logits.squeeze(0) if logits.shape[0] == 1 else logits[0]
                        current_generated_tokens = generated_tokens_only.squeeze(0) if generated_tokens_only.dim() == 2 and generated_tokens_only.shape[0] == 1 else generated_tokens_only
                    else:
                        log_rank(f"Warning: Unexpected logits dimensions: {logits.dim()}", logger=logger, level=logging.WARNING, rank=0)
                        current_logits = None
                        current_generated_tokens = None

                    if current_logits is not None and current_generated_tokens is not None:
                        # Ensure we have matching dimensions
                        if current_generated_tokens.dim() == 1 and current_logits.dim() == 2:
                            # Create a mask to ignore padding tokens when calculating loss
                            mask = (current_generated_tokens != tokenizer.pad_token_id)
                            
                            if mask.any(): # Only calculate if there are non-padding tokens in the generated sequence
                                # Filter logits and labels based on the mask
                                filtered_logits = current_logits[mask]
                                filtered_labels = current_generated_tokens[mask]

                                if filtered_logits.numel() > 0 and filtered_labels.numel() > 0:
                                    # Calculate cross-entropy loss
                                    lm_loss = F.cross_entropy(filtered_logits, filtered_labels).item()
                                    # Calculate perplexity
                                    perplexity = torch.exp(torch.tensor(lm_loss)).item()
                                else:
                                    # Case where mask is true but filtered tensors are empty (shouldn't happen with mask.any() check)
                                    lm_loss = float('nan')
                                    perplexity = float('nan')
                            else:
                                # No non-padding tokens to calculate loss on
                                lm_loss = float('nan')
                                perplexity = float('nan')
                        else:
                            log_rank(f"Warning: Dimension mismatch - logits: {current_logits.shape}, tokens: {current_generated_tokens.shape}", logger=logger, level=logging.WARNING, rank=0)

            # Record the results, including all original dataset variables, logits, loss, and perplexity
            result_entry = {
                "original_prompt_text": original_input_text,
                "generated_response": generated_text,
                "dataset_entry": original_data_item["original_data"], # Include the full original dataset item
                "generated_logits": logits.cpu() if logits is not None else None, # Save as CPU tensor for pickle
                "lm_loss": lm_loss,
                "perplexity": perplexity,
            }
            all_generated_results.append(result_entry)

            # Log for debugging/monitoring on rank 0
            log_rank(
                f"--- Sample {i+1} ---",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            log_rank(
                f"Input: {original_input_text[:1000]}", # Truncate long inputs for logging
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            log_rank(
                f"Generated: {generated_text}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            if not torch.isnan(torch.tensor(lm_loss)): # Only log if loss is a valid number
                log_rank(
                    f"LM Loss: {lm_loss:.4f}, Perplexity: {perplexity:.4f}",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )
            else:
                log_rank(
                    "LM Loss and Perplexity: Not applicable (e.g., no non-padding tokens generated)",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )
            log_rank(
                "--------------------------------------------------",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
        
        # Save all generated results to a PICKLE file on rank 0
        if dist.get_rank(parallel_context.world_pg) == 0:
            log_rank(f"Inference complete. Total {len(all_generated_results)} samples processed.", logger=logger, level=logging.INFO, rank=0)
            output_filename = f"inference_results_{args.dataset_split}.pkl" # Changed to .pkl extension
            with open(output_filename, "wb") as f: # Open in binary write mode for pickle
                pickle.dump(all_generated_results, f)
            log_rank(f"Results saved to {output_filename}", logger=logger, level=logging.INFO, rank=0)

    else: # Fallback if transformers or datasets library is not found
        log_rank("transformers or datasets library not found. Please install them to use dataset inference.", logger=logger, level=logging.ERROR, rank=0)
        # Original dummy inference logic (kept as a fallback if libraries are missing)
        outputs = decode_tokenized(
            input_ids=torch.zeros(1, 1).to(dtype=torch.int64, device="cuda"),
            input_mask=torch.ones(1, 1).to(dtype=torch.bool, device="cuda"),
            model=model.model,
            parallel_context=parallel_context,
            generation_config=GenerationArgs(sampler=sampler_tech, use_cache=True),
            max_micro_batch_size=1,
            max_new_tokens=12,
            # returns_logits=True, # Also set to True for dummy case
        )
        for output in outputs:
            input_ids = output.input_ids
            generated_ids = output.generation_ids
            logits = output.logits # Get the logits
            if isinstance(input_ids, TensorPointer):
                assert isinstance(generated_ids, TensorPointer)
                assert isinstance(logits, TensorPointer)
                continue
            assert isinstance(generated_ids, torch.Tensor)
            assert isinstance(logits, torch.Tensor)

            log_rank(
                f"generation: {generated_ids[len(input_ids) :]}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            log_rank(
                f"Logits shape: {logits.shape}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            log_rank(
                "--------------------------------------------------",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

    dist.barrier()


if __name__ == "__main__":
    main()