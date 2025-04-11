"""
Nanotron Inference Script

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=1 run_generate.py --ckpt-path checkpoints/10
```
"""

import argparse
import os
from pathlib import Path

import torch
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
except ImportError:
    AutoTokenizer = None

# import lovely_tensors as lt

# lt.monkey_patch()

logger = logging.get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=0)
    parser.add_argument("--tp", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    parser.add_argument("--use-cache", action="store_true", help="Use KV cache to speed up generation")
    return parser.parse_args()


def main():
    args = get_args()

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
    # TODO @nouamane: this is only needed for training, can we just mark params as NanotronParameter instead?
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
    if AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # tokenizer.pad_token_id = tokenizer.eos_token_id
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
        tokenizer.truncation_side = "left"  # TODO @nouamane: do we want this?
        dummy_inputs = [
            # "The future of AI is",
            "Here is a story:\n\nRam is a very good kid. He is very good at math. He is very good at science. He is very good at history. He is very good at geography. He is very good at English. He is very good at Hindi. He is very good at computer science. He is very good at physical education. He is very good at art and craft. He is very good at music. He is very good at dance. He is very good at sports. He is very good at everything he does.\n\nNow answer the following question:\n\nQuestion: What is Ram good at?\n\nAnswer:",
            # "The future of AI is",
            # 'Here is an extract from a webpage: "Have you ever experienced heel pain after a heavy physical activity, or even right after a long period of standing? If you regard this as something usual and normal, then think again. Miscalled as heel pain, plantar fasciitis causes these frequent mild pains experienced in the soles of the feet. It is the inflammation and enlargement the plantar fascia tissue that is located in the heels of the feet, stretching to the base of the toes. This tissue is responsible for absorbing shock in the feet and for supporting the arches. It also plays a vital role in foot movements during walking and standing. Many factors such as excessive walking, standing, and running trigger heel pain and plantar fasciitis. A sudden increase in intensity of activities, increase in weight, and abrupt change of footwear also cause the swelling of the ligament. Non-supportive footwear lacking arch cushions and improper and worn out running or training can also lead to the problem. It is also most evident among those". Write an extensive and detailed course unit suitable for a textbook targeted at college students, related to the given extract, within the context of "Medicine". Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on: - Rigor: Ensure in-depth coverage of the concepts/sections. - Engagement: Write with an academic, professional and engaging tone that captivates interest. - Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history. Do not include a title or an introduction, simply write the content without headlines and introductory phrases. Do not use images.',
            # "Advancements in technology will lead to",
            "Tomorrow's world is shaped by",
            """Here is a story:

Once upon a time, in a sunny little town, was a playground! Leo the lion loved the playground. It had a big red slide, a swing that went WHOOSH!, and a sandy sandbox.

One day, Leo went to the playground with his friend Mia the monkey. Mia was holding a bright yellow bucket and a little blue shovel. "Let's build a castle!" she said. Leo nodded, his mane bouncing.

They started digging in the sandbox. Leo used his big paws to scoop lots and LOTS of sand. Mia carefully packed it into the bucket. But then, a little cloud came over the sun, and a few raindrops started to fall. *Plink, plink, plink!*

"Oh no!" cried Mia. "Our castle will get washed away!"

Leo looked around. He saw the slide was covered, so nobody could slide. He saw the swings weren't swinging anymore. Then, he had an idea! "Let's cover the castle with my big leaves!" he said, running to a nearby tree. He gathered some big green leaves and they carefully put them over the sandcastle.

The rain stopped after a little while and the sun peeked out again. Mia and Leo took the leaves off their castle. It was still there! It looks a little smushed, but it was still a castle!

"Yay! We saved it!" shouted Mia. Leo smiled. Building a castle with a friend, even when it rained, was the best! They played on the slide and swings until it was time to go home. Leo was so happy he helped save his friend's castle. He knew Mia was happy too, because she gave him a big hug!

Now, can you tell me - what do you think Mia was feeling when the rain started? And why did Leo run to the tree? Do you think they'll play at the playground again tomorrow? Why?

Answer some questions based on the context above:

Question: Mia was making a castle, but then it started to rain. How do you think Mia felt when the rain started falling on her castle? Tell me why you think she felt that way.""",
        ]

        outputs = decode_text(
            input_iter=(GenerationInput(text=text) for text in dummy_inputs),
            tokenizer=tokenizer,
            model=model.model,
            parallel_context=parallel_context,
            max_new_tokens=args.max_new_tokens,
            max_micro_batch_size=2,
            generation_config=GenerationArgs(sampler="greedy", use_cache=args.use_cache),
            tokenizer_config=TokenizerConfig(max_input_length=None),
            is_bench=os.environ.get("USE_BENCH", "0") == "1",
        )
        for output in outputs:
            input_ids = output.input_ids
            generated_ids = output.generation_ids
            if isinstance(input_ids, TensorPointer):
                assert isinstance(generated_ids, TensorPointer)
                continue
            assert isinstance(generated_ids, torch.Tensor)

            log_rank(
                f"input: {tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)[:1000]}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            log_rank(
                f"generation: {tokenizer.decode(generated_ids[len(input_ids) :], clean_up_tokenization_spaces=False)}",
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
    else:
        outputs = decode_tokenized(
            input_ids=torch.zeros(1, 1).to(dtype=torch.int64, device="cuda"),
            input_mask=torch.ones(1, 1).to(dtype=torch.bool, device="cuda"),
            model=model.model,
            parallel_context=parallel_context,
            generation_config=GenerationArgs(sampler="greedy", use_cache=True),
            max_micro_batch_size=1,
            max_new_tokens=12,
            returns_logits=False,
        )
        for output in outputs:
            input_ids = output.input_ids
            generated_ids = output.generation_ids
            if isinstance(input_ids, TensorPointer):
                assert isinstance(generated_ids, TensorPointer)
                continue
            assert isinstance(generated_ids, torch.Tensor)
            log_rank(
                f"generation: {generated_ids[len(input_ids) :]}",
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
