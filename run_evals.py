import argparse
import os
from typing import Optional

from nanotron.config import ParallelismArgs

from lighteval.config.lighteval_config import (
    GenerationArgs,
    LightEvalConfig,
    LightEvalLoggingArgs,
    LightEvalTasksArgs,
)
from lighteval.main_nanotron import nanotron


def create_lighteval_config(
    output_dir: str = "./eval_results",
    tasks: str = "lighteval|agieval:aqua-rat|5|0",
    custom_tasks: str = None,
    batch_size: int = 16,
    dp: int = 1,
    pp: int = 1,
    tp: int = 1,
    max_samples: Optional[int] = None,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 42,
    use_cache: bool = True,
    save_details: bool = True,
    push_to_hub: bool = False,
    results_org: Optional[str] = None,
) -> LightEvalConfig:
    """
    Create a LightEvalConfig object programmatically.

    Args:
        output_dir: Directory where evaluation results will be saved
        tasks: Task specification in format "suite|task|num_few_shots|truncate_few_shots"
        batch_size: Batch size for evaluation
        dp: Data parallel size
        pp: Pipeline parallel size
        tp: Tensor parallel size
        max_samples: Maximum number of samples to evaluate (None for all)
        temperature: Generation temperature
        top_k: Top-k for sampling
        top_p: Top-p for sampling
        seed: Random seed
        use_cache: Whether to use KV cache during generation
        save_details: Whether to save detailed results
        push_to_hub: Whether to push results to Hugging Face Hub
        results_org: Organization to push results to on the Hub

    Returns:
        LightEvalConfig: Config object for lighteval
    """
    # Create logging config
    logging_args = LightEvalLoggingArgs(
        output_dir=output_dir,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=False,
        public_run=False,
        results_org=results_org,
        tensorboard_metric_prefix="eval",
    )

    # Create tasks config
    tasks_args = LightEvalTasksArgs(
        tasks=tasks,
        custom_tasks=custom_tasks,
        max_samples=max_samples,
        dataset_loading_processes=8,
        multichoice_continuations_start_space=None,
        pairwise_tokenization=False,
    )

    # Create parallelism config
    parallelism_args = ParallelismArgs(
        dp=dp,
        pp=pp,
        tp=tp,
    )

    # Create generation config
    generation_args = GenerationArgs(
        sampler="greedy",
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        n_samples=1,
        seed=seed,
        use_cache=use_cache,
    )

    # Return the full config
    return LightEvalConfig(
        logging=logging_args,
        tasks=tasks_args,
        parallelism=parallelism_args,
        batch_size=batch_size,
        generation=generation_args,
    )


def save_lighteval_config_as_yaml(config: LightEvalConfig, output_path: str) -> None:
    """
    Save a LightEvalConfig object as a YAML file.

    Args:
        config: LightEvalConfig object
        output_path: Path to save the YAML file
    """
    # Make directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save config as YAML
    with open(output_path, "w") as f:
        # Create a clean dictionary representation
        config_dict = {
            "logging": {
                "output_dir": config.logging.output_dir,
                "save_details": config.logging.save_details,
                "push_to_hub": config.logging.push_to_hub,
                "push_to_tensorboard": config.logging.push_to_tensorboard,
                "public_run": config.logging.public_run,
                "results_org": config.logging.results_org,
                "tensorboard_metric_prefix": config.logging.tensorboard_metric_prefix,
            },
            "tasks": {
                "tasks": config.tasks.tasks,
                "custom_tasks": config.tasks.custom_tasks,
                "max_samples": config.tasks.max_samples,
                "dataset_loading_processes": config.tasks.dataset_loading_processes,
                "multichoice_continuations_start_space": config.tasks.multichoice_continuations_start_space,
                "pairwise_tokenization": config.tasks.pairwise_tokenization,
            },
            "parallelism": {
                "dp": config.parallelism.dp,
                "pp": config.parallelism.pp,
                "tp": config.parallelism.tp,
            },
            "batch_size": config.batch_size,
            "generation": {
                "sampler": config.generation.sampler.name.lower()
                if hasattr(config.generation.sampler, "name")
                else config.generation.sampler,
                "temperature": config.generation.temperature,
                "top_k": config.generation.top_k,
                "top_p": config.generation.top_p,
                "n_samples": config.generation.n_samples,
                "seed": config.generation.seed,
                "use_cache": config.generation.use_cache,
            },
        }

        # Convert to YAML
        import yaml

        yaml.dump(config_dict, f, default_flow_style=False)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-config-path",
        type=str,
        help="Path to the brr checkpoint YAML or python config file, potentially on S3",
    )
    parser.add_argument(
        "--lighteval-override",
        type=str,
        help="Path to a YAML Lighteval config file for evaluation. Example config: configs/examples/lighteval-config.yaml",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, unknowns = parser.parse_known_args()

    if args.lighteval_override is None:
        lighteval_config_path = "lighteval-config.yaml"

        # Create a custom config
        custom_config = create_lighteval_config(
            output_dir="./eval_results/",
            tasks="leaderboard|hellaswag|0|0,leaderboard|winogrande|0|0,helm|commonsenseqa|0|0,helm|piqa|0|0",
            batch_size=8,
            dp=1,
            pp=1,
            tp=1,
            max_samples=1,  # Use a small number for testing
            temperature=0.0,
        )

        # Save it to a YAML file
        save_lighteval_config_as_yaml(custom_config, lighteval_config_path)
    else:
        lighteval_config_path = args.lighteval_override

    exit()

    nanotron(
        checkpoint_config_path=args.checkpoint_config_path,
        lighteval_config_path=lighteval_config_path,
        cache_dir=args.cache_dir,
    )
