"""Load a DigRee checkpoint and generate sequences."""
import argparse
import os
import sys
import warnings


def _setup_gpu_before_torch():
    """CUDA_VISIBLE_DEVICES must be set before importing torch."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


_setup_gpu_before_torch()

import torch
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from sampling_core import (
    load_model,
    resolve_target_zscore,
    run_sampling,
    save_samples,
    task_config_path,
)

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a DigRee checkpoint")
    parser.add_argument(
        "--ckpt",
        default="../checkpoints/DigRee_IRES_EIF3A_forhuman.ckpt",
        help="Path to checkpoint (.ckpt)",
    )
    parser.add_argument("--n-samples", type=int, default=20, help="Number of sequences to generate")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: checkpoint parent dir)")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Physical GPU id (sets CUDA_VISIBLE_DEVICES before torch init)",
    )
    parser.add_argument(
        "--target-zscore",
        type=float,
        default=None,
        help="Target Z-score for guided sampling. If omitted, use target.default from the matched task config.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    output_dir = args.output_dir or os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: cuda:0 (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})")
    else:
        device = torch.device("cpu")
        print("CUDA not available, running on CPU")

    ckpt_path = os.path.abspath(args.ckpt)
    target_zscore = resolve_target_zscore(ckpt_path, args.target_zscore)
    model = load_model(ckpt_path, device)
    print(f"Task config: {task_config_path(ckpt_path)}")
    print(f"y init range: {model.y_init_range}")
    print(f"Target Z-score: {target_zscore}")
    print(f"Random seed: {args.seed}")
    print(f"Sampling {args.n_samples} sequences from {args.ckpt}")

    samples = run_sampling(
        model=model,
        n_samples=args.n_samples,
        target_zscore=target_zscore,
        seed=args.seed,
        progress_callback=lambda done, total: print(f"  remaining: {total - done}/{total}"),
    )

    txt_path, csv_path = save_samples(samples, output_dir)
    print(f"Saved {len(samples)} samples to {txt_path}")
    print(f"Saved CSV summary to {csv_path}")


if __name__ == "__main__":
    main()
