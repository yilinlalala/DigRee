"""Sampling utilities: load a DigRee checkpoint, generate sequences and save them."""
import csv
import json
import os
import re
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffusion.graph_features import GraphFeatures
from diffusion_model_discrete import DiscreteDenoisingDiffusion

warnings.filterwarnings("ignore", category=PossibleUserWarning)

NUC_MAP = {0: "A", 1: "U", 2: "G", 3: "C"}
BATCH_SIZE = 4
REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG_DIR = REPO_ROOT / "configs" / "task_config"
_DATASET_INFO_ATTRS = (
    "n_nodes",
    "nodes_dist",
    "max_n_nodes",
    "node_types",
    "edge_types",
    "input_dims",
    "output_dims",
)


def task_config_path(ckpt_path: str) -> Path:
    """Map checkpoints/DigRee_<name>.ckpt to configs/task_config/DigRee_<name>_config.json."""
    stem = Path(ckpt_path).stem
    return TASK_CONFIG_DIR / f"{stem}_config.json"


def load_task_sampling_config(ckpt_path: str) -> dict:
    """Read y.init_range and target.default from the checkpoint's task config.

    Sequence length distribution and X/E marginals come from the checkpoint's
    dataset_infos, not from this file.
    """
    path = task_config_path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No task config for checkpoint '{Path(ckpt_path).name}'. Expected {path}."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Task config {path} must be a JSON object")
    try:
        y_range = data["y"]["init_range"]
        target_default = data["target"]["default"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Task config {path} must contain y.init_range and target.default"
        ) from exc
    if not isinstance(y_range, (list, tuple)) or len(y_range) != 2:
        raise ValueError(f"Task config {path} y.init_range must be a pair [low, high]")
    try:
        y_low, y_high = float(y_range[0]), float(y_range[1])
        target_default = float(target_default)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Task config {path} has a non-numeric y.init_range or target.default"
        ) from exc
    if not y_low < y_high:
        raise ValueError(
            f"Task config {path} y.init_range must satisfy low < high, got {(y_low, y_high)}"
        )
    return {
        "config_path": path,
        "y_init_range": (y_low, y_high),
        "target_default": target_default,
    }


def resolve_target_zscore(ckpt_path: str, target_zscore: Optional[float]) -> float:
    """Use the caller value when it was set; otherwise target.default from the matched config."""
    if target_zscore is not None:
        return float(target_zscore)
    return load_task_sampling_config(ckpt_path)["target_default"]


def _load_checkpoint(ckpt_path: str):
    """Model config, dataset_infos and model weights saved in the checkpoint.

    dataset_infos (length distribution, X/E marginals, input/output dims) must
    come from the checkpoint itself; it differs between tasks.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    try:
        cfg = ckpt["hyper_parameters"]["cfg"]
        infos = ckpt["hyper_parameters"]["dataset_infos"]
        state_dict = ckpt["state_dict"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"Checkpoint {ckpt_path} is missing cfg, dataset_infos or state_dict"
        ) from exc
    missing = [name for name in _DATASET_INFO_ATTRS if getattr(infos, name, None) is None]
    if missing:
        raise ValueError(
            f"Checkpoint {ckpt_path} dataset_infos is missing: {', '.join(missing)}"
        )
    # Training-time metric buffers are not part of the inference model.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("sampling_metrics.")}
    return cfg, infos, state_dict


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def zscore_to_float(zscore) -> float:
    if isinstance(zscore, torch.Tensor):
        return float(zscore.reshape(-1)[0].item())
    match = re.search(r"\[([-\d.eE+]+)\]", str(zscore))
    if match:
        return float(match.group(1))
    return float(zscore)


def sample_to_dict(item, sample_id: int) -> dict:
    seq_codes = item[0].tolist()
    sequence = "".join(NUC_MAP[c] for c in seq_codes)
    edge_types = item[1]
    if isinstance(edge_types, torch.Tensor):
        edge_types = edge_types.numpy()
    pairs = []
    n = len(seq_codes)
    for i in range(n):
        for j in range(i + 1, n):
            if edge_types[i, j] != 0:
                pairs.append((i, j))
    return {
        "id": sample_id,
        "sequence": sequence,
        "length": len(seq_codes),
        "zscore": zscore_to_float(item[2]),
        "pairs": pairs,
    }


def save_samples_csv(samples, csv_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = ["id", "sequence", "length", "zscore"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, item in enumerate(samples, start=1):
            row = sample_to_dict(item, i)
            writer.writerow({k: row[k] for k in fieldnames})
    return csv_path


def save_samples(samples, output_dir) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "generated_samples1.txt")
    for i in range(2, 100):
        if not os.path.exists(filename):
            break
        filename = os.path.join(output_dir, f"generated_samples{i}.txt")

    with open(filename, "w") as f:
        for item in samples:
            f.write(f"N={item[0].shape[0]}\n")
            f.write(f"Zscore={item[2]}\n")
            f.write("X: \n")
            for at in item[0].tolist():
                f.write(f"{at} ")
            f.write("\n")
            f.write("E: \n")
            for bond_list in item[1]:
                for bond in bond_list:
                    f.write(f"{bond} ")
                f.write("\n")
            f.write("\n")

    csv_path = os.path.splitext(filename)[0] + ".csv"
    save_samples_csv(samples, csv_path)
    return filename, csv_path


def load_model(ckpt_path: str, device: torch.device) -> DiscreteDenoisingDiffusion:
    sampling_cfg = load_task_sampling_config(ckpt_path)
    cfg, dataset_infos, state_dict = _load_checkpoint(ckpt_path)
    graph_features = GraphFeatures(cfg.model.graph_features, dataset_info=dataset_infos)
    model = DiscreteDenoisingDiffusion(cfg=cfg, dataset_infos=dataset_infos, graph_features=graph_features)
    model.load_state_dict(state_dict, strict=True)
    model.y_init_range = sampling_cfg["y_init_range"]
    model.target_default = sampling_cfg["target_default"]
    model = model.to(device)
    model.eval()
    return model


def run_sampling(
    model: DiscreteDenoisingDiffusion,
    n_samples: int,
    target_zscore: float,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List:
    set_seed(seed)
    model.target = target_zscore

    samples = []
    left = n_samples
    done = 0

    with torch.no_grad():
        while left > 0:
            to_generate = min(left, BATCH_SIZE)
            batch_results = model.sample_batch(batch_size=BATCH_SIZE, num_nodes=None)
            samples.extend(batch_results[:to_generate])
            left -= to_generate
            done += to_generate
            if progress_callback is not None:
                progress_callback(done, n_samples)

    return samples
