# DigRee

DigRee designs **RNA regulatory elements** to modulate translation through structure-guided sequence generation. It models **folded stem-loop structures** and their interactions with regulatory proteins, representing structural submotifs as **compact triplets** and assembling them according to their orientation and spatial proximity. Candidate sequences are continuously optimized based on **structural stability and binding ability**, with each new structure required to improve upon the current binding score. Generation and optimization are integrated into a unified model, enabling DigRee to design **novel regulatory elements beyond known sequences**. The framework supports **IRES, SECIS, and 5′ UTR** design for both **linear and circular translatable RNA**.

Source code: https://github.com/yilinlalala/DigRee

Web server: https://www.lilab-ecust.cn/digree/

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

# Overview

Each sample is denoised jointly as a nucleotide sequence and a secondary-structure graph, while the prediction is steered towards a user-defined target value.

This repository contains the sampling code. The three trained checkpoints are published as release attachments and are not stored in Git. Training code and datasets are not included.

## Repository layout

```
DigRee/
├── checkpoints/                   # place the downloaded .ckpt files here
├── configs/task_config/           # per-checkpoint sampling settings
├── src/
│   ├── sample.py                  # command-line entry point
│   ├── sampling_core.py           # checkpoint loading, sampling loop, output files
│   ├── diffusion_model_discrete.py  # reverse diffusion with target-value guidance
│   ├── utils.py
│   ├── models/                    # graph transformer
│   ├── diffusion/                 # noise schedule, transitions, graph features
│   └── datasets/                  # class needed to unpickle checkpoint metadata
├── run_generate_test.sh
└── requirements.txt
```



# System Requirements



## Hardware requirements

A CUDA GPU is recommended. This release was tested on an NVIDIA A30 (24 GB) with NVIDIA driver 595.71.05.

## Software requirements

### OS Requirements

This package has been tested on the following system:

- Linux: Debian GNU/Linux 13 (trixie), kernel 6.12.95+deb13-amd64 (x86_64)
- Python: 3.12.0



### Python Dependencies

DigRee depends on the Python scientific stack. Tested versions:

```
numpy==2.5.1
scipy==1.18.0
torch==2.12.0+cu132
torch_geometric==2.8.0
pytorch_lightning==2.6.5
omegaconf==2.3.1
```

`torch==2.12.0+cu132` is the PyTorch build for CUDA 13.2.

# Installation Guide

Install the tested PyTorch build first, then the remaining dependencies:

```bash
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu132
cd DigRee
pip install -r requirements.txt
```



# Demo

`sample.py` resolves its imports relative to `src/`, so run it from that directory:

```bash
cd DigRee/src
python sample.py --gpu 0
```

Or use the helper script from the project root (edit the GPU id inside if needed):

```bash
bash run_generate_test.sh
```



## Output

Each run writes `generated_samples1.txt` and `generated_samples1.csv` to the output directory. If those files already exist, the next free index is used (`generated_samples2`, and so on).

The CSV has the columns `id`, `sequence`, `length`, and the predicted target value.

Each block in the txt file contains:

- `N=` — sequence length
- `Zscore=` — target value predicted by the model
- `X:` — nucleotide indices (`0=A`, `1=U`, `2=G`, `3=C`)
- `E:` — secondary-structure adjacency matrix, where `0` = no edge, `1` = backbone bond between neighbouring positions, `2` = A–U pair, `3` = G–C pair



# Usage

```bash
cd DigRee/src
python sample.py [OPTIONS]
```


| Option            | Default                                          | Description                                                           |
| ----------------- | ------------------------------------------------ | --------------------------------------------------------------------- |
| `--ckpt`          | `../checkpoints/DigRee_IRES_EIF3A_forhuman.ckpt` | Path to the model checkpoint                                          |
| `--n-samples`     | `20`                                             | Number of sequences to generate                                       |
| `--output-dir`    | Parent directory of `checkpoints/`               | Directory for output files                                            |
| `--gpu`           | `0`                                              | Physical GPU id; sets `CUDA_VISIBLE_DEVICES` before PyTorch is loaded |
| `--target-zscore` | `target.default` of the task config              | Target value for guided sampling                                      |
| `--seed`          | `42`                                             | Random seed                                                           |


Example:

```bash
python sample.py \
  --ckpt ../checkpoints/DigRee_5UTR_EL.ckpt \
  --n-samples 50 \
  --gpu 0 \
  --target-zscore 4.0 \
  --output-dir ../outputs
```

Sequences are generated in batches of 4. Runs with the same checkpoint, target value and seed reproduce the same sequences.

## Checkpoints

The weight files are [v1.0 release](https://github.com/yilinlalala/DigRee/releases/tag/v1.0) attachments, not files in the Git tree. Download them and save each one as `checkpoints/<filename>`:

- [DigRee_IRES_EIF3A_forhuman.ckpt](https://github.com/yilinlalala/DigRee/releases/download/v1.0/DigRee_IRES_EIF3A_forhuman.ckpt)
- [DigRee_5UTR_EL.ckpt](https://github.com/yilinlalala/DigRee/releases/download/v1.0/DigRee_5UTR_EL.ckpt)
- [DigRee_SECIS_SBP2.ckpt](https://github.com/yilinlalala/DigRee/releases/download/v1.0/DigRee_SECIS_SBP2.ckpt)

```bash
mkdir -p checkpoints
curl -L -o checkpoints/DigRee_IRES_EIF3A_forhuman.ckpt \
  https://github.com/yilinlalala/DigRee/releases/download/v1.0/DigRee_IRES_EIF3A_forhuman.ckpt
curl -L -o checkpoints/DigRee_5UTR_EL.ckpt \
  https://github.com/yilinlalala/DigRee/releases/download/v1.0/DigRee_5UTR_EL.ckpt
curl -L -o checkpoints/DigRee_SECIS_SBP2.ckpt \
  https://github.com/yilinlalala/DigRee/releases/download/v1.0/DigRee_SECIS_SBP2.ckpt
```

The sampler looks for them at `checkpoints/<name>.ckpt`.


| Checkpoint                        | Task                      | Default target value |
| --------------------------------- | ------------------------- | ---------------------- |
| `DigRee_IRES_EIF3A_forhuman.ckpt` | circRNA IRES, human EIF3A | 12.0                   |
| `DigRee_5UTR_EL.ckpt`             | 5'UTR expression level    | 4.4                    |
| `DigRee_SECIS_SBP2.ckpt`          | SECIS, SBP2               | 1.0                    |


Every checkpoint needs a matching `configs/task_config/<checkpoint name>_config.json` holding two values:

- `y.init_range` — interval the initial target value is drawn from at the start of sampling
- `target.default` — target value used when `--target-zscore` is not given

Everything else (sequence length distribution, nucleotide and base-pair marginals, model dimensions) is stored inside the checkpoint itself.

# Citation

If you use DigRee, please cite:

> [Authors]. [Title]. *[Journal]*. [Year].

# License

This project is covered under the **Apache 2.0 License**.