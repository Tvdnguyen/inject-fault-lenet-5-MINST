# The Implementation of Fault Injection

## 📑 Table of Contents

- [Update](#update)
- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Reference](#reference)

## 🔥 Update
- **July 2024**: We release 1st version of codebase.

## 📍 Introduction

The repository is structured as follows:

- **data/**: This directory contains scripts and utilities for downloading and preprocessing benchmark datasets.
- **output/**: This directory contains processes' outcome including logs and checkpoints.
- **scripts/**: This directory is intended to store experimental scripts.
- **src/**: This directory contains the source code for training, evaluating, and visualizing models.
- **README.md**: This file contains information about the project, including installation instructions, usage examples, and a description of the repository structure.
- **environment.yml**: This file lists all Python dependencies required to run the project.
- **.gitignore**: This file specifies which files and directories should be ignored by Git version control.

## ⚙️ Installation

To re-produce this project, you will need to have the following dependencies installed:
- Ubuntu 18.04.6 LTS
- CUDA Version: 11.7
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3
- [PyTorch](https://pytorch.org/) (version 2.0 or later)

After installing Miniconda, you can create a new environment and install the required packages using the following commands:

```bash
conda create -n inject python=3.9
conda activate inject
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install -r requirements.txt
```

## 📦 Data

After finishing the download process, please put them into the directory **/data**.

## 👾 Usage

- Clone this repo and check out your branch following the template:

```bash
git clone https://github.com/Tvdnguyen/inject-fault-lenet-5-MINST.git
git checkout -b <yourGitName>/<yourFeatureName>
```

- All relevant data is stored within the **/data** directory. To configure the settings for each dataset, corresponding configuration files are provided in the **/src/config** folder.

To reproduce experiments, please refer ```scripts/run.sh```:
```bash
export WORKDIR=./
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
python src/train_model_lenet5_float.py          ## for training
python src/quant.py                             ## for testing quantization
```

and run:

```bash
bash scripts/run.sh
```

## Reference
- [How to Quantize an MNIST network to 8 bits in Pytorch from scratch](https://karanbirchahal.medium.com/how-to-quantise-an-mnist-network-to-8-bits-in-pytorch-no-retraining-required-from-scratch-39f634ac8459)
- [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)