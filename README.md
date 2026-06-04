# Muon Optimizer: Targeted Newton-Schulz and Spectral Tracking

This repository contains an experimental extension of the Muon optimizer based on the Airbench94 CIFAR-10 framework (https://github.com/KellerJordan/cifar10-airbench) and the NanoGPT speedrunning framework (https://github.com/KellerJordan/modded-nanogpt). It is designed to track optimizer metrics, analyze gradient spectra, and test novel "targeted" variants of the Newton-Schulz iteration.

## Overview

The core script trains a custom ResNet-style architecture (CifarNet) on CIFAR-10, attaining ~94% accuracy. The primary goal of this codebase is to benchmark standard SGD, standard Muon, and targeted Muon optimizers while observing how gradients and parameters behave in the isospectral manifold tangent space.

## Key Features & Additions

### 1. Targeted Newton-Schulz Iteration
The standard 5th-order Newton-Schulz iteration has been expanded to support **targeted spectral projection**:
* **`targeted_newtonschulz5`**: Modifies the iteration to shift the spectrum by a scalar `tau`. 
* Can isolate and return the top (`return_top=True`) or bottom (`return_top=False`) spectral components of the matrix.

### 2. Tangent Space Projection & Spectral Tracking
To better understand how updates affect the network weights over time:
* **`project_onto_tangent_space`**: A custom SVD-based utility that projects gradient updates onto the isospectral manifold tangent space of the current weight matrix.
* **`TrackedSGD` & Updated `Muon`**: Both optimizers now support a statistical tracking mechanism. Based on `svd_prob`, the optimizers periodically compute and store:
    * Original Gradient SVD spectrum.
    * Parameter SVD spectrum.
    * Update SVD spectrum.
    * Tangent Projection Metric (measuring how much of the update lies in the tangent space).

### 3. Configurable Data Skew
The `CifarLoader` now includes a `skew` parameter. When provided, it artificially imbalances the CIFAR-10 training set according to a power-law distribution, allowing researchers to evaluate optimizer robustness against long-tail class distributions.

### 4. Multi-Configuration Testbed
Instead of a single execution loop, the runner automatically sequences through several optimization strategies in a single run:
* **Tracked SGD** (Baseline)
* **Muon Standard**
* **Muon Targeted (Bot & Top)** across various `tau` thresholds (`0.1`, `0.5`, `1.0`, `5.0`, `10.0`).

### 5. GPT-2 Training Enhancements (train_gpt2_new.py)
The GPT-2 training script has been updated with several new features to support tracking and targeted projection:
* **Weights & Biases (WandB) Integration:** The training script now initializes a `wandb` run to track hyperparameter configurations and log metrics during training. 
* **Spectral Tracking & Effective Rank:** A `compute_effective_rank(svds)` function was added to compute the effective rank of singular values. These metrics for parameters, gradients, and updates are now logged to WandB every 100 steps inside the Muon optimizer's `step()` method.
* **Targeted Newton-Schulz Backends:** The zero-power approximation mappings have been expanded. The script now includes `targeted_top_newtonschulz5`, `targeted_bot_newtonschulz5`, and an `identity` pass-through, augmenting the original `svd` and `newtonschulz5` backends.
* **Optimizer Toggling & Hyperparameters:** The `Hyperparameters` dataclass was refactored to allow users to toggle between `Muon` and `AdamW` optimizers (`args.optim`). It also introduces explicit settings for `muon_lr` and `backend`, while increasing the `warmup_iters` to `250` (up from `0`).

## Usage

Run the scripts directly via Python. They require PyTorch 2.4.1+ and a CUDA-enabled GPU (optimally an A100/H100 for benchmark timings).

**For CIFAR-10 ResNet:**
```bash
python airbench94_muon_new.py
```
**For NanoGPT-2:**



```bash
# Distributed Data Parallel is supported via torchrun
torchrun --nproc_per_node=8 train_gpt2_new.py
```
