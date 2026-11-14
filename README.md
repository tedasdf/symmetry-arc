# Learnable Symmetry-Preserving Operators for ARC-AGI

**Independent research | ARC Prize 2026 Paper Track**  
[Research write-up](https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/new-writeup-1793313461168)

Can handcrafted spatial primitives in an ARC solver be replaced by learnable operators **without losing the symmetry constraints that make them useful**?

This project extends [CompressARC](https://github.com/iliao2345/CompressARC), a test-time-training approach that optimises a small neural network from scratch for each ARC task. I implemented alternatives to its spatial and directional operations, evaluated their effects through controlled ablations, and built diagnostics to understand **how solutions are discovered, selected and retained** during training—not just how many tasks are solved at the end.

## Research contributions

### Learnable spatial operators

The experiments replace three CompressARC primitives individually:

| Original operation | Investigated replacement | Main idea |
| --- | --- | --- |
| Directional shift | Projected C4/D4 convolution | Learnable directional kernels with rotation/reflection-constrained weight sharing |
| Cumulative maximum | Directional log-sum-exp (LSE) | Smooth directional aggregation with symmetry-shared offset biases |
| Directional sharing | Projected D4 directional sharing | Learnable, symmetry-constrained communication between directional features |

The work studies these operators as alternatives to handcrafted computational priors. Preserving D4 equivariance does **not**, by itself, guarantee better task-solving performance.

### Experimental design and diagnostics

- Compared the baseline with individual operator replacements on **40 ARC-AGI-1 tasks**, using **1,500 optimisation steps per task** and **two random seeds**.
- Recorded **Top-1** and **Pass@2** outcomes, task-level disagreements, and training-time candidate rankings.
- Distinguished three failure modes:
  - **Discovery:** no correct candidate generated.
  - **Selection:** a correct candidate generated but never ranked in the Top-2.
  - **Retention:** a correct candidate reached the Top-2 but was absent from the final Top-2.
- Inspected candidate recurrence, ranking scores, loss/gradient diagnostics, and seed sensitivity to avoid interpreting aggregate accuracy in isolation.

## Results

**Final Pass@2 (tasks solved out of 40)**

| Architecture | Seed 0 | Seed 2 |
| --- | ---: | ---: |
| Original CompressARC | 20 | 23 |
| Projected D4 convolution | **23** | 22 |
| Projected D4 directional sharing | 17 | 23 |
| Directional LSE | 18 | 16 |

D4 convolution performed comparably to the baseline across both seeds; its three-task advantage in seed 0 **did not reproduce in seed 2**. Other operator replacements had less consistent or lower outcomes. The diagnostic analysis showed that generating a correct candidate does not necessarily mean the ranking process selects or retains it.

**Scope and limitations:** This is a selected 40-task investigation, not a full ARC benchmark claim. Half the tasks were drawn from previously reported CompressARC successes. Two seeds are insufficient to establish a systematic advantage, and the LSE variant also differs from the original operator in some rescaling and masked-position details. See the [research write-up](https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/new-writeup-1793313461168) for methodology and interpretation.

## ML systems and performance engineering

Alongside the architecture experiments, I investigated the computational cost of per-task training:

- Profiled small-kernel launch overhead and CPU-to-GPU dispatch bottlenecks.
- Implemented and benchmarked **vectorised directional sharing**, **Triton directional scan kernels**, and **fused/optimised training components**.
- Explored **packed GEMM-based directional operations**, alternative operator backends and configuration-driven timing benchmarks.
- Added an **optional CUDA Graph training path** with parity/preflight checks, including investigations of non-deterministic atomic gradient accumulation in LSE backward kernels.

The CUDA Graph path is optional; the repository documentation describes validation constraints and does not claim bit-identical multi-step trajectories or a verified end-to-end speedup. See [CUDA Graph documentation](src/CUDA_GRAPH.md) and the [validation/interview brief](INTERVIEW_BRIEF.md).

## Running an experiment

Install the pinned dependencies from the repository root (Python 3.11 is documented in `src/requirements.txt`):

```bash
python -m pip install -r src/requirements.txt
```

One documented D4-convolution configuration can be launched as follows, with the required ARC dataset available to the training code:

```bash
python src/train.py \
  --config experiments/seed0_d4conv_retest/configs/projected_conv_d4.json \
  --task 00576224 \
  --set training.split=evaluation \
  --set training.iterations=1500 \
  --set logging.wandb=false
```

This example follows the repository's training configuration and command structure; it has **not** been re-executed as part of preparing this README. Data availability, hardware, and selected backend determine whether a particular experiment runs locally. GPU-specific Triton/CUDA features require compatible hardware and dependencies.

The baseline configuration and backend selections are defined in [`src/config/default.yaml`](src/config/default.yaml). Experiment-specific JSON configs and SLURM launchers are mapped in [`SLURM_EXPERIMENT_MAP.md`](SLURM_EXPERIMENT_MAP.md).

## Repository guide

| Path | Contents |
| --- | --- |
| [`src/`](src/) | Model, training entrypoint, operators, configs, testing and profiling tools |
| [`experiments/`](experiments/) | Experiment configurations and ablation runs |
| [`group_cnn/`](group_cnn/) | Group-convolution work |
| [`kaggle/`](kaggle/) | Kaggle experiment notebooks and launch workflows |
| [`SLURM_EXPERIMENT_MAP.md`](SLURM_EXPERIMENT_MAP.md) | SLURM jobs, configurations, tags and experiment mapping |
| [`ablation_completeness.md`](ablation_completeness.md) | Recorded run coverage and missing task IDs |
| [`src/CUDA_GRAPH.md`](src/CUDA_GRAPH.md) | Optional graph execution, constraints and validation status |
| [`INTERVIEW_BRIEF.md`](INTERVIEW_BRIEF.md) | Technical notes on numerical and validation findings |

## Attribution

This repository is a **fork of [CompressARC by Isaac Liao and collaborators](https://github.com/iliao2345/CompressARC)**. CompressARC supplies the underlying solver and test-time-training setting; the symmetry-preserving operator replacements, ablations, diagnostics and optimisation investigations described above are the focus of this research extension. The original project should be credited when reusing or discussing the baseline implementation.

**Research write-up:** [Learnable Symmetry-Preserving Operators for ARC-AGI](https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/new-writeup-1793313461168)
