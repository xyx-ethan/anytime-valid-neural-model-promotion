# Anytime-Valid Gates for Plasticity and Retention in Continual Learning

This repository contains the analysis code, numerical source data, figures, and
proof checks for the article.

## Statistical simulations and proof checks

Create a Python environment and install the recorded dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Reproduce the false-promotion simulations, verify the implementation, and
rebuild the figures:

```bash
.venv/bin/python analysis/run_analysis.py
.venv/bin/python analysis/verify_proofs.py
.venv/bin/python analysis/make_figure.py
```

The simulation uses 8,000 runs, a horizon of 1,000 observations, and the fixed
seed recorded in `analysis/run_analysis.py`.

## Split CIFAR-10 experiment

The neural experiment uses PyTorch and torchvision from the NVIDIA PyTorch
26.01 container. Exact package versions, the container digest, and execution
source hashes are recorded in `source_data/computational_environment.json`.
The script downloads CIFAR-10, creates mutually exclusive model-fitting,
development, and confirmation partitions, trains the initial ResNet-18, adapts
three candidates, freezes them, and evaluates the four promotion requirements.

The confirmation configuration can be reproduced with:

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD":/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:26.01-py3 \
  python analysis/run_plasticity_retention_stream.py \
    --data-dir .data \
    --initial-epochs 20 \
    --adaptation-epochs 5 \
    --adaptation-size 15000 \
    --evaluation-size 5000 \
    --batch-size 128 \
    --adaptation-learning-rate 0.003 \
    --replay-size 15000 \
    --replay-ratio 3.25 \
    --candidate-modes replay naive no_update \
    --evaluation-split confirmation \
    --partition-seed 20260723 \
    --output-dir reproduced_confirmation \
    --seeds 1 2 3 4 5
```

The reported seeds were executed independently. Outputs from independent runs
can be checked and combined with:

```bash
.venv/bin/python analysis/merge_confirmation_outputs.py \
  --input-root independent_runs \
  --output-dir reproduced_confirmation
```

Byte-for-byte copies of the source used for the confirmation runs are preserved
in `provenance/`. The script under `analysis/` is a self-contained equivalent
that inlines its helper functions and adds input validation.

The development grid and its frozen selection rule are recorded in
`source_data/plasticity_retention_development/`. Each row gives the replay
ratio, learning rate, seed, evaluation partition, four crossing outcomes, and
performance summaries used for selection.

## Included outputs

`source_data/Figure1_source_data.csv` and
`source_data/analysis_summary.json` contain the simulation results.
`source_data/plasticity_retention_confirmation/` contains the five-seed
candidate table, initial-model table, run metadata, and all 60 paired
difference arrays. Figures 1-3 are included as PDF and SVG files.

`analysis/proof_verification_report.json` records symbolic checks, direct
agreement tests between two implementations of the evidence process, and
finite checks of the conjunctive rule. The cited probability theorem supplies
the general time-uniform guarantee.

## Data and licensing

The repository does not redistribute CIFAR-10 images. Those data retain their
original terms. The generated paired-loss arrays and numerical summaries are
included for independent verification. The analysis code is released under
the MIT License.

The manuscript cites the immutable `v1.0.1` repository release.
