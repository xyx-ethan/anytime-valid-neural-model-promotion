# Anytime-Valid Multi-Metric Promotion of Neural Networks in Data Streams

This repository contains the analysis, numerical source data, figures, and
proof checks for the article.

## Reproduction

Create a Python environment and install the recorded dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the analysis and proof checks from the project root:

```bash
.venv/bin/python analysis/run_analysis.py
.venv/bin/python analysis/verify_proofs.py
.venv/bin/python analysis/make_figure.py
```

The tabular-stream evaluation takes about three minutes on a current desktop CPU. The public streams are provided by River. The analysis fixes the simulation seed, network initialization seeds, stream order, warm-up length, and model hyperparameters.

The image-stream experiments require PyTorch and torchvision. They were run
with the NVIDIA PyTorch 26.01 container:

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD":/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:26.01-py3 \
  python analysis/run_cifar_stream.py
```

The script downloads CIFAR-10, creates the warm-up and stream partitions independently for each seed, trains the common ResNet-18 starting point, and records prequential Brier-loss and classification-error differences before updating the candidate. The reported promotion time is the first point by which both prespecified evidence processes have crossed their thresholds.

The CIFAR-10-C experiment uses the 15 original corruption arrays from the
official benchmark release. Download and extract the archive into
`.data/CIFAR-10-C`, then run:

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD":/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:26.01-py3 \
  python analysis/run_cifar10c_stream.py \
    --cifar10c-dir /workspace/.data/CIFAR-10-C \
    --seeds 0 1 2 --warmup-epochs 20 --batch-size 256
```

For each seed, the script trains one clean starting model and evaluates every
corruption from that same model. Each base test image appears once in a
10,000-image stream, and severity increases in five blocks. All predictions
are recorded before the candidate receives the corresponding labels.

## Outputs

`source_data/analysis_summary.json` contains the simulation and tabular-stream
summaries. `source_data/Figure1_source_data.csv` contains simulation summaries
and stream trajectories. `source_data/Neural_stream_results.csv` contains one
row for each tabular dataset, network architecture, and initialization.
`source_data/cifar10_stream/` contains the corresponding clean CIFAR-10
summaries and complete paired sequences.
`source_data/cifar10c_stream/` contains the CIFAR-10-C summaries and paired
sequences for all 15 corruption types. The submitted PDF and SVG figures are
included in `figures/`; running `analysis/make_figure.py` also creates EPS,
TIFF, and PNG exports.

`analysis/proof_verification_report.json` records symbolic and finite-case checks of the algebra used in the proofs. These checks support the implementation; the cited probability theorems remain external mathematical results.

## Data and licensing

The repository does not redistribute CIFAR-10, CIFAR-10-C, or the River
datasets. Those datasets retain their original terms. The paired loss
sequences and numerical summaries included here are generated analysis
outputs. The analysis code is released under the MIT License.
