# HC-CDCT Reproducibility Repository

This repository is a modular version of the original single-file experiment script for the HC-CDCT watermarking experiment. It is organized for paper submission, result reproduction, and research data archiving.

## Repository purpose

The repository is intended to support a data/code availability statement in a manuscript. It separates the experiment into modules for:

1. loading CIFAR data;
2. building and loading the trained neural network;
3. embedding watermarks in DCT-domain non-DC coefficients;
4. extracting watermarks after pruning with compensation;
5. testing robustness under supervised fine-tuning attacks;
6. running the spatial-domain HC baseline;
7. saving BER, distortion, and optional accuracy results.

The original unmodified script is retained at:

```text
archive/original_single_file_experiment.py
```

## Directory structure

```text
hc_cdct_data_repository/
├── archive/
│   └── original_single_file_experiment.py
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── checkpoints/
├── external/
│   └── README.md
├── metadata/
│   ├── codebook.md
│   ├── dataset_description.json
│   └── repository_file_manifest.csv
├── results/
│   ├── tables/
│   │   └── pruning_results_template.csv
│   └── logs/
├── scripts/
│   ├── run_pruning_experiment.py
│   └── summarize_results.py
├── src/
│   └── hc_cdct/
│       ├── __init__.py
│       ├── data.py
│       ├── dct_utils.py
│       ├── experiment.py
│       ├── finetune_attack.py
│       ├── metrics.py
│       ├── model_factory.py
│       ├── pruning.py
│       ├── state_utils.py
│       ├── train_eval.py
│       └── watermark.py
├── DATA_AVAILABILITY.md
├── requirements.txt
└── environment.yml
```

## Required external project files


## Quick start

Install dependencies:

```bash
conda env create -f environment.yml
conda activate hc-cdct
```

Alternatively:

```bash
pip install -r requirements.txt
```

Edit `configs/default.yaml`, especially:

```yaml
model:
  checkpoint_path: "checkpoints/best.pth"
  name: "ResNet18"
experiment:
  embedding_layer: "layer1.1.conv2.weight"
```

Run the pruning robustness experiment:

```bash
python scripts/run_pruning_experiment.py --config configs/default.yaml
```

The script saves a CSV file similar to:

```text
results/tables/pruning_results.csv
```

Run the fine-tuning attack experiment:

```bash
python scripts/run_finetuning_attack.py --config configs/default.yaml
```

The fine-tuning script embeds the watermark into a fresh model copy for each learning rate, performs supervised fine-tuning on the training split, and extracts the watermark after each epoch. It saves results to:

```text
results/tables/finetuning_results.csv
```

The fine-tuning parameters are controlled in `configs/default.yaml`:

```yaml
finetuning_attack:
  schemes: ["hc_cdct", "hc_space"]
  learning_rates: [0.001, 0.005, 0.01]
  epochs: 5
  optimizer: "SGD"
```

## Notes for manuscript submission

For a journal submission, upload the repository to GitHub, OSF, Zenodo, Figshare, or an institutional repository. If the model checkpoint is too large, upload the checkpoint separately and provide its DOI or download link in `DATA_AVAILABILITY.md`.

Suggested statement:

> The source code, experimental configuration files, and result templates used in this study are available in the accompanying repository. The CIFAR datasets are downloaded through `torchvision` and are not redistributed in the repository. Trained model checkpoints are provided separately due to file size constraints.

## Reproducibility checklist

Before publishing the repository, confirm the following items:

- [ ] The exact model checkpoint used in the paper is included or linked.
- [ ] The random seed in `configs/default.yaml` is fixed.
- [ ] The final BER, distortion, pruning, and fine-tuning attack tables are placed under `results/tables/`.
- [ ] The environment file matches the actual machine used for experiments.
- [ ] Large datasets and generated checkpoints are excluded from Git unless intentionally archived.
