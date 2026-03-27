# Measurement Error Framework for Spatial Omics Inference

**Core claim:** Spatial imputation methods create false cell-cell interactions due to correlated reconstruction errors. We provide a regression-calibrated correction and reliability framework to detect and mitigate these artifacts.

## Framework

```
Poisson Measurement Model → Regression Calibration (Bias Correction) → Panel Perturbation Stability → Reliability Certificate
```

## Three Components

| Component | Role | Status |
|---|---|---|
| Poisson measurement model | Data-generating model for count data under gene-panel sparsity | Foundation |
| Regression calibration | Corrects CCI inflation caused by spatially correlated imputation errors | Core contribution |
| Panel perturbation stability | Quantifies how sensitive biological conclusions are to panel composition | Validation |

## Project Structure

```
Spatial-Omics/
├── src/
│   ├── data_loader.py          # Load and preprocess spatial omics data
│   ├── panel_simulation.py     # Simulate sparse gene panels
│   ├── imputation/
│   │   ├── __init__.py
│   │   ├── tangram_runner.py   # Reference-based (Tangram)
│   │   ├── gimvi_runner.py     # Deep generative (gimVI)
│   │   ├── gnn_runner.py       # GNN-based spatial imputation
│   │   └── diffusion_runner.py # Diffusion-based imputation
│   ├── cci_inference.py        # CCI computation (naive + calibrated)
│   ├── regression_calibration.py # Bias correction for CCI statistics
│   ├── stability_analysis.py   # Panel perturbation stability
│   ├── evaluation.py           # 5 key figures generation
│   └── utils.py                # Shared utilities
├── notebooks/
│   └── pilot_experiment.ipynb  # End-to-end pilot notebook
├── configs/
│   └── pilot_config.yaml       # Experiment configuration
├── tests/
│   └── test_cci.py             # Unit tests
├── run_pilot.py                # Main entry point
├── requirements.txt
└── README.md
```

## Datasets

- **Xenium breast cancer** (GSE243280 or 10x public datasets)
- **scRNA-seq reference** (matched or Human Breast Cell Atlas)

## Quick Start

```bash
pip install -r requirements.txt
python run_pilot.py --spatial path/to/xenium.h5ad --reference path/to/scrna.h5ad --panel-size 300
```

## Go / No-Go Criteria

The pilot succeeds if:
- CCI inflation > 20% across >= 2 imputation methods
- Regression calibration removes > 50% of false positives
- Reliability-error correlation > 0.5

## Target Venues

- **Primary:** Genome Biology, Bioinformatics
- **Stretch:** Nature Methods
- **Theory-heavy:** Biostatistics, Biometrics
