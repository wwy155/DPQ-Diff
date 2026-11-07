# DPQ-Diff

Code for *Distribution-Preserving Quantized Diffusion Models for Probabilistic Time Series Forecasting*.

## Setup

```bash
pip install -r requirements.txt
```

`run.sh` fetches the official [DiffPTS](https://github.com/wwy155/DiffPTS) implementation into `DiffPTS/`, and the datasets are downloaded into `data/` on first use.

## Run

```bash
bash run.sh ETTh1
```

The script trains a DiffPTS forecaster, quantizes it with DPQ-Diff to W8A8 and W4A8, and reports CRPS, MSE, and model size. Any of `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Electricity`, `ExchangeRate`, `ILI`, `SolarEnergy`, and `Traffic` can be given, optionally followed by a seed.

The supplementary material of the paper is [supplementary.pdf](supplementary.pdf).
