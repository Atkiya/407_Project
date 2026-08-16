# Regression: Linear Regression vs MLP Regressor on CPU and GPU

Two regressors on the same large tabular dataset, same seed, same hyperparameters. The point of this pair is to see whether a bigger dataset alone is enough to make a GPU pay off, or whether the model itself still has to do enough work per sample.

## Dataset

YearPredictionMSD (UCI Machine Learning Repository). 463,715 training rows, 51,630 test rows, 90 features.

## Setup

| | Linear Regression | MLP Regressor |
|---|---|---|
| Parameters | 91 | 56,321 |
| Architecture | 90 -> 1 | 90 -> 256 -> 128 -> 1, ReLU |
| Epochs | 20 | 20 |
| Batch size | 1024 | 1024 |
| Learning rate | 0.001 | 0.001 |
| GPU | Tesla T4 | Tesla T4 |

## Notebooks

`linear-regression-cpu.ipynb`, `linear-regression-gpu.ipynb`, `mlp-regressor-cpu.ipynb`, `mlp-regressor-gpu.ipynb`

## Results

### Linear Regression

| Metric | CPU | GPU |
|---|---|---|
| Training time | 115.29 s | 77.92 s |
| Training throughput | 80,445 rows/s | 119,017 rows/s |
| RMSE | 9.5152 yrs | 9.5152 yrs |
| R² | 0.2312 | 0.2312 |
| Training energy | 0.000557 kWh | 0.000903 kWh |
| Training CO2e | 0.149 g | 0.258 g |

### MLP Regressor

| Metric | CPU | GPU |
|---|---|---|
| Training time | 158.28 s | 81.30 s |
| Training throughput | 58,593 rows/s | 114,071 rows/s |
| RMSE | 9.1116 yrs | 9.1680 yrs |
| R² | 0.2950 | 0.2863 |
| Training energy | 0.000716 kWh | 0.000937 kWh |
| Training CO2e | 0.393 g | 0.424 g |

## Reading the numbers

Linear Regression only trains one weighted sum of 90 inputs per row. Even across 463,715 rows, that is too little arithmetic per row to load a GPU properly. The GPU finished 1.48x faster but used 62% more energy (0.000903 kWh vs 0.000557 kWh) doing it, and result quality was identical on both devices (RMSE 9.5152, R² 0.2312), since gradient descent on this problem converges to the same solution regardless of hardware.

The MLP regressor adds two hidden layers and pushes the speedup up to 1.95x, roughly a third higher than Linear Regression's gain. That confirms the direction: more per-sample computation gives the GPU more to work with. But 56,321 parameters is still small next to the CNN or DCGAN in this project, so the GPU still used about 31% more energy (0.000937 kWh vs 0.000716 kWh) to get there. RMSE and R² moved slightly between devices (9.1116 vs 9.1680, R² 0.2950 vs 0.2863), a normal amount of run-to-run variation for a stochastically trained model, not a device effect.

Dataset size on its own does not decide this. Both models trained on the same 463,715 rows, and the outcome still depended entirely on how much math each model does per row.

## Where the crossover would be

Both regressors here still land on the CPU side of the energy comparison. The trend across the two models points toward a crossover once the network is large enough that the GPU speedup climbs well past 2x, similar to what happens with the CNN in the classification pair.
