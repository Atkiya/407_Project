# Classification: MLP vs CNN on CPU and GPU

Two classifiers trained on the same dataset, same seed, same hyperparameters, only the device changes. The goal is to see how much a model's architecture, not its task label, decides whether a GPU is worth using.

## Dataset

EMNIST Digits (NIST). Full split used, no sampling.

## Setup

| | MLP | CNN |
|---|---|---|
| Parameters | 235,146 | 421,642 |
| Epochs | 10 | 10 |
| Batch size | 128 | 128 |
| Learning rate | 0.001 | 0.001 |
| GPU | Tesla T4 | Tesla T4 |

## Notebooks

`mlp-cpu.ipynb`, `mlp-gpu.ipynb`, `cnn-cpu.ipynb`, `cnn-gpu.ipynb`

## Results

### MLP

| Metric | CPU | GPU |
|---|---|---|
| Training time | 346.0 s | 284.1 s |
| Training throughput | 6,936 ex/s | 8,448 ex/s |
| Inference time | 4.82 s | 4.29 s |
| Accuracy | 98.97% | 98.98% |
| Training energy | 0.00157 kWh | 0.00349 kWh |
| Peak process RAM | 769 MB | 1,457 MB |
| Peak GPU memory | - | 22.1 MB |

### CNN

| Metric | CPU | GPU |
|---|---|---|
| Training time | 1,577.6 s | 379.0 s |
| Training throughput | 1,521 ex/s | 6,332 ex/s |
| Inference time | 13.34 s | 4.99 s |
| Accuracy | 99.56% | 99.45% |
| Training energy | 0.00788 kWh | 0.00527 kWh |
| Peak process RAM | 874 MB | 1,617 MB |
| Peak GPU memory | - | 82.8 MB |

## Reading the numbers

The MLP gained only 1.22x from the GPU, and that small gain cost more than double the energy (0.00349 kWh vs 0.00157 kWh). Fully connected layers on a small model do not give the GPU enough parallel work per batch to offset the fixed cost of kernel launches and CPU-GPU transfers. Accuracy stayed flat either way (98.97% vs 98.98%), so nothing was gained on the quality side either.

The CNN flips this. Convolution repeats the same small multiply-and-add operation across every pixel and every channel, which is exactly the kind of workload a GPU is built for. Training dropped from 1,577.6 s to 379.0 s (4.16x), and for the first time in this comparison the GPU also used less total energy (0.00527 kWh vs 0.00788 kWh, a 33% cut) despite running at higher instantaneous power. The 0.10-point accuracy gap between devices (99.56% vs 99.45%) falls inside normal floating-point variation and is not a real quality difference.

Parameter count by itself does not predict the outcome. The CNN has less than twice the parameters of the MLP but delivers over three times the speedup, because convolution creates far more repeated, independent arithmetic per parameter than a dense layer does.

## Inference note

CNN inference favors GPU on speed (2.67x) but not on energy (GPU used about 3.5% more). Inference skips backpropagation, so there is less computation to amortize the GPU's fixed overhead against. Device choice for inference should depend on whether request volume and latency matter more than per-request energy.
