# CPU vs GPU for Machine Learning: Time, Energy, and Carbon Across Five Tasks

Ten model pairs, five tasks, one question: does GPU acceleration actually save energy and carbon, or does it just save wall-clock time. Each pair uses the same dataset, same seed, same hyperparameters, same evaluation code. The only thing that changes between the two runs is the device.

| Task | Report |
|---|---|
| Classification | [README-classification.md](README-classification.md) |
| Regression | [README-regression.md](README-regression.md) |
| Clustering | [README-clustering.md](| Classification | [README-classification.md]([README-classification.md](https://github.com/Atkiya/407_Project/blob/main/Clustering/README.md)) |) |
| Text Generation | [README-text-generation.md](README-text-generation.md) |
| Image Generation | [README-image-generation.md](README-image-generation.md) |

## Method

CPU and GPU experiments for each algorithm keep the dataset, preprocessing, initialization, batch size, epochs, learning rate, optimizer, and evaluation procedure fixed; only the device changes. Neural network models run in PyTorch. Energy and CO2e are measured with CodeCarbon during the same run that produces the timing numbers. GPU is an NVIDIA Tesla T4. CPU runs used four logical cores.

| Task | Algorithms | Dataset |
|---|---|---|
| Classification | MLP, CNN | EMNIST Digits |
| Regression | Linear Regression, MLP Regressor | YearPredictionMSD |
| Clustering | K-Means, GMM | EMNIST Digits |
| Text Generation | Vanilla RNN, LSTM | Tiny Shakespeare |
| Image Generation | VAE, DCGAN | EMNIST Digits |

## Results at a glance

| Task | Lighter model speedup | Heavier model speedup | Energy trend |
|---|---|---|---|
| Classification | 1.22x (MLP) | 4.16x (CNN) | MLP: GPU uses more energy. CNN: GPU uses less. |
| Regression | 1.48x (Linear) | 1.95x (MLP) | GPU uses more energy on both. |
| Clustering | 33.8x (K-Means) | 42.1x (GMM) | GPU uses far less energy on both. |
| Text Generation | 8.12x (RNN) | 31.1x (LSTM) | GPU uses less energy on both. |
| Image Generation | 24.8x (DCGAN) | 29.2x (VAE) | GPU uses far less energy on both. |

Speed and energy do not move together automatically. The MLP and both regression models get faster on GPU while using more energy to do it. Every other model gets faster and greener at the same time. What separates the two groups is not the task category, it is how much of the model's arithmetic can run as independent, parallel operations per sample.

## Why the gap exists: time complexity

| Model | Dominant cost | Parallel-friendliness |
|---|---|---|
| Linear Regression | O(n·d) | High per-operation, but too little total work per sample |
| MLP | O(n·d·H) | Moderate |
| K-Means | O(n·k·d) | Very high |
| GMM | O(n·k·d),  | Very high |
| CNN | O(n·k²·C·H·W) | Very high |
| RNN | O(n·T·H²) | Moderate, limited by sequential timesteps |
| LSTM | O(n·T·4H²) | Moderate to high |
| VAE | O(n × convolution cost) | Very high |
| DCGAN | O(n × 2 × convolution cost) | Very high |

Notation: n = samples, d = features, H = hidden size, T = sequence length, k = clusters/components (or kernel size for convolution), C = channels.

The models with the biggest constant factor on top of their complexity term (CNN, GMM, LSTM, VAE, DCGAN) are the ones where the GPU speedup is large enough to also cut energy and carbon. A bigger constant means more independent multiply-accumulate operations per sample, which is exactly what a GPU is built to run side by side. Linear Regression and the MLP sit on the other side: their per-sample arithmetic is too light for the GPU's fixed overhead (kernel launches, memory transfers, synchronization) to disappear into the workload.

## Four things that decide the outcome

**Computation per sample.** Going from RNN to LSTM nearly quadrupled the speedup (8.12x to 31.1x) just from adding gates. Going from Linear Regression to an MLP Regressor barely moved it (1.48x to 1.95x). The jump in per-sample arithmetic has to be large, not just present.

**How much time is actually saved.** A 1.22x speedup, like the classification MLP got, is not enough to offset a GPU using more than double the energy. The runtime cut has to be large before it pays for the GPU's higher power draw.

**Memory footprint.** Every GPU run in this project used more combined host-plus-device memory than its CPU counterpart, including the biggest wins (VAE, DCGAN, GMM, K-Means). GPU speed is rarely free of a memory cost.

**Carbon estimate reliability.** CodeCarbon estimates grid carbon intensity from the detected location of each session. That location was not always the same between a CPU run and its matching GPU run. United States, the Netherlands, and Australia all showed up across the twenty notebooks. Where energy and CO2e figures disagreed, the kWh number is the more dependable one to trust, since it does not depend on which grid region got detected.

## Decision table

| Task | Stick with CPU when | Move to GPU when |
|---|---|---|
| Classification | The network is shallow and energy matters more than a few seconds | Convolutional layers create enough parallel work to outrun the GPU's power draw |
| Regression | The model stays arithmetically simple, regardless of dataset size | Models grow substantially larger, or training runs repeatedly |
| Clustering | Sample count, dimensions, and cluster count stay genuinely small | Distance or probability math scales with data, which happens sooner than expected |
| Text Generation | Sequences or models are small, or generation runs one token at a time | Recurrent or gated models train on large batches |
| Image Generation | Only small tests, debugging, or preprocessing are needed | Full convolutional generative training is running |

## Two algorithms per task, on purpose

Each task pairs a lighter and a heavier implementation of the same problem, not to rank one algorithm above the other, but to isolate what algorithmic complexity alone does to the CPU-GPU trade-off. K-Means was never meant to compete with GMM, and the RNN was never meant to compete with the LSTM. Keeping the dataset and hardware fixed while only the model complexity changes is what makes the pattern in this project visible instead of assumed.



# Libraries Used

Every library that appears across the twenty notebooks, grouped by what it is doing in this project.

## Core deep learning

**PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)**
Builds and trains all the neural network models: the MLP and CNN classifiers, the Linear Regression and MLP regressors, the RNN and LSTM, and the VAE and DCGAN. Handles the forward pass, backpropagation, and the actual switch between CPU and GPU execution through the `device` argument, which is the whole point of this project.

**torchvision (`datasets`, `transforms`, `utils.make_grid`)**
Downloads and loads the EMNIST Digits dataset, applies preprocessing transforms (normalization, tensor conversion), and arranges generated or reconstructed images into a grid for visual inspection in the VAE and DCGAN notebooks.

**torch.utils.data (`Dataset`, `DataLoader`, `TensorDataset`)**
Wraps the raw data into batches and feeds them to the model during training. `DataLoader` is what actually produces the batch-size-128 or batch-size-1024 chunks referenced throughout the results tables.

## Clustering

**pomegranate (`pomegranate.gmm.GeneralMixtureModel`, `pomegranate.distributions.Normal`)**
Implements the Gaussian Mixture Model used in the clustering task. This library runs on both CPU and GPU through PyTorch tensors underneath, which is why the GMM notebooks can measure a genuine device comparison rather than switching to a different algorithm implementation for GPU.

**scikit-learn (`sklearn.preprocessing.StandardScaler`, `sklearn.decomposition.PCA`, `sklearn.metrics`)**
`StandardScaler` normalizes pixel features before clustering. `PCA` reduces the EMNIST images from 784 raw pixels down to 50 dimensions before GMM fitting. `sklearn.metrics` supplies the evaluation functions: `adjusted_rand_score`, `normalized_mutual_info_score`, and the regression metrics `mean_squared_error`, `mean_absolute_error`, `median_absolute_error`, and `r2_score`.

K-Means itself is not from scikit-learn in these notebooks. It is a custom PyTorch implementation, which is what lets it run on GPU without a separate GPU-only clustering library.

**scipy (`scipy.optimize.linear_sum_assignment`)**
Solves the assignment problem between predicted cluster labels and true digit labels, since K-Means and GMM produce arbitrary cluster indices that need to be matched to the actual 0 to 9 digit classes before accuracy can be computed.

## Measurement and monitoring

**CodeCarbon (`codecarbon.EmissionsTracker`)**
Tracks energy consumption and estimates CO2e emissions during training and inference by reading CPU, GPU, and RAM power draw and combining it with a regional grid carbon-intensity estimate. This is the source of every energy and carbon figure in the report.

**psutil**
Reads live process memory usage. Used to record peak process RAM during training and inference, reported as "peak process RAM" in every results table.

## Data handling and numerical computation

**NumPy (`numpy`)**
Backs most of the array math outside of PyTorch tensors: label arrays, intermediate calculations, and conversions between PyTorch tensors and plain arrays for use with scikit-learn and scipy.

**pandas (`pandas`)**
Builds the results tables that get saved to CSV at the end of each notebook (for example `cnn_emnist_cpu_result.csv`), and loads the paired CPU and GPU result files back in for the side-by-side comparison cells.

**matplotlib (`matplotlib.pyplot`)**
Plots training curves, confusion matrices, and generated or reconstructed image grids.

## Standard library and utility

**os, sys, subprocess, importlib.util**
Used together in an `ensure_package` helper at the top of most notebooks that checks whether CodeCarbon (and occasionally other packages) is already installed and installs it with pip if not.

**time**
Provides `time.perf_counter()` for all the wall-clock timing measurements: training time, inference time, generation time.

**threading**
Runs a background thread that polls `psutil` at a fixed interval during training, which is how peak RAM is captured even while the main thread is busy training.

**random, platform, warnings, gc, copy, math, glob, zipfile, urllib.request**
Standard housekeeping: `random` and the PyTorch/NumPy seed calls fix the random seed for reproducibility, `platform` reports CPU core count and system info, `warnings` suppresses noisy library warnings, `gc` forces garbage collection between CPU and GPU runs to avoid memory carryover, and `copy`, `math`, `glob`, `zipfile`, and `urllib.request` handle small preprocessing and file-handling tasks such as downloading and unzipping the YearPredictionMSD dataset.

**pathlib.Path**
Handles file paths for saving and loading the CSV result files across notebooks, instead of manual string concatenation.

**collections.Counter**
Counts character or token frequency while building the vocabulary for the Tiny Shakespeare character-level text-generation models.

## Summary table

| Library | Role |
|---|---|
| torch | Model definition, training, backpropagation, CPU/GPU device switching |
| torchvision | Dataset loading, image transforms, image grid visualization |
| pomegranate | Gaussian Mixture Model implementation (CPU and GPU) |
| scikit-learn | Preprocessing (scaling, PCA), evaluation metrics |
| scipy | Cluster-to-label assignment matching |
| codecarbon | Energy and CO2e measurement |
| psutil | Peak RAM tracking |
| numpy | General array math |
| pandas | Results tables, CSV read/write |
| matplotlib | Plots and generated image grids |
| threading | Background memory polling during training |
| time | Wall-clock timing |
