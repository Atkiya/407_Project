# Image Generation: VAE vs DCGAN on CPU and GPU

A convolutional VAE and a DCGAN trained on the same image dataset, same seed, same fixed latent vectors for sampling. Both models lean entirely on convolution, so this pair shows what happens at the far end of the GPU-friendly spectrum.

## Dataset

EMNIST Digits (NIST), 28x28 grayscale images.

## Setup

| | VAE | DCGAN |
|---|---|---|
| Parameters | 370,945 | Generator 778,305 / Discriminator 138,817 |
| Latent dimension | 32 | 100 |
| Epochs | 10 | 10 |
| Batch size | 128 | 128 |
| Learning rate | 0.0002 | 0.0002 |
| GPU | Tesla T4 | Tesla T4 |

## Notebooks

`vae-cpu.ipynb`, `vae-gpu.ipynb`, `dcgan-cpu.ipynb`, `dcgan-gpu.ipynb`

## Results

### VAE

| Metric | CPU | GPU |
|---|---|---|
| Training time | 568.422 s | 19.445 s |
| Training throughput | 2,111.1 images/s | 61,713.9 images/s |
| Reconstruction time | 7.5633 s | 0.2051 s |
| Generation time (10,000 images) | 1.1318 s | 0.0276 s |
| Test loss | 140.1085 | 140.1012 |
| Training energy | 2.875 Wh | 0.494 Wh |
| Training CO2e | 0.399 g | 0.108 g |
| Peak process RAM | 1,608.26 MB | 2,393.28 MB |
| Peak GPU memory | - | 896.79 MB |

### DCGAN

| Metric | CPU | GPU |
|---|---|---|
| Training time | 2,618.71 s | 105.62 s |
| Training throughput | 458.2 images/s | 11,361.7 images/s |
| Generation time (10,000 images) | 2.1614 s | 0.2244 s |
| Final generator loss | 1.9346 | 1.8947 |
| Final discriminator loss | 0.6138 | 0.6631 |
| Training energy | 12.785 Wh | 2.423 Wh |
| Training CO2e | 3.422 g | 1.097 g |
| Peak process RAM | 1,826.8 MB | 2,429.9 MB |
| Peak GPU memory | - | 982.3 MB |

## Reading the numbers

The VAE's encoder and decoder are both convolutional stacks, and that structure gave the GPU the largest speedup in the whole project: 29.2x on training (19.445 s vs 568.422 s), 36.9x on reconstruction, and 41x on sampling new images. Training energy fell 82.8% (0.494 Wh vs 2.875 Wh) and CO2e fell 72.9%. Test loss barely moved between devices (140.1012 vs 140.1085, a difference of 0.0073), so the two runs converged to the same model, just at very different speeds.

DCGAN trains a generator and a discriminator at once, doubling the convolutional work per step compared to a single network, and it needed the most wall-clock time of any model here: 43.6 minutes on CPU against 105.62 seconds on GPU, a 24.8x speedup. Energy dropped 81.0% (2.423 Wh vs 12.785 Wh). The final generator and discriminator losses stayed close between devices (generator 1.8947 vs 1.9346, discriminator 0.6631 vs 0.6138), so image quality did not depend on which device trained the model.

One caveat on the carbon numbers for DCGAN specifically: the CPU run was logged under an estimated Netherlands grid and the GPU run under a United States grid, so the CO2e comparison (67.9% lower on GPU) carries more noise from that location difference than the energy comparison does. The energy figures are the more reliable read here.

## Memory note

Both models used more combined memory on GPU once host RAM and VRAM are added together (VAE: 2,393.28 MB host + 896.79 MB VRAM vs 1,608.26 MB on CPU; DCGAN: 2,429.9 MB host + 982.3 MB VRAM vs 1,826.8 MB on CPU). That extra memory is small next to the time and energy saved, but it is the one line item where CPU comes out ahead.
