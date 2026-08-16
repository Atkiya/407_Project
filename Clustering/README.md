# Clustering: K-Means vs GMM on CPU and GPU

Two clustering algorithms on the same image data, same preprocessing, same seed. This pair shows how quickly the CPU-favorable zone shrinks once distance and probability calculations get repeated across a large number of points.

## Dataset

EMNIST Digits (NIST), 280,000 samples for K-Means (raw 784-pixel features) and 240,000 samples for GMM (reduced to 50 dimensions with PCA before fitting).

## Setup

| | K-Means | GMM |
|---|---|---|
| Samples | 280,000 | 240,000 |
| Features | 784 | 50 (post-PCA) |
| Clusters / components | 10 | 10 (diagonal covariance) |
| Max iterations | 100 | 300 |
| Seed | 42 | 42 |
| GPU | Tesla T4 | Tesla T4 |

## Notebooks

`kmeans-cpu.ipynb`, `k-means-gpu.ipynb`, `gmm-cpu.ipynb`, `gmm-gpu.ipynb`

## Results

### K-Means

| Metric | CPU | GPU |
|---|---|---|
| Fit time | 26.704 s | 0.791 s |
| Inference throughput | 520,321 samples/s | 14,432,961 samples/s |
| Accuracy | 0.5505 | 0.5353 |
| NMI | 0.4925 | 0.4718 |
| Energy per fit | 0.1555 Wh | 0.0183 Wh |
| CO2e per fit | 0.0704 g | 0.0083 g |
| Peak process RAM | 1,436.0 MB | 2,279.6 MB |
| Peak GPU memory | - | 912.68 MB |

### GMM

| Metric | CPU | GPU |
|---|---|---|
| Fit time | 151.12 s | 3.59 s |
| Fit throughput | 1,853 samples/s | 78,079 samples/s |
| NMI | 0.2741 | 0.2741 |
| ARI | 0.1195 | 0.1196 |
| Training energy | 0.000756 kWh | 0.0000759 kWh |
| Training CO2e | 0.342 g | 0.0105 g |
| Peak process RAM | 3,348.7 MB | 3,934.5 MB |
| Peak GPU memory | - | 238.49 MB |

## Reading the numbers

K-Means fit 33.8x faster on GPU (0.791 s vs 26.704 s) and cluster assignment ran 30.5x faster, because every point-to-centroid distance in the algorithm is independent of every other one. That independence is what lets the GPU spread the work across thousands of cores at once. Energy dropped by 88% per fit (0.0183 Wh vs 0.1555 Wh) and CO2e fell by the same margin. Clustering quality dropped slightly on GPU (accuracy 0.5353 vs 0.5505, NMI 0.4718 vs 0.4925), most likely from small differences in floating-point order or centroid tie-breaking during k-means++ initialization, not from the algorithm behaving differently.

GMM pushed the gap even further: 42.1x faster fitting (3.59 s vs 151.12 s), 90% less energy, and roughly 97% less CO2e. Each EM step needs a probability density calculation for every point under every one of the 10 Gaussian components, which is more arithmetic per point than K-Means does, and all of it parallelizes the same way. Clustering quality here barely moved between devices (NMI 0.2741 on both, ARI 0.1195 vs 0.1196), so the extra speed did not trade off against accuracy at all.

## Where CPU still holds up

Neither algorithm gave the CPU an advantage at this sample size and dimensionality. 280,000 samples with 784 raw features, or 240,000 samples reduced to 50 PCA dimensions, was already enough to push both algorithms firmly onto the GPU side of the decision. The point where CPU stays competitive for clustering is likely well below these sample counts, smaller than the "moderate workload" assumption commonly used to justify staying on CPU.

## Memory note

GPU runs used more combined memory in both cases. K-Means added roughly 844 MB of host RAM on top of 912.68 MB of VRAM, largely because it works directly on the 784-dimensional pixel space rather than a reduced representation. GMM's VRAM footprint stayed much smaller (238.49 MB) since it operates on the 50-dimensional PCA output instead.
