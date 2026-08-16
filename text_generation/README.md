# Text Generation: Vanilla RNN vs LSTM on CPU and GPU

A character-level RNN and LSTM trained on the same corpus, same seed, same hyperparameters. This pair isolates what the extra gate math in an LSTM costs on each device, since both models process the same sequences the same number of times.

## Dataset

Tiny Shakespeare (Karpathy's char-rnn repository), full text, character-level tokenization.

## Setup

| | Vanilla RNN | LSTM |
|---|---|---|
| Parameters | 583,873 | 2,210,497 |
| Sequence length | 128 | 128 |
| Hidden dimension | 384 | 384 |
| Embedding dimension | 256 | 256 |
| Recurrent layers | 2 | 2 |
| Epochs | 5 | 5 |
| Batch size | 64 | 64 |
| Learning rate | 0.001 | 0.001 |
| GPU | Tesla T4 | Tesla T4 |

## Notebooks

`rnn-cpu.ipynb`, `rnn-gpu.ipynb`, `lstm-cpu.ipynb`, `lstm-gpu.ipynb`

## Results

### Vanilla RNN

| Metric | CPU | GPU |
|---|---|---|
| Training time | 237.18 s | 29.22 s |
| Training throughput | 21,162 chars/s | 171,801 chars/s |
| Test inference time | 0.765 s | 0.264 s |
| Generation time (500 chars) | 3.689 s | 1.880 s |
| Test perplexity | 5.2412 | 5.2398 |
| Training energy | 0.001168 kWh | 0.000663 kWh |
| Training CO2e | 0.529 g | 0.300 g |
| Generation energy | 0.0000184 kWh | 0.0000290 kWh |

### LSTM

| Metric | CPU | GPU |
|---|---|---|
| Training time | 782.83 s | 25.14 s |
| Training throughput | 6,412 chars/s | 199,656 chars/s |
| Test inference time | 2.539 s | 0.124 s |
| Generation time (500 chars) | 8.764 s | 1.933 s |
| Test perplexity | 5.1796 | 5.1875 |
| Training energy | 0.003865 kWh | 0.000564 kWh |
| Training CO2e | 2.121 g | 0.255 g |
| Generation energy | 0.0000438 kWh | 0.0000425 kWh |

## Reading the numbers

The RNN already benefits from GPU training despite its step-by-step structure: 8.12x faster (29.22 s vs 237.18 s), 43% less training energy, and roughly the same drop in CO2e. Each timestep still runs a batch of matrix multiplies that parallelize fine even though the timesteps themselves have to run in order.

The LSTM is where the gap opens up. Its four gates roughly quadruple the arithmetic done at every timestep compared to a plain RNN, and that shows up almost entirely as more parallel matrix work rather than more sequential steps. Training sped up 31.1x (25.14 s vs 782.83 s), nearly four times the RNN's own speedup, and training energy dropped 85% (0.000564 kWh vs 0.003865 kWh). The GPU-trained LSTM (25.14 s) finished faster in absolute terms than the GPU-trained RNN (29.22 s), even though it is doing more math per step, while on CPU the LSTM took more than three times as long as the RNN. The CPU is what makes the extra gates expensive; the GPU absorbs them almost for free.

Perplexity stayed close across devices for both models (RNN: 5.2412 vs 5.2398; LSTM: 5.1796 vs 5.1875), so none of the speed difference traded away model quality.

## Where the pattern breaks

Single-sequence generation does not scale the same way training does. RNN generation was only 1.96x faster on GPU and actually used 58% more energy there (0.0000290 kWh vs 0.0000184 kWh). Autoregressive generation runs one token at a time with a batch size of one, which removes most of the parallel work a GPU depends on. LSTM generation showed the same softened pattern, 4.53x faster but with energy use nearly identical between devices (0.0000425 kWh vs 0.0000438 kWh). Training and interactive generation should be judged separately when picking a device for a language model.
