# Graph Convolutional Network in MLX

An example of [GCN](https://arxiv.org/pdf/1609.02907.pdf%EF%BC%89) implementation with MLX. Other examples are available <a href="https://github.com/ml-explore/mlx-examples">here</a>.

The actual benchmark on **M1 Pro**, **M2 Ultra**, **M3 Max** and **Tesla V100**s is explained in <a href="https://towardsdatascience.com/mlx-vs-mps-vs-cuda-a-benchmark-c5737ca6efc9">this Medium article</a>.

### Install env and requirements

```
CONDA_SUBDIR=osx-arm64 conda create -n mlx python=3.10 numpy pytorch scipy requests -c conda-forge

conda activate mlx
pip install mlx
```

### Run
To try the model, just run the `main.py` file. This will download the Cora dataset, run the training and testing. The actual MLX code is located in `main.py`, whereas the PyTorch equivalent is in `main_torch.py`.

```
python main.py
```

### Run benchmark
To run the benchmark on CUDA device, a new env needs to be set up without the `CONDA_SUBDIR=osx-arm64` prefix, to be in i386 mode and not arm. For all other experiments on arm and Apple Silicon, just use the env created previously.
```
python benchmark.py --experiment=[ mlx | torch_mps | torch_cpu | torch_cuda ]
```

### Process benchmark figure
This needs to install additional packages: `matplotlib` and `scikit-learn`.

```
python viz.py
```

<img src="bench.png" width="80%" alt="Benchmark of GCN on MLX, MPS, CPU, CUDA">
