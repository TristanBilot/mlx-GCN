from argparse import ArgumentParser

import numpy as np


def run(args):
    if args.experiment == 'mlx':
        from main import main as mlx_main

    else:
        import torch
        from main_torch import main as torch_main

        if args.experiment == 'torch_mps':
            device = torch.device("mps")
        elif args.experiment == 'torch_cuda':
            device = torch.device("cuda")
        elif args.experiment == 'torch_cpu':
            device = torch.device("cpu")

    times = []
    for i in range(args.nb_experiment):
        if args.experiment == 'mlx':
            t = mlx_main(args)
        else:
            t = torch_main(args, device)
        times.append(t)

    mean = np.mean(times)
    std = np.std(times)
    
    print("")
    print(f"Mean training epoch time: {mean:.5f} seconds")
    print(f"Std training epoch time: {std:.5f} seconds")


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, default="mlx") # ["torch_cpu", "torch_mps", "torch_cuda", "mlx"]
    parser.add_argument("--nb_experiment", type=int, default=5)

    parser.add_argument("--nodes_path", type=str, default="cora/cora.content")
    parser.add_argument("--edges_path", type=str, default="cora/cora.cites")
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--nb_layers", type=int, default=2)
    parser.add_argument("--nb_classes", type=int, default=7)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--patience", type=int, default=1000000) # we do not use patience in benchmark
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    run(args)
