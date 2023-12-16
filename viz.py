import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Color map for each class
cora_label_to_color_map = {0: "red", 1: "blue", 2: "green",
                           3: "orange", 4: "yellow", 5: "pink", 6: "gray"}


def benchmark_plot():
    backends = [
        '  CPU', '  MPS', '  MLX',
        ' CPU', ' MPS', ' MLX',
        'CPU', 'MPS', 'MLX',
        'CUDA (PCIe)', 'CUDA (NVLINK)',
    ]
    times = [
         45.58, 21.10, 9.02,
        9.31, 7.19,  5.8,
        7.07, 4.8,  4.72,
        3.83, 3.51,
    ]

    # Setting colors
    colors = [
        '#FFCC99' for _ in range(3)] \
        + ['#9999FF' for _ in range(3)] \
        + ['#FF9999' for _ in range(3)] \
        + ['skyblue' for _ in range(2)]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(backends, times, color=colors, edgecolor='black')

    plt.xlabel('Mean time per epoch (ms)')
    plt.title('Benchmarking MLX and PyTorch Backends')
    plt.gca().invert_yaxis()

    for index, value in enumerate(times):
        plt.text(value, index, f' {value} ms', va='center')

    # Adding a legend
    legend_elements = [
        plt.Line2D([0], [0], color='#FFCC99', lw=4, label='M1 Pro'),
        plt.Line2D([0], [0], color='#9999FF', lw=4, label='M2 Ultra'),
        plt.Line2D([0], [0], color='#FF9999', lw=4, label='M3 Max'),
        plt.Line2D([0], [0], color='skyblue', lw=4, label='Tesla V100')
    ]
    plt.legend(handles=legend_elements, loc='lower right')


    plt.savefig("bench.png")


def visualize_embedding_tSNE(labels, out_features, num_classes):
    """ https://github.com/gordicaleksa/pytorch-GAT """
    t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(out_features)

    plt.figure()
    for class_id in range(num_classes):
        plt.scatter(t_sne_embeddings[node_labels == class_id, 0],
                    t_sne_embeddings[node_labels == class_id, 1], s=20,
                    color=cora_label_to_color_map[class_id],
                    edgecolors='black', linewidths=0.15)

    plt.axis("off")
    plt.title("t-SNE projection of the learned features")
    plt.show()


def visualize_validation_performance(val_acc, val_loss):
    f, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    axs[0].plot(val_loss, linewidth=2, color="red")
    axs[0].set_title("Validation loss")
    axs[0].set_ylabel("Cross Entropy Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].grid()

    axs[1].plot(val_acc, linewidth=2, color="red")
    axs[1].set_title("Validation accuracy")
    axs[1].set_ylabel("Acc")
    axs[1].set_xlabel("Epoch")
    axs[1].grid()

    plt.show()


if __name__ == '__main__':
    benchmark_plot()
