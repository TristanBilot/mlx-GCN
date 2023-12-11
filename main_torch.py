from argparse import ArgumentParser

import torch
import torch.nn as nn

from datasets import download_cora, load_data, train_val_test_mask


class GCNLayer(nn.Module):
    def __init__(self, x_dim, h_dim, bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(x_dim, h_dim))))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(h_dim,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias

        return torch.mm(adj, x)


class GCN(nn.Module):
    def __init__(self, x_dim, h_dim, out_dim, nb_layers=2, dropout=0.5, bias=True):
        super(GCN, self).__init__()

        layer_sizes = [x_dim] + [h_dim] * nb_layers + [out_dim]
        self.gcn_layers = nn.Sequential(*[
            GCNLayer(in_dim, out_dim, bias)
            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])
        ])
        self.dropout = nn.Dropout(p=dropout)

    def initialize_weights(self):
        self.gcn_1.initialize_weights()
        self.gcn_2.initialize_weights()

    def forward(self, x, adj):
        for layer in self.gcn_layers[:-1]:
            x = torch.relu(layer(x, adj))
            x = self.dropout(x)
        
        x = self.gcn_layers[-1](x, adj)
        return x


def to_torch(device, x, y, adj, train_mask, val_mask, test_mask):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    adj = torch.tensor(adj, dtype=torch.float32, device=device)
    train_mask = torch.tensor(train_mask, device=device)
    val_mask = torch.tensor(val_mask, device=device)
    test_mask = torch.tensor(test_mask, device=device)
    return x, y, adj, train_mask, val_mask, test_mask


def eval_fn(x, y):
    return torch.mean((torch.argmax(x, axis=1) == y).float())

def main(args, device):

    # Data loading
    download_cora()

    x, y, adj = load_data(args)
    train_mask, val_mask, test_mask = train_val_test_mask(y, args.nb_classes)

    x, y, adj, train_mask, val_mask, test_mask = \
        to_torch(device, x, y, adj, train_mask, val_mask, test_mask)

    gcn = GCN(
        x_dim=x.shape[-1],
        h_dim=args.h_dim,
        out_dim=args.nb_classes,
        nb_layers=args.nb_layers,
        dropout=args.dropout,
        bias=args.bias,
    ).to(device)
    

    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    

    best_val_loss = float("inf")
    cnt = 0

    # Training loop
    for epoch in range(args.epochs):

        optimizer.zero_grad()
        gcn.train()

        y_hat = gcn(x, adj)
        loss = loss_fn(y_hat[train_mask], y[train_mask])
        
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            gcn.eval()
            val_loss = loss_fn(y_hat[val_mask], y[val_mask])
            val_acc = eval_fn(y_hat[val_mask], y[val_mask])

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                cnt = 0
            else:
                cnt += 1
                if cnt == args.patience:
                    break

        print(
            " | ".join(
                [
                    f"Epoch: {epoch:3d}",
                    f"Train loss: {loss.item():.3f}",
                    f"Val loss: {val_loss.item():.3f}",
                    f"Val acc: {val_acc.item():.2f}",
                ]
            )
        )

    # Test
    test_y_hat = gcn(x, adj)
    test_loss = loss_fn(y_hat[test_mask], y[test_mask])
    test_acc = eval_fn(y_hat[test_mask], y[test_mask])

    print(f"Test loss: {test_loss.item():.3f}  |  Test acc: {test_acc.item():.2f}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--nodes_path", type=str, default="cora/cora.content")
    parser.add_argument("--edges_path", type=str, default="cora/cora.cites")
    parser.add_argument("--h_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--nb_layers", type=int, default=2)
    parser.add_argument("--nb_classes", type=int, default=7)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS.")
    else:
        device = torch.device("cpu")
        print ("MPS device not found.")

    main(args, device)
