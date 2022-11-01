from pickletools import optimize
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class SampleNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(SampleNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def main():
    # Coraは論文の被引用ネットワークを示すデータセットらしい
    # なのでグラフは1つのみ, 半教師あり学習を行う
    dataset = Planetoid(root='tmp/Cora', name='Cora')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = dataset.num_node_features
    hidden_channels = 16
    out_channels = dataset.num_classes
    model = SampleNet(in_channels, out_channels, hidden_channels).to(device)
    data: Data = dataset[0].to(device) # 1個のグラフで学習
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("--- training start ---")
    # train
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    print("--- evaluation start ---")
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))


if __name__ == "__main__":
    main()