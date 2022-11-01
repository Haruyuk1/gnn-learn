# %%
# import
import torch
import torch.nn.functional as F
from sklearn import datasets
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, aggr
from tqdm import tqdm

# %%
# データ準備
mnist_dataset = GNNBenchmarkDataset(root="tmp/mnist", name="MNIST")
mnist_dataset = mnist_dataset.shuffle()
mnist_size = len(mnist_dataset)
train_dataset = mnist_dataset[:int(mnist_size*0.8)]
test_dataset = mnist_dataset[int(mnist_size*0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

print(train_dataset[1].x.shape, train_dataset[1].pos.shape)

# %%
# モデルの用意
class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
        self.conv3 = GCNConv(hidden_channels*2, hidden_channels*4)
        self.conv4 = GCNConv(hidden_channels*4, out_channels)
        self.aggr = aggr.MeanAggregation()

    def forward(self, data: Batch):
        x = torch.concat([data.x, data.pos], dim=1)
        edge_index = data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x)
        x = self.conv4(x, edge_index)

        # readout per graph
        hg = self.aggr(x, data.batch)
        
        return hg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 3
hidden_channels = 16
out_channels = train_dataset.num_classes
model = Model(in_channels, hidden_channels, out_channels).to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

# %%
num_epoch = 100
for epoch in range(1, num_epoch+1):
    # train
    print(f"--- training {epoch}/{num_epoch} ---")
    model.train()
    correct = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        _, pred = out.max(dim=1)
        correct += float(pred.eq(batch.y).sum().item())
        loss = loss_func(out, batch.y)
        loss.backward()
        optimizer.step()
    acc = correct / len(train_dataset)
    print('Train Accuracy: {:.4f}'.format(acc))

    # eval
    print(f"--- evaluation {epoch}/{num_epoch} ---")
    model.eval()
    correct = 0
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        _, pred = model(batch).max(dim=1)
        correct += float(pred.eq(batch.y).sum().item())
    acc = correct / len(test_dataset)
    print('Test Accuracy: {:.4f}'.format(acc))


# %%
