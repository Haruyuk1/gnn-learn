from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch

# ENZYMESデータセットのロード
dataset = TUDataset(root='tmp/ENZYMES', name='ENZYMES')

# Datasetオブジェクトの基本
print("# of graphs: " + str(dataset))
print("# of classes: " + str(dataset.num_classes))
print("# of node features: " + str(dataset.num_node_features))
print("-"*10)

# Dataオブジェクトの基本
data = dataset[0]

print(data)

print("is undirected?: " + str(data.is_undirected()))
print("-"*10)

# データセットシャッフル
dataset = dataset.shuffle() # shuffle
train_dataset = dataset[:540] # split dataset into train and test dataset
test_dataset = dataset[540:]

# Batchオブジェクトの基本
loader = DataLoader(dataset, batch_size=32, shuffle=True)

batch: Batch = loader.__iter__().next() # 1つのバッチ
print("is batch type of 'Batch'?: " + str(isinstance(batch, Batch)))
print("batch size (i.e. # of graphs): " + str(batch.num_graphs))
print("batch attribute: " + str(batch.batch))
# まとめ
# - Batch は Data のサブクラス
# - やっていることは複数のグラフをまとめて1つのグラフにしてDataオブジェクトにしている
# -- その際，どのノードが元の何番目のグラフに所属しているかという情報をBatch.batchが保持している
# -- なのでbatchの定義域は[0, batch_size-1]

