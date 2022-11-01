# %%
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch

# ENZYMESデータセットのロード
dataset = TUDataset(root='tmp/ENZYMES', name='ENZYMES')

# %%
# ex1
# contiguousメソッドによってedgeのリストが返される
what_is_contiguous = dataset[0].edge_index.t().contiguous()
print(what_is_contiguous)

# %%
# ex3 (ex2はスキップ)
# Q. DataBatch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32]) は何を意味するか？
# A. batch: バッチ化されたグラフのノード数, edge_index: 枝のindex, x: [ノード数, 特徴量の次元], y: ラベル数(この場合はグラフ単位のラベルか) 






