from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

# Choose a local directory to store the dataset, e.g. “data/Planetoid”
root = "data/Planetoid"

# Choose which dataset: “Cora”, “CiteSeer”, or “PubMed”
name = "Cora"  # or “CiteSeer”, “PubMed”

# Optionally apply transforms, e.g. normalization
transform = T.NormalizeFeatures()

dataset = Planetoid(root=root, name=name, transform=transform)

data = dataset[0]  # the graph object: node features, edges, labels, masks
print(data)
