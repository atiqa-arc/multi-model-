# GMC/model/gcn_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        # x: [N, in_features]
        # edge_index: [2, num_edges]

        N = x.size(0)
        out = torch.zeros_like(x)

        # Scatter add messages
        src, dst = edge_index
        messages = self.linear(x[src])
        out.index_add_(0, dst, messages)

        # Normalize by degree
        deg = torch.bincount(dst, minlength=N).clamp(min=1).float().unsqueeze(1)
        out = out / deg

        return F.relu(out)
