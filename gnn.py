# 3D-GCNN model
# The encoder and decoder of cnn can be replaced by any more powerful segmentation network


from gat_layers import *


import torch
import torch.nn as nn
import torch.nn.functional as F


# Input:(N, in_features)(N, N), Output:(N, out_features)
class GAT3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, heads=2):
        super(GAT3D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_channels if i == 0 else out_channels, out_channels))
            self.heads = heads
            
    def forward(self, x):
        # x: [B, N, C]
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=2, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.heads = heads
        self.out_per_head = out_features
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha

        self.fc = nn.Linear(in_features, heads * out_features, bias=False)
        self.attn_fc = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj=None):
        # x: [B, N, C]
        B, N, _ = x.size()
        h = self.fc(x)  # [B, N, heads*out_per_head]
        h = h.view(B, N, self.heads, self.out_per_head)  # [B, N, heads, out_per_head]
        h = h.transpose(1, 2)  # [B, heads, N, out_per_head]

        outputs = []
        for head in range(self.heads):
            h_head = h[:, head, :, :]  # [B, N, out_per_head]
            # 计算注意力分数
            h1 = h_head.unsqueeze(2).repeat(1, 1, N, 1)
            h2 = h_head.unsqueeze(1).repeat(1, N, 1, 1)
            attn_input = torch.cat([h1, h2], dim=-1)  # [B, N, N, 2*out_per_head]
            e = self.leakyrelu(self.attn_fc(attn_input)).squeeze(-1)  # [B, N, N]
            if adj is not None:
                zero_vec = -9e15 * torch.ones_like(e)
                attention = torch.where(adj > 0, e, zero_vec)
            else:
                attention = e
            attention = F.softmax(attention, dim=-1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            out_head = torch.matmul(attention, h_head)  # [B, N, out_per_head]
            outputs.append(out_head)
        h_out = torch.cat(outputs, dim=-1)  # [B, N, heads*out_per_head]
        if self.concat:
            return F.elu(h_out)
        else:
            return h_out.mean(dim=1)  # [B, N, out_per_head]
