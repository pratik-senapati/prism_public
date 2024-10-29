# chan.py
import torch
import torch.nn as nn

class CrossModalHierarchicalAttentionNetwork(nn.Module):
    def __init__(self, dim, heads):
        super(CrossModalHierarchicalAttentionNetwork, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)

    def forward(self, img_features, text_features):
        # Assuming img_features and text_features are of shape (batch_size, seq_len, dim)
        img_attended, _ = self.attn(img_features, text_features, text_features)
        text_attended, _ = self.attn(text_features, img_features, img_features)
        return img_attended, text_attended