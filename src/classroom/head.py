"""Heads trained on cached per-person V-JEPA 2 features (encoder stays frozen)."""

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Baseline: LayerNorm + linear on the in-box max token (what mmaction's AVA RoI head sees)."""

    PEAK = 4

    def __init__(self, dim, num_classes, dropout=0.5, **_):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, box_tok, ctx_tok):
        return self.fc(self.drop(self.norm(box_tok[:, self.PEAK].float())))


class TokenHead(nn.Module):
    """Small transformer over the box tokens + keyframe context tokens, read out from a CLS token."""

    def __init__(self, dim, num_classes, n_tokens, hidden=512, depth=2, heads=8, dropout=0.1, **_):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden))
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, hidden))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, 4 * hidden, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, box_tok, ctx_tok):
        x = self.inp(torch.cat([box_tok, ctx_tok], 1).float())
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        x = self.blocks(x)
        return self.fc(self.drop(self.norm(x[:, 0])))


def build_head(name, dim, num_classes, n_tokens, **kw):
    if name == "linear":
        return LinearHead(dim, num_classes, **kw)
    if name == "tokens":
        return TokenHead(dim, num_classes, n_tokens, **kw)
    raise ValueError(name)
