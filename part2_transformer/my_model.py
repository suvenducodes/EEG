import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, n_chan, n_time, n_classes=2, F1=8, D=2,
                 d_model=32, n_heads=4, depth=2, p=0.3):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(F1))
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_chan, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 15), (1, 15)), nn.Dropout(p))
        with torch.no_grad():
            n_tok = self.spatial(self.temporal(
                torch.zeros(1, 1, n_chan, n_time))).shape[-1]
        self.proj = nn.Linear(F1 * D, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_embed=False):      # x: (B, C, T)
        z = self.spatial(self.temporal(x.unsqueeze(1)))   # (B, F, 1, N)
        tok = self.proj(z.squeeze(2).transpose(1, 2))     # (B, N, d)
        cls = self.cls.expand(x.size(0), -1, -1)
        z = self.encoder(torch.cat([cls, tok], 1) + self.pos)
        embed = self.norm(z[:, 0])
        logits = self.head(embed)
        return (logits, embed) if return_embed else logits 
