import torch
import pytorch_lightning as pl

class PositionEncoding(pl.LightningModule):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, n):
        zero = torch.zeros(n, device=self.device)
        pos = torch.arange(0, n, device=self.device).to(torch.float)
        pe = torch.stack([pos == 1] + [zero]*(self.size-1), dim=1)
        return pe