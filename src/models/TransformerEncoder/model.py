import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .transformer import build_transformer

from sklearn.metrics import classification_report


class Acceptor(pl.LightningModule):
    """ This is the Acceptor module that learns language recognition """

    def __init__(self, transformer, num_queries = 1, backbone = None):  # , aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: possible backbone to use
            num_queries: number of attention queries
            transformer: torch module of the transformer architecture. See transformer.py
        """
        super().__init__()
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.final = MLP(hidden_dim, hidden_dim, 2, 2)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # CNN --> Transformer projector
        self.input_proj = nn.Conv1d(backbone.num_channels, hidden_dim, 1)
        self.backbone = backbone
        self.transformer = transformer
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """ The forward expects a NestedTensor, which consists of:
               - x.tensor: batched images, of shape [batch_size x S]
        """
        src, pos = self.backbone(x)
        src = src.unsqueeze(1).float()
        src = self.input_proj(src)
        # no attention mask needed
        # assert mask is not None

        assert mask is not None
        hs = self.transformer(src, mask, self.query_embed.weight, pos)[0]
        #bs, num_queries, hidden
        out = self.final(hs[-1])
        out = self.sigmoid(out)
        #possibly reshape?
        return out

    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        y_pred = self.forward(x, mask)
        #print(y_pred.size())
        y_pred = y_pred.squeeze()
        # run through criterion
        loss = self.loss(y_pred, y.long())
        self.log("Training Loss", loss)
        return loss


    def validation_step(self, batch, barch_idx):
        x, mask, y = batch
        y_pred = self.forward(x, mask)

        y_pred = y_pred.squeeze()
        # run through criterion
        loss = self.loss(y_pred, y.long())
        self.log("Validation Loss", loss)
        return (np.array(y.cpu()).flatten(), np.array(torch.argmax(y_pred.cpu(), dim=1)).flatten())

    def validation_epoch_end(self, outputs) -> None:
        """
        Called at the end of validation.
        """
        y_true, y_hat = zip(*outputs)
        y_true = np.concatenate(y_true).flatten()
        y_hat = np.concatenate(y_hat).flatten()
        report = classification_report(y_true, y_hat, digits=4, output_dict=True)
        for i in report.keys():
            log_object = report[i]
            if isinstance(log_object, dict):
                for k in log_object.keys():
                    self.log(f'{i}_{k}', log_object[k])
            else:
                self.log(f'{i}', log_object)
        logging.info(f'classification_report:\n {report}')
        print(f'classification_report:\n {report}')

    def test_step(self, batch, barch_idx):
        x, mask, y = batch
        y_pred = self.forward(x, mask)

        y_pred = y_pred.squeeze()
        # run through criterion
        loss = self.loss(y_pred, y.long())
        self.log("Test Loss", loss)
        return (np.array(y.cpu()).flatten(), np.array(torch.argmax(y_pred.cpu(), dim=1)).flatten())


    def test_epoch_end(self, outputs) -> None:
        """
        Called at the end of testing.
        """
        y_true, y_hat = zip(*outputs)
        y_true = np.concatenate(y_true).flatten()
        y_hat = np.concatenate(y_hat).flatten()
        report = classification_report(y_true, y_hat, digits=4, output_dict=True)
        for i in report.keys():
            log_object = report[i]
            if isinstance(log_object, dict):
                for k in log_object.keys():
                    self.log(f'{i}_{k}', log_object[k])
            else:
                self.log(f'{i}', log_object)
        logging.info(f'Test classification_report:\n  {report}')
        print(f'Test classification_report:\n  {report}')
        for i in report.keys():
            logging.info(f"{i}: {report[i]}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 0)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 0 else layer(x)
        return x


def build(args):


    # device = torch.device(args["device"])
    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = Acceptor(
        backbone=backbone,
        transformer=transformer,
    )
    return model
