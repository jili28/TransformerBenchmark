import logging

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report

import src.models.SigmoidEncoder.transformer as encoder
from src.models.SigmoidEncoder.position_encoding import PositionEncoding


class Encoder(pl.LightningModule):
    def __init__(self, alphabet_size, layers, heads, d_model, d_ffnn, scaled=False, eps=1e-5):
        super().__init__()

        self.word_embedding = torch.nn.Embedding(num_embeddings=alphabet_size, embedding_dim=d_model)
        self.pos_encoding = PositionEncoding(d_model)

        if scaled:
            encoder_layer = encoder.ScaledTransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn,
                                                                  dropout=0., batch_first=True)
        else:
            encoder_layer = encoder.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn,
                                                            dropout=0., batch_first=True)
        encoder_layer.norm1.eps = encoder_layer.norm2.eps = eps
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.output_layer = torch.nn.Linear(d_model, 1)
        self.loss = torch.nn.LogSigmoid()

    def forward(self, w):
        x = self.word_embedding(w) + self.pos_encoding(len(w[0]))
        #y = self.encoder(x.unsqueeze(1)).squeeze(1)
        y = self.encoder(x)
        y = y[:, 0, :]
        z = self.output_layer(y)
        return z

    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        y_pred = self.forward(x)
        # print(y_pred.size())
        y_pred = y_pred.squeeze()

        y_pred[~y] = -y_pred[~y]
        # run through criterion
        loss = -self.loss(y_pred)
        loss = torch.mean(loss)
        self.log("Training Loss", loss)
        return loss

    def validation_step(self, batch, barch_idx):
        x, mask, y = batch
        y_pred = self.forward(x)
        # print(y_pred.size())
        y_pred = y_pred.squeeze()

        y_pred[~y] = -y_pred[~y]
        # run through criterion
        loss = -self.loss(y_pred)
        loss = torch.mean(loss)
        validation_alpha = torch.zeros(len(self.encoder.layers))
        for l, layer in enumerate(self.encoder.layers):
            # weight of CLS attending to first symbol
            validation_alpha[l] += self.encoder.layers[l].last_weights[0][0][1].detach().cpu()
        self.log("Validation Loss", loss)
        return (np.array((y_pred > 0 ).cpu().flatten()), np.array(validation_alpha))
        #return (np.array(y.cpu()).flatten(), np.array(torch.argmax(y_pred.cpu(), dim=1)).flatten())

    def validation_epoch_end(self, outputs) -> None:
        """
        Called at the end of validation.
        """
        y_hat, validation_alpha = zip(*outputs)
        y_hat = np.concatenate(y_hat).flatten()
        validation_alpha = np.sum(validation_alpha, axis = 1)
        validation_alpha /= len(y_hat)
        accuracy = np.mean(y_hat)
        report = {'accuracy': accuracy, 'validation_alpha' : validation_alpha}
        # y_true, y_hat = zip(*outputs)
        # y_true = np.concatenate(y_true).flatten()
        # y_hat = np.concatenate(y_hat).flatten()
        # report = classification_report(y_true, y_hat, digits=4, output_dict=True)
        # for i in report.keys():
        #     log_object = report[i]
        #     if isinstance(log_object, dict):
        #         for k in log_object.keys():
        #             self.log(f'{i}_{k}', log_object[k])
        #     else:
        #         self.log(f'{i}', log_object)

        self.log("Validation Accuracy", float(accuracy), prog_bar=True)
        logging.info(f'classification_report:\n {report}')
        #print(f'classification_report:\n {report}')

    def test_step(self, batch, barch_idx):
        x, mask, y = batch
        y_pred = self.forward(x)
        # print(y_pred.size())
        y_pred = y_pred.squeeze()

        y_pred[~y] = -y_pred[~y]
        # run through criterion
        loss = -self.loss(y_pred)
        loss = torch.mean(loss)
        validation_alpha = torch.zeros(len(self.encoder.layers))
        for l, layer in enumerate(self.encoder.layers):
            # weight of CLS attending to first symbol
            validation_alpha[l] += self.encoder.layers[l].last_weights[0][0][1].detach().cpu()
        self.log("Test Loss", loss)

        return (np.array((y_pred > 0 ).cpu().flatten()), np.array(validation_alpha))

    def test_epoch_end(self, outputs) -> None:
        """
        Called at the end of testing.
        """
        y_hat, validation_alpha = zip(*outputs)
        y_hat = np.concatenate(y_hat).flatten()
        validation_alpha = np.sum(validation_alpha, axis=1)
        validation_alpha /= len(y_hat)
        validation_alpha /= len(y_hat)
        accuracy = np.mean(y_hat)
        report = {'accuracy': accuracy, 'validation_alpha': validation_alpha}
        # y_true, y_hat = zip(*outputs)
        # y_true = np.concatenate(y_true).flatten()
        # y_hat = np.concatenate(y_hat).flatten()
        # report = classification_report(y_true, y_hat, digits=4, output_dict=True)
        # for i in report.keys():
        #     log_object = report[i]
        #     if isinstance(log_object, dict):
        #         for k in log_object.keys():
        #             self.log(f'{i}_{k}', log_object[k])
        #     else:
        #         self.log(f'{i}', log_object)
        self.log("Test Accuracy", float(accuracy), prog_bar=True)
        logging.info(f'classification_report:\n {report}')
        #print(f'classification_report:\n {report}')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

def build_encoder(args):


    # device = torch.device(args["device"])
    model = Encoder(
        alphabet_size=2,
        layers = args['layers'],
        heads = args['heads'],
        d_model= args['d_model'],
        d_ffnn=args['d_ffnn'],
        scaled=args['scaled'],
        eps=args['eps']
    )
    return model
