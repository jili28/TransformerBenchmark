# adopted from PrincetonNLP group

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm
import math
import pytorch_lightning as pl
import utils
from transformers import GPT2Config, GPT2Model
from torch.optim.lr_scheduler import ReduceLROnPlateau
MAX_LEN = 1024
P_DROP = 0


class TransformerModel(pl.LightningModule):
    def __init__(self, num_heads, input_size, hidden_size, num_layers, vocab_size=5,
                 bracket_num=10, lr=1e-3, embedding_type='pw'):
        super(TransformerModel, self).__init__()
        # self.input_size = input_size  # dimensionality of initial embeddings
        input_size = hidden_size
        self.input_size = hidden_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_heads = num_heads
        self.e_type = embedding_type
        self.hidden_size = hidden_size
        config = GPT2Config(n_embd=hidden_size, n_layer=num_layers, n_inner=hidden_size, attn_pdrop=P_DROP,
                            embd_pdrop=P_DROP, resid_pdrop=P_DROP, vocab_size=self.vocab_size, n_head=self.n_heads,
                            n_positions=MAX_LEN, n_ctx=MAX_LEN)
        print(config)
        self.model = GPT2Model(config)
        print(self.model)
        self.model.to(self.device)
        tqdm.write(
            'Constructing a GPT2 pytorch model w hidden size {}, layers {}, dropout {}'.format(hidden_size, num_layers,
                                                                                               0.0))

        if self.e_type == 'cos':
            funcs = [math.sin, math.cos]
            self.model.wpe.weight.data = torch.tensor([[funcs[i % 2](pos / 10000 ** (2 * i / hidden_size))
                                                                    for i in range(hidden_size)] for pos in
                                                                   range(MAX_LEN)])
            self.model.wpe.weight.requires_grad = False

        if self.e_type == 'p' or self.e_type == 'pw':
            self.model.wpe.weight.data.zero_()
            self.model.wpe.weight.requires_grad = False

            self.embedding = nn.Embedding(self.vocab_size, input_size - 1)  # learned inputq
            self.embedding.to(self.device)

            self.embedding_p = nn.Embedding(MAX_LEN, 1)
            self.embedding_p.weight.data = torch.tensor([[i / MAX_LEN] for i in range(MAX_LEN)])
            self.embedding_p.weight.requires_grad = False
            self.embedding_p.to(self.device)

        if self.e_type == 'pw':
            # k = args['language']['bracket_types']
            k = vocab_size//2
            self.embedding = nn.Embedding(self.vocab_size, input_size - 1 - k - 4)
            self.embedding_e = nn.Embedding(self.vocab_size, k + 4)  # maing

            def get_row(i):
                arr = [0] * (k + 4)  # conversion to bracket typing in initial input
                if i < 2 * k:  # one of the many bracket_types
                    # arr[i % k] = arr[k + (i < k)] = 1 # opening-->
                    arr[i // 2] = 1
                    arr[k + (i % 2 == 1)] = 1
                else:  # other characters #start end? #we not use these anyways
                    arr[i - 2 * k + k + 2] = 1
                return arr

            self.embedding_e.weight.data = torch.tensor([get_row(i) for i in range(2 * k + 2)])
            self.embedding_e.weight.requires_grad = False
            self.embedding.to(self.device)
            self.embedding_e.to(self.device)

        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        self.loss = nn.BCELoss()
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, batch, mask, lengths):
        """ Computes the forward pass to construct prefix representations.
        Arguments:
          batch: (batch_len, seq_len) vectors representing
                 contexts
          mask: (batch_len, seq_len) 0: masked positions, 1: unmasked positions
        Returns:
          hiddens: (batch_len, seq_len, hidden_size)
                   recurrent state vectors for each token in input.
        """
        if self.e_type == 'default' or self.e_type == 'cos':
            last_hidden, _ = self.model.forward(batch).values()
            return self.final_layer(last_hidden)

        else:
            vec1 = self.embedding(batch)
            pos = torch.ones(batch.size(), device=self.device).cumsum(-1) - 1
            vec2 = self.embedding_p(pos.long())
            if self.e_type == 'p':
                vec = torch.cat((vec1, vec2), -1)
            else:
                vec3 = self.embedding_e(batch)
                vec = torch.cat((vec1, vec2, vec3), -1).float()
            last_hidden, _ = self.model.forward(inputs_embeds=vec, attention_mask=mask).values()
            batch = []
            #print(last_hidden.size())
            for i in range(len(last_hidden)):
                t = torch.mean(last_hidden[i][:lengths[i]], dim=0)
                #print(t.size())
                batch.append(t)

            last_hidden = torch.stack(batch)
            #print(last_hidden.size())
            return self.final_layer(last_hidden)

    def training_step(self, batch, batch_idx):
        x, mask, lengths, y = batch
        y_pred = self.forward(x, mask, lengths)
        # print(y_pred.size())
        # print(lengths.size())

        #y_pred = y_pred[torch.arange(0, len(x)), (lengths-1)]
        y_pred = y_pred.squeeze()
        loss = self.loss(y_pred, y.float())
        self.log("Training Loss", loss)
        return loss

    def validation_step(self, batch, barch_idx):

        # run through criterion
        x, mask, lengths, y = batch
        y_pred = self.forward(x, mask, lengths)
        y_pred = y_pred.squeeze()
        # print(y_pred.size())
        # print(max(lengths.size()))
        #y_pred = y_pred[torch.arange(0, len(x)), lengths-1]
        y_pred = y_pred.squeeze()
        # print(y_pred.size())
        # print(y.size())
        loss = self.loss(y_pred, y.float())
        self.log("Validation Loss", loss)
        return np.array(torch.round(y_pred).cpu().flatten()), np.array(y.cpu().flatten())

    def validation_epoch_end(self, outputs) -> None:
        """
        Called at the end of validation.
        """
        y_hat, y_true = zip(*outputs)
        y_true = np.concatenate(y_true).flatten()
        y_hat = np.concatenate(y_hat).flatten()
        self.log("# positive Val example", float(np.sum(y_true)))
        report = classification_report(y_true, y_hat, digits=4, output_dict=True)
        for i in report.keys():
            log_object = report[i]
            if isinstance(log_object, dict):
                for k in log_object.keys():
                    self.log(f'Validation_{i}_{k}', log_object[k])
            else:
                self.log(f'Validation_{i}', log_object)
        logging.info(f'classification_report:\n {report}')

    def test_step(self, batch, barch_idx):
        x, mask, lengths, y = batch
        y_pred = self.forward(x, mask, lengths)
        #y_pred = y_pred[torch.arange(0, len(x)), lengths-1]
        y_pred = y_pred.squeeze()
        loss = self.loss(y_pred, y.float())
        self.log("Test Loss", loss)
        return np.array(torch.round(y_pred).cpu().flatten()), np.array(y.cpu().flatten())

    def test_epoch_end(self, outputs) -> None:
        """
        Called at the end of testing.
        """
        y_hat, y_true = zip(*outputs)
        y_true = np.concatenate(y_true).flatten()
        y_hat = np.hstack(y_hat).flatten()
        self.log("# positive test examples", float(np.sum(y_true)))
        report = classification_report(y_true, y_hat, digits=4, output_dict=True)
        for i in report.keys():
            log_object = report[i]
            if isinstance(log_object, dict):
                for k in log_object.keys():
                    self.log(f'Test_{i}_{k}', log_object[k])
            else:
                self.log(f'Test_{i}', log_object)
        logging.info(f'classification_report:\n {report}')

    def configure_optimizers(self):
        optimizer =  torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0),
                "monitor": "Training Loss",
                "frequency": 2
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

def build_causal_encoder(args):
    # device = torch.device(args["device"])
    model = TransformerModel(
        num_heads=args['num_heads'], input_size=args['input_size'],
        hidden_size=args['hidden_size'], num_layers=args['num_layers'],
        vocab_size=args['M'] *2,
        bracket_num=args['M'], lr=args['lr'], embedding_type=args['embedding_type'])

    return model
