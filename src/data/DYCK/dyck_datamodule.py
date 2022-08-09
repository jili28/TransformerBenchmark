import logging
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from itertools import product

def collate_tuples(batch):
    # print(batch)
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])
    # print(batch[0].size())
    return tuple(batch)


class DYCK_Dataset(Dataset):
    """
    while theoretically an iterable dataset, abusing Dataset API
    works out to smoother implementation
    """
    def __init__(self, word_length, len, k, M, leq = True):
        super(DYCK_Dataset, self).__init__()
        assert word_length > 0
        self.word_length = word_length
        self.leq = leq
        self.len = len #pre-generated until length 512(*2)
        self.k = k #recursion bound <= 15
        self.M = M #bracket types
        self.dyck = np.load("dyck_15_{}.npy".format(M),allow_pickle='TRUE')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        mask = torch.zeros((int(2*self.word_length)))
        if self.leq:
            word_length = np.random.randint(1, self.word_length+1)
            mask[2*word_length:] = 1
        else:
            word_length = self.word_length
        
        is_dyck = np.random.randint(2)
        
        if is_dyck:
            depth = np.random.randint(1, min(self.k, word_length)+1)
            word = np.random.choice(self.dyck[word_length][depth])
            word = [int(x) for x in word]
        else: 
            word = np.random.randint(2, size=(int(2*word_length)))

        word = np.pad(word, (0, 2*(self.word_length - word_length)), mode="constant")

        return torch.tensor(word), is_dyck

        

class DYCK_DataModule(pl.LightningDataModule):
    def __init__(self, word_length, leq, len, k, M, batch_size=32, collate_fn=None):
        super().__init__()
        self.batch_size = batch_size
        self.transform = None  # define dataset specific transforms here
        self.collate_fn = collate_fn
        self.word_length = word_length
        self.leq = leq
        self.len = len
        self.k = k #recursion bound <= 15
        self.M = M #bracket types


    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        There are also data operations you might want to perform on every GPU. Use setup() to do things like:
            - count number of classes
            - build vocabulary
            - perform train/val/test splits
            - create datasets
            - apply transforms (defined explicitly in your datamodule)
        :param stage: fit, test, predict
        :return: Nothing
        """

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = DYCK_Dataset(word_length=self.word_length, len=self.len, k=self.k, M=self.M, leq=True)
            self.val_set = DYCK_Dataset(word_length=self.word_length, len=self.len, leq=False)

        if stage == "test" or stage is None:
            self.test_set = DYCK_Dataset(word_length=self.word_length, len=self.len, k=self.k, M=self.M, leq=False)

        if stage == "predict" or stage is None:
            self.predict_set = NotImplemented

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def predict_dataloader(self):
        raise NotImplementedError("We do not have a predict set in this datamodule")