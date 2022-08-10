import logging
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from typing import Tuple, List
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, IterableDataset
from itertools import product

from sklearn.model_selection import train_test_split


def collate_tuples(batch):
    # print(batch)
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])
    # print(batch[0].size())
    return tuple(batch)


file_path = '/home/jimmy/TransformerBenchmark/data/raw/dyck_files/dyck_15_{}.npy'


class DYCK_GeneratorDataset(Dataset):
    """
    while theoretically an iterable dataset, abusing Dataset API
    works out to smoother implementation
    """

    def __init__(self, word_length, len, k, M, min_word_length=2, leq=True, invert_mask=True):
        super(DYCK_GeneratorDataset, self).__init__()

        assert word_length > 0
        assert word_length % 2 == 0
        self.word_length = int(word_length / 2)

        assert min_word_length > 0 and min_word_length % 2 == 0 and min_word_length < word_length
        self.min_word_length = int(min_word_length / 2)
        self.leq = leq
        self.len = len  # pre-generated until length 512(*2)
        self.k = k  # recursion bound <= 15
        self.M = M  # bracket types
        self.vocab_size = 2 * M
        self.invert_mask = invert_mask
        print("Loading dyck file...", end="")
        self.dyck = np.load(file_path.format(M), allow_pickle=True)
        print(" Done")

    def __len__(self):
        return self.len

    def generate_random_word(self):
        # mask = torch.zeros((int(2 * self.word_length)))
        if self.leq:
            word_length = np.random.randint(self.min_word_length, self.word_length + 1)
            # mask[2 * word_length:] = 1
        else:
            word_length = self.word_length

        if self.invert_mask:
            mask = torch.ones((int(2 * self.word_length)))
            # print(len(mask))
            # print(word_length)
            mask[2 * word_length:] = 0
        else:
            mask = torch.zeros((int(2 * self.word_length)))
            mask[2 * word_length:] = 1

        is_dyck = np.random.randint(low=0, high=6)
        is_dyck = is_dyck < 1
        if is_dyck:
            depth = np.random.randint(1, min(self.k, word_length) + 1)
            word = np.random.choice(self.dyck[word_length][depth])
            word = np.array([int(x) for x in word])
        else:
            word = np.random.randint(0, high=self.vocab_size, size=(int(2 * word_length)))

        return word, mask, word_length, is_dyck

    def pad_word(self, word, length, mode="constant"):
        return np.pad(word, (0, 2 * (self.word_length - length)), mode=mode)

    def unique_generator(self, seen_words=None):
        if seen_words is None:
            seen_words = set()
        seen_words = seen_words

        while True:
            word, mask, length, is_dyck = self.generate_random_word()

            flat = "".join(str(x) for x in word.flatten())  # this is reasonable since we're using bit-strings
            if flat in seen_words:
                continue
            else:
                seen_words.add(flat)

            word = self.pad_word(word, length)
            yield torch.tensor(word), mask, 2 * length, is_dyck

    def __getitem__(self, idx):
        # generate a word
        word, mask, length, is_dyck = self.generate_random_word()

        # pad the word
        word = self.pad_word(word, length)
        return torch.tensor(word), mask, 2 * length, is_dyck


class DYCK_Dataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DYCK_DataModule(pl.LightningDataModule):
    def __init__(self, word_length, leq, len, k, M, test_min_length = 2, test_max_length = None,batch_size=32, collate_fn=None):
        super().__init__()
        self.batch_size = batch_size
        self.transform = None  # define dataset specific transforms here
        self.collate_fn = collate_fn
        # assert word_length % 2 == 0
        self.word_length = word_length
        self.leq = leq
        self.len = len
        self.k = k  # recursion bound <= 15
        self.M = M  # bracket types

        if test_max_length is None:
            self.test_max_length = word_length
        else:
            self.test_max_length = test_max_length

        self.test_min_length = test_min_length

    def prepare_data(self):
        generator = DYCK_GeneratorDataset(word_length=self.word_length, len=self.len, k=self.k, M=self.M,
                                          leq=True).unique_generator()

        # generate self.len unique samples
        full_set = list(next(generator) for _ in tqdm(range(int(self.len*9/10)), desc=f"Generating train dataset (n={self.len})"))

        print("Creating train/val/test splits (80/10/10)")

        test_generator = DYCK_GeneratorDataset(word_length=self.test_max_length, min_word_length=self.test_min_length,
                                               len=self.len, k=self.k, M=self.M,
                                               leq=True).unique_generator(set(full_set))

        test_split = list(next(test_generator) for _ in
                          tqdm(range(self.len // 10), desc=f"Generating val dataset (n={self.len // 9})"))
        train_split, val_split = train_test_split(full_set, test_size=1 / 9)

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        print(
            f"Created splits: train (n={len(self.train_split)}), "
            f"val (n={len(self.val_split)}), test (n={len(self.test_split)})")

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

        print("Setting up DataLoaders")
        # len here is pretty useless using the new generator

        # val_split, test_split = train_test_split(val_test_split, test_size=0.5)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = DYCK_Dataset(items=self.train_split)
            self.val_set = DYCK_Dataset(items=self.val_split)
        if stage == "test" or stage is None:
            self.test_set = DYCK_Dataset(items=self.test_split)

        if stage == "predict" or stage is None:
            self.predict_set = NotImplemented


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True,
                          num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=2)

    def predict_dataloader(self):
        raise NotImplementedError("We do not have a predict set in this datamodule")
