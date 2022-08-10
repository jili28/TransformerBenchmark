"""
Create dataloaders depending on settings in config.py
"""

from config import config
from src.models.hyperparameters import params
from src.data.FIRST.first_datamodule import FIRST_DataModule
from src.data.PARITY.parity_datamodule import PARITY_DataModule
from src.data.DYCK.dyck_datamodule import DYCK_DataModule
from pathlib import Path


def get_datamodule():
    if config['dataset'] == 'first':
        return FIRST_DataModule(
            word_length=params[config['model']]['word_length'],
            len = params[config['model']]['len'],
            leq = params[config['model']]['leq'],
            batch_size=params[config['model']]['batch_size']
            )
    elif config['dataset'] == 'parity':
        return PARITY_DataModule(
            word_length=params[config['model']]['word_length'],
            len = params[config['model']]['len'],
            leq = params[config['model']]['leq'],
            batch_size=params[config['model']]['batch_size']
            )
    elif config['dataset'] == 'dyck':
        print(params[config['model']]['word_length'])
        return DYCK_DataModule(
            word_length=params[config['model']]['word_length'],
            len = params[config['model']]['len'],
            test_len=  params[config['model']]['test_len'],
            leq = params[config['model']]['leq'],
            k = params[config['model']]['k'],
            M = params[config['model']]['M'],
            batch_size=params[config['model']]['batch_size']
            )
    else:
        raise NotImplementedError("Choose valid dataset in config.py")


if __name__ == '__main__':
    print(f"Loading data")
    datamodule = get_datamodule()
    datamodule.prepare_data()
    datamodule.setup()
    # test dataloader
    for batch in datamodule.train_dataloader():
        print(batch[0].shape)
        print(batch[1].shape)
        break
