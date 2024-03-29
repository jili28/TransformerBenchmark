import wandb
import yaml
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from config import config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from src.utils.model_factory import get_model
from src.models.hyperparameters import params as default_params
import time
from src.utils.logger import Logger, log_params, WandbImageLogger
import os
from datamodule_factory import get_datamodule, get_datamodule_sweep
import logging


def main():
    wandb.init()
    #wandb.init(config=default_params)
    params = wandb.config

    # Create model directory and Logger
    run_id = time.strftime("%Y%m%d-%H%M%S")
    # log_dir = \
    #     f"reports/logs/{run_id}_{config['model']}_" \
    #     f"{config['dataset']}_{params[config['model']]['word_length']}"
    log_dir = \
        f"reports/logs/{run_id}_{params['model']}_" \
        f"{config['dataset']}_{params['word_length']}_k_{params['k']}_M_{params['M']}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(print_fp=os.path.join(log_dir, 'out.txt'))
    # Create logging file
    logging.basicConfig(filename=f"{log_dir}/info.log", encoding='utf-8', level=logging.INFO)
    logging.info("Started logging.")

    # Obtain datamodule based on config settings for dataset
    #data_module = get_datamodule()
    data_module = get_datamodule_sweep(params) #used for sweepiong
    logging.info("Created data module.")
    print(params)
    # Create model based on config.py and hyperparameters.py settings
    # changed to include model factory
    #model = get_model(params[config['model']], config['model'])
    model = get_model(params, params['model'])
    logging.info("Created model.")
    # print model summary
    # summary(model, (config['input_height'], config['input_width']))

    # Log hyperparameters and config file
    log_params(log_dir)

    # Run the model
    tb_logger = TensorBoardLogger("./reports/logs/",
                                  name=f"{run_id}_{config['model']}k_{params['k']}_M_{params['M']}"
                                  )
    #tb_logger.log_hyperparams(params[config['model']])  # log hyperparameters
    tb_logger.log_hyperparams(params)
    wandb_logger = WandbLogger(project=f"{config['project']}",
                               entity=config['entity'],
                               # save_dir=f"reports/logs/{run_id}_{config['model']}_" \
                               #          f"{config['dataset']}_{params[config['model']]['word_length']}",
                               save_dir=f"reports/logs/{run_id}_{config['model']}_" \
                                        f"{config['dataset']}_{params['word_length']}_k_{params['k']}_M_{params['M']}",
                               # id=f"{run_id}_{config['model']}_" \
                               #    f"{config['dataset']}_{params[config['model']]['word_length']}"
                               id=f"{run_id}_{config['model']}_" \
                                  f"{config['dataset']}_{params['word_length']}_k_{params['k']}_M_{params['M']}"
                               )
    wandb_logger.experiment.config["Model"] = config['model']
    #wandb_logger.experiment.config.update(params[config['model']])
    wandb_logger.experiment.config.update(params)
    trainer = pl.Trainer(accelerator="gpu",  # cpu or gpu
                         devices=-1,  # -1: use all available gpus, for cpu e.g. 4
                         enable_progress_bar=True,  # disable progress bar
                         # show progress bar every 500 iterations
                         # precision=16, # 16 bit float precision for training
                         logger=[tb_logger, wandb_logger],  # log to tensorboard and wandb
                         #logger = [tb_logger],

                         #max_epochs=params[config['model']]['epochs'],  # max number of epochs
                         max_epochs=params['epochs'],
                         callbacks=[EarlyStopping(monitor="Validation Loss", patience=3),  # early stopping
                                    ModelSummary(max_depth=1),  # model summary
                                    ModelCheckpoint(log_dir, monitor='Validation Loss', save_top_k=1),
                                    # save best model
                                    # TQDMProgressBar(10)
                                    ],
                         auto_lr_find=True  # automatically find learning rate
                         )
    logging.info("Start training.")
    trainer.fit(model, data_module)  # train the model
    logging.info("Finished training.")
    trainer.test(model, data_module)  # test the model

    # Log test images to wandb
    # inference_loader = DataLoader(data_module, batch_size=1, shuffle=False, num_workers=2)
    # WandbImageLogger(model, inference_loader, wandb_logger).log_images()


if __name__ == "__main__":
    #load sweep files
    with open('sweep.yaml') as f:
        sweep_params = yaml.load(f, Loader=yaml.FullLoader)
    print(sweep_params)
    sweep_id = wandb.sweep(sweep_params)
    wandb.agent(sweep_id, function=main)

    main()
