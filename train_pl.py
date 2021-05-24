# Run options
import os

from data.semi_supervised import SemiSupervised
import torch
import pytorch_lightning as pl

from modules.lightning.classifier import LightningClassifier
from modules.lightning.vae import LightningVAE
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.experiment_utils import run_visualizations
from utils.latent_analysis import analyze_latents
from utils.wandb_filesync import WandBFilesync
from utils.wandb_model_checkpoint import WandBModelCheckpoint

wandb.init(project='vae_classifier', entity='mbml')

USE_CUDA = torch.cuda.is_available()

@hydra.main(config_path='config', config_name="default")
def run_training(cfg : DictConfig) -> dict:
    print(OmegaConf.to_yaml(cfg))

    cfg_file = os.path.join(wandb.run.dir, 'config.yaml')
    with open(cfg_file, 'w') as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    wandb.save(cfg_file)

    data_module = SemiSupervised(cfg["labelled_per_class"], cfg["missing_batch_size"], cfg["labelled_batch_size"], cfg["n_steps"])
    hyperparams = {"a": cfg["a"], "labelled_per_class": cfg["labelled_per_class"], **cfg["optimizer_config"], **cfg["model_config"], }

    model_config = dict(cfg["model_config"])
    is_vae =  model_config.pop("mode") == "vae"
    if is_vae:
        model = LightningVAE(a=cfg["a"], vae_config=model_config, optimizer_config=cfg["optimizer_config"])
    else:
        model = LightningClassifier(classifier_config=model_config, optimizer_config=cfg["optimizer_config"])

    callbacks = []
    callbacks.append(pl.callbacks.EarlyStopping(patience=5, monitor='val_loss'))
    callbacks.append(pl.callbacks.ModelCheckpoint(dirpath=wandb.run.dir,
                                                  monitor='val_loss',
                                                  filename='model',
                                                  verbose=True,
                                                  period=1))
    wandb.save('*.ckpt') # should keep it up to date
    # callbacks.append(WandBFilesync(filename='model.ckpt', period=10)) # not sure if this one is necessary

    if cfg["use_wandb"]:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(hyperparams)
    else:
        logger = pl.loggers.CSVLogger

    gpus = 0
    if USE_CUDA:
        gpus = 1
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, default_root_dir="training/logs", max_epochs=cfg["max_epochs"], gpus=gpus)

    # trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    if is_vae:
        run_visualizations(model, data_module.test_dataloader(), os.path.join(wandb.run.dir, 'fig'))
        analyze_latents(model, data_module.test_dataloader(), os.path.join(wandb.run.dir, 'fig'))

    #
    # return {
    #     'vae': vae,
    #     'results_train': results_train,
    #     'results_test': results_test
    # }

if __name__ == '__main__':
    run_training()