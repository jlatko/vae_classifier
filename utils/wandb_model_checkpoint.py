import wandb
import pytorch_lightning as pl

class WandBModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _save_model(self, filepath, trainer, pl_module):
        # this seems to differ between pl versions
        # up to version 1.2.4 this should work
        # monkey patch, support both versions
        # if 'pl_module' in super()._save_model.__code__.co_varnames:
        #     super()._save_model(trainer, filepath)
        # else:
        super()._save_model(filepath, trainer, pl_module)
        wandb.save(filepath, base_path=wandb.run.dir)