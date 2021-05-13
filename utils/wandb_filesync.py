import wandb
import pytorch_lightning as pl

class WandBFilesync(pl.callbacks.Callback):
    def __init__(self, filename, period=1):
        self.filename = filename
        self.period = period

    def on_validation_end(self, trainer, pl_module, checkpoint):
        if (trainer.current_epoch % self.period) == 0:
            wandb.save(self.filename, base_path=wandb.run.dir)