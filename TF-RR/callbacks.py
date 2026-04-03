import os
import csv
from pytorch_lightning import Callback

class MetricsLogger(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.file_path = os.path.join(save_dir, "metrics.csv")
        self.fieldnames = ["epoch", "train_loss", "val_loss", "train_metric", "val_metric"]

    def on_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics["train_loss"]
        val_loss = trainer.callback_metrics["val_loss"]
        train_acc = trainer.callback_metrics["train_acc"]
        val_acc = trainer.callback_metrics["val_acc"]

        # Create or append to CSV file
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()

        with open(self.file_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow({
                "epoch": trainer.current_epoch,
                "train_loss": train_loss.item(),
                "val_loss": val_loss.item(),
                "train_acc": train_acc.item(),
                "val_acc": val_acc.item()
            })


class LossLogger(Callback):
    def __init__(self, save_dir, time):
        super().__init__()
        self.save_dir = save_dir
        self.train_losses = []
        self.val_losses = []
        self.time = time   # Save file identifier

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss'].item()
        self.train_losses.append(loss)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        loss = outputs.item()
        self.val_losses.append(loss)

    def on_train_end(self, trainer, pl_module):
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, '{}_losses.csv'.format(self.time)), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
            for epoch, (train_loss, val_loss) in enumerate(zip(self.train_losses, self.val_losses)):
                writer.writerow([epoch+1, train_loss, val_loss])