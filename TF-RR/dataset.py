import pytorch_lightning as pl
from torch.utils.data import DataLoader

class ImDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, valid_dataset, test_dataset, batch_size=40):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        # Return the training DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        # Return the validation DataLoader
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def test_dataloader(self):
        # Return the test DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )
