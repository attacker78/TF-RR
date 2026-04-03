import sys

print('Python %s on %s' % (sys.version, sys.platform))
import torch

torch.multiprocessing.set_start_method('spawn', force=True)
torch.set_num_threads(1)
import logging
import math
import numpy as np
import pytorch_lightning as pl
import os
import torch

# If serialization fails due to the S4 module, please add the code as shown in the comment to restrict multiprocessing
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use only single GPU
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Set Lightning to use single process
os.environ["PL_FAULT_TOLERANT"] = "0"
"""

from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss
import torch.optim as optim
from pytorch_lightning.callbacks import RichProgressBar  # Progress bar
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset
from dataset import ImDataModule
from callbacks import LossLogger
from model_freq_time import FED_s4

save_file = './result/'

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def NegPearsonCorrelation_Loss(y_true, y_pred):
    # Calculate mean
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)

    # Calculate covariance
    covar = torch.mean((y_true - mean_true) * (y_pred - mean_pred))

    # Calculate standard deviation
    std_true = torch.std(y_true)
    std_pred = torch.std(y_pred)

    # Calculate negative Pearson correlation coefficient
    pearson_corr = covar / (std_true * std_pred + 1e-8)

    # Return 1 minus negative Pearson correlation coefficient as loss
    return 1 - pearson_corr


def ConcordanceCorrelationCoefficient_Loss(y_true, y_pred):
    # Calculate mean
    mean_pred = torch.mean(y_pred)
    mean_true = torch.mean(y_true)

    # Calculate covariance
    covar = torch.mean((y_pred - mean_pred) * (y_true - mean_true))

    # Calculate variance
    var_pred = torch.var(y_pred)
    var_true = torch.var(y_true)

    # Calculate Concordance Correlation Coefficient
    ccc = 2 * covar / (var_pred + var_true + (mean_pred - mean_true) ** 2)

    # Convert CCC to Loss, aim is to maximize CCC, so take 1 - CCC
    loss = 1 - ccc

    return loss


class PLModel(pl.LightningModule):
    def __init__(
            self,
            fds=False,
            d_model=256,
            learning_rate=0.001,
            start_update=0,
            start_smooth=1,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            scheduler_name="cosine",
            steps_per_epoch=None,
            optimizer_name='adamw',
            mode='TBPTT',
            scaler=None,
            log_prefix="",
            file_path='',
            **scheduler_kwargs,
    ):
        super().__init__()
        self.fds = fds
        self.start_update = start_update
        self.start_smooth = start_smooth
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.d_model = d_model
        self.optimizer_name = optimizer_name
        self.betas = betas
        self.scheduler_name = scheduler_name
        self.steps_per_epoch = steps_per_epoch
        self.scaler = scaler
        self.scheduler_kwargs = scheduler_kwargs
        self.log_prefix = log_prefix
        self.mode = mode
        # self.model = vgg_model
        self.file_path = file_path
        self._build_model()

    def _build_model(self):
        self.model = FED_s4(d_input=2)

    def training_epoch_end(self, outputs):
        # if self.fds and self.current_epoch >= self.start_update:
        # Check if self.current_features exists, starting from epoch
        if hasattr(self, 'current_features') and self.current_features is not None and self.fds:
            with torch.no_grad():
                # This is called at the end of each epoch
                # if self.current_epoch >= self.start_update:
                # Collect features and labels for all training samples
                # Example:
                training_features, training_labels = self.current_features, self.current_labels,
                self.model.FDS.update_last_epoch_stats(self.current_epoch)
                self.model.FDS.update_running_stats(training_features, training_labels, self.current_epoch)
        else:
            pass

    def forward(self, batch):
        x, y = batch  # Input data, target data
        current_epoch = self.current_epoch
        if self.fds and self.training:
            y_hat, feature = self.model(x, y, current_epoch)
            return y_hat, y, feature
        else:
            y_hat = self.model(x)
            return y_hat, y

    def _shared_step(self, batch, mode, batch_idx=None, prefix="train"):
        # x, y = batch   # Input data, target data
        if self.training and self.fds:
            y_hat, y, weight, feature = self.forward(batch=batch)
            y_hat = y_hat.squeeze()
            loss = NegPearsonCorrelation_Loss(y_true=y, y_pred=y_hat)
            metric = l1_loss(y_hat.view(-1), y)
            return loss, metric, y_hat, y, feature

        else:
            y_hat, y = self.forward(batch=batch)
            y_hat = y_hat.squeeze()
            loss1 = NegPearsonCorrelation_Loss(y_true=y, y_pred=y_hat)
            loss2 = mse_loss(input=y_hat, target=y)
            a = 0.4
            loss = a * loss1 + ((1 - a) * loss2)
            metric = ConcordanceCorrelationCoefficient_Loss(y_pred=y_hat, y_true=y)
            return loss, metric, y_hat, y
        # loss = mse_loss(y_hat.view(-1), y)
        # loss = weighted_l1_loss(y_hat.view(-1), y, weights=weight.view(-1))
        # metric = l1_loss(y_hat.view(-1), y)
        # metric = weighted_focal_l1_loss(inputs=y_hat.view(-1), targets=y, weights=weight.view(-1))
        # self.log(f'{mode}_loss',loss, f'{mode}_metrics',metric,on_step=False, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=False)
        # self.log(f'{mode}_metrics',metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=False)

        # return loss, metric, y_hat, y

    def training_step(self, batch, batch_idx):
        # Need to update FDS at each training_epoch
        if self.fds:
            train_loss, train_metric, y_hat, y, train_feature = self._shared_step(batch=batch, mode='train',
                                                                                  batch_idx=batch_idx,
                                                                                  prefix="train")
            self.log('train_loss', train_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True,
                     add_dataloader_idx=False)
            self.current_output = y_hat
            self.current_labels = y
            self.current_features = train_feature

            return train_loss
        else:
            train_loss, train_metric, y_hat, y = self._shared_step(batch=batch, mode='train', batch_idx=batch_idx,
                                                                   prefix="train")
            self.log('train_loss', train_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True,
                     add_dataloader_idx=False)
            # self.current_features = None
            return train_loss
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger

        # metrics = {"train_loss": train_loss, "train_metric": train_metric}
        # self.log("train_loss", train_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True,add_dataloader_idx=False,)
        # self.log("train_metrics",metrics, on_step=True, on_epoch=False, prog_bar=False, logger=True, add_dataloader_idx=False,)
        # self.log('train_loss', train_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True,
        # add_dataloader_idx=False)

        # return train_loss, y_hat, y

    def validation_step(self, batch, batch_idx):
        if self.training:
            validation_loss, validation_metric, _, _, _ = self._shared_step(batch=batch, mode='validation',
                                                                            prefix="validation")
            self.log('validation_loss', validation_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     add_dataloader_idx=False)
            return validation_loss  # , validation_metric
        else:
            validation_loss, validation_metric, _, _ = self._shared_step(batch=batch, mode='validation',
                                                                         prefix="validation")
            self.log('validation_loss', validation_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     add_dataloader_idx=False)
            return validation_loss

    def test_step(self, batch, batch_idx):
        if self.training:
            test_loss, test_metric, _, _, _ = self._shared_step(batch=batch, mode='test', prefix="test")
            self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     add_dataloader_idx=False)
            return test_loss  # , test_metric
        else:
            test_loss, test_metric, _, _ = self._shared_step(batch=batch, mode='test', prefix="test")
            self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     add_dataloader_idx=False)
            return test_loss  # , test_metric

    # Set optimizer
    def configure_optimizers(self):
        all_params = list(self.parameters())
        if self.optimizer_name == 'adamw':
            optimizer = optim.AdamW(params=self.parameters(), lr=self.lr, betas=self.betas,
                                    weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        # Create constant learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=[get_constant_schedule()],last_epoch=0,verbose=False)
        # scheduler = get_constant_schedule(optimizer)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch',
                                 # Learning rate scheduling interval (per epoch)
                                 'frequency': 1,
                                 'monitor': 'val/loss'}  # Scheduling frequency (per epoch)
                }
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Set data directories
    data_dir = os.path.join(current_dir, 'data', 'capnobase')
    result_dir = os.path.join(current_dir, 'results')
    checkpoint_dir = os.path.join(current_dir, 'checkpoint', 'c')
    loss_dir = os.path.join(current_dir, 'results', 'loss', 'c')
    train_result_dir = os.path.join(result_dir, 'train_capnobase')
    valid_result_dir = os.path.join(result_dir, 'valid_capnobase')
    test_result_dir = os.path.join(result_dir, 'test_capnobase')

    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(train_result_dir, exist_ok=True)
    os.makedirs(valid_result_dir, exist_ok=True)
    os.makedirs(test_result_dir, exist_ok=True)

    # Save results, Model, Loss, Result
    for i in range(49):
        # Dataset
        # Define dataset
        print("######################### Training {} ####################".format(i))

        trainx = np.load(os.path.join(data_dir, 'trainx_{}.npy'.format(i)))
        trainy = np.load(os.path.join(data_dir, 'trainy_{}.npy'.format(i)))
        validx = np.load(os.path.join(data_dir, 'validx_{}.npy'.format(i)))
        validy = np.load(os.path.join(data_dir, 'validy_{}.npy'.format(i)))
        testx = np.load(os.path.join(data_dir, 'testx_{}.npy'.format(i)))
        testy = np.load(os.path.join(data_dir, 'testy_{}.npy'.format(i)))

        train_dataset = TensorDataset(torch.FloatTensor(trainx), torch.FloatTensor(trainy))
        valid_dataset = TensorDataset(torch.FloatTensor(validx), torch.FloatTensor(validy))
        test_dataset = TensorDataset(torch.FloatTensor(testx), torch.FloatTensor(testy))

        data_module1 = ImDataModule(train_dataset, valid_dataset, test_dataset, batch_size=40)

        import argparse
        import time

        loca = time.strftime('%Y-%m-%d-%H-%M-%S')
        new_name = str(loca)

        parser = argparse.ArgumentParser("Begin Train")
        parser.add_argument("-is_train", default='no', type=str, choices=['yes', 'no'])

        # FDS
        parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
        parser.add_argument('--fds_kernel', type=str, default='gaussian',
                            choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
        parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
        parser.add_argument('--fds_sigma', type=float, default=3, help='FDS gaussian/laplace kernel sigma')
        parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
        parser.add_argument('--start_smooth', type=int, default=1,
                            help='which epoch to start using FDS to smooth features')
        parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
        parser.add_argument('--bucket_start', type=int, default=0, choices=[0, 3],
                            help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
        parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

        # args = parser.parse_args()
        args1, unknown = parser.parse_known_args()
        # Fix random seeds for pytorch, numpy, python.random
        pl.seed_everything(0, workers=True)

        print("############# Whether to start training #############", args1.is_train)
        if args1.is_train == 'yes':  # Training
            # Define dataset for training+validation stage
            # data_module.setup(stage='fit')

            # callbacks
            checkpoint_callback = ModelCheckpoint(
                monitor="validation_loss",
                mode="min",
                dirpath=checkpoint_dir,
                filename=str(i) + "_capnobase_best_model",
                save_last=True,
                save_top_k=1,
                auto_insert_metric_name=False,
                verbose=True
            )

            early_stopping_callback = EarlyStopping(monitor="validation_loss", mode="min", patience=5)

            lr_monitor = LearningRateMonitor(logging_interval="step")

            loss_callback = LossLogger(save_dir=loss_dir, time=new_name)

            trainer = pl.Trainer(accelerator="gpu",
                                 callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor, loss_callback],
                                 max_epochs=30, )
            # Trainer(callbacks=[RichProgressBar()])
            print('=====> Building model...')
            model = PLModel(fds=False)
            trainer.fit(model, train_dataloaders=data_module1.train_dataloader(),
                        val_dataloaders=data_module1.val_dataloader())
            print("Capnobase Training DONE.")

        else:
            import pickle
            from tqdm import tqdm
            # Set device
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            print(f"Using device: {device}")
            checkpoint_path = os.path.join(checkpoint_dir, '{}_capnobase_best_model.ckpt'.format(i))
            model = PLModel.load_from_checkpoint(checkpoint_path)
            model = model.to(device)  # Move model to GPU
            model.eval()

            # Training set prediction
            train_data = data_module1.train_dataloader()
            train_predictions = []
            train_reals = []
            print("Starting training set inference...")

            with torch.no_grad():
                # Add tqdm progress bar
                for batch in tqdm(train_data, desc="Training set progress", unit="batch"):
                    # Move data to GPU
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)

                    batch_predictions = model((x, y))  # Pass data moved to GPU
                    train_predictions.append(batch_predictions[0].cpu())  # Move results back to CPU for saving
                    train_reals.append(batch_predictions[1].cpu())

            # Save results...
            with open(os.path.join(train_result_dir, 'train_predictions{}.pkl'.format(i)), 'wb') as f:
                pickle.dump(train_predictions, f)

            with open(os.path.join(train_result_dir, 'train_reals{}.pkl'.format(i)), 'wb') as f:
                pickle.dump(train_reals, f)

            print('=====> Train Dataset Predict Done...')

            # Validation set with tqdm
            valid_data = data_module1.val_dataloader()
            valid_predictions = []
            valid_reals = []

            print("Starting validation set inference...")

            with torch.no_grad():
                for batch in tqdm(valid_data, desc="Validation set progress", unit="batch"):
                    # Move data to GPU
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)

                    batch_predictions = model((x, y))
                    valid_predictions.append(batch_predictions[0].cpu())
                    valid_reals.append(batch_predictions[1].cpu())

            # Save model predictions
            with open(os.path.join(valid_result_dir, 'valid_predictions{}.pkl'.format(i)), 'wb') as f:
                pickle.dump(valid_predictions, f)

            with open(os.path.join(valid_result_dir, 'valid_reals{}.pkl'.format(i)), 'wb') as f:
                pickle.dump(valid_reals, f)

            print('=====> Validation Dataset Predict Done...')

            # Test set with tqdm
            test_data = data_module1.test_dataloader()
            test_predictions = []
            test_reals = []

            print("Starting test set inference...")

            with torch.no_grad():
                for batch in tqdm(test_data, desc="Test set progress", unit="batch"):
                    # Move data to GPU
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)

                    batch_predictions = model((x, y))
                    test_predictions.append(batch_predictions[0].cpu())
                    test_reals.append(batch_predictions[1].cpu())

            # Save model predictions
            with open(os.path.join(test_result_dir, 'test_predictions{}.pkl'.format(i)), 'wb') as f:
                pickle.dump(test_predictions, f)

            with open(os.path.join(test_result_dir, 'test_reals{}.pkl'.format(i)), 'wb') as f:
                pickle.dump(test_reals, f)

            print('=====> Test Dataset Predict Done...')


if __name__ == '__main__':
    main()

