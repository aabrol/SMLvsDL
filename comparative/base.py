import copy
from dataclasses import dataclass
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from sklearn import metrics
from scipy.stats import pearsonr

from data import get_dataloaders
from utils import torch_stack_dicts, sanitize_dict


@dataclass
class Config:

    sample_size: int = 10000
    repetition_num: int = 0
    sample_splits_dir: str = '/data/users2/AA/Comp_SL_ML/SampleSplits_Age/'
    num_workers: int = 8
    batch_size: int = 16
    scorename: str = 'age'

    learning_rate: float = 0.01
    weight_decay: float = 0.001
    channels: Tuple[int, ...] = (32, 64, 128, 256, 256, 64)
    dropout: bool = True
    lr_decay: float = 0.3
    arch: str = 'peng'


def train(cfg: Config):
    if cfg.arch == 'peng':
        from peng import PengNet as Model
        epochs = 150
    else:
        raise ValueError

    model = Model(cfg)

    trainer = Trainer(gpus=1, progress_bar_refresh_rate=0,
                      max_epochs=epochs,
                      distributed_backend='ddp',
                      default_root_dir=f'./rep_{cfg.repetition_num}_{cfg.scorename}'
                      )
    trainer.fit(model)
    trainer.test()
#    trainer.test(ckpt_path=None)


class BaseNet(pl.LightningModule):
    def __init__(self, hparams: Config):
        super().__init__()
        torch.set_num_threads(8)
        self.hparams = sanitize_dict(
            copy.copy(vars(hparams)))

        self.dataloaders, self.regression, self.output_dim = get_dataloaders(
            hparams.sample_size, hparams.repetition_num, hparams.sample_splits_dir, hparams.num_workers, hparams.batch_size, hparams.scorename)

        self.learning_rate = hparams.learning_rate
        self.weight_decay = hparams.weight_decay
        self.channels = hparams.channels
        self.dropout = hparams.dropout
        self.lr_decay = hparams.lr_decay

        # print(self.output_dim)
        # print(self.dropout)

        self.build_model()

    def build_model(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['val']

    def test_dataloader(self):
        return self.dataloaders['test']

    def loss(self, x, y, postfix='train'):
        logits = self.forward(x)

        if self.regression:
            logits = logits.view(logits.size()[:1])
            loss = F.mse_loss(logits, y)
            y_pred = logits.detach().cpu()
            y = y.cpu()
            r2 = metrics.r2_score(y, y_pred)
            mae = metrics.mean_absolute_error(y, y_pred)
            r, _ = pearsonr(y, y_pred)
            log = {
                'loss': loss,
                'm1': torch.tensor(r2, dtype=torch.float),
                'm2': torch.tensor(mae, dtype=torch.float),
                'm3': torch.tensor(r, dtype=torch.float),
            }
        else:
            logits = logits.view(logits.size()[:2])
            loss = F.cross_entropy(logits, y)
            y_pred = logits.argmax(1)
            acc = y_pred.eq(y).float().mean()
            log = {
                'loss': loss,
                'm1': acc,
                'm2': acc,
                'm3': acc,
            }

        log = {f'{k}/{postfix}': log[k] for k in log}
        log['lr'] = torch.tensor(self.optimizer.param_groups[0]['lr'])

        return log

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss_dict = self.loss(x, y, postfix='train')
        return {'loss': loss_dict['loss/train'],
                'progress_bar': {'train_m1': loss_dict['m1/train'], 'train_m2': loss_dict['m2/train'], 'train_m3': loss_dict['m3/train']},
                'log': loss_dict}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss_dict = self.loss(x, y, postfix='val')
        return loss_dict

    def validation_epoch_end(self, outputs):
        loss_dict = torch_stack_dicts(outputs)
        return {'progress_bar': {'val_m1': loss_dict['m1/val'], 'val_m2': loss_dict['m2/val'], 'val_loss': loss_dict['loss/val'], 'val_m3': loss_dict['m3/val']}, 'log': loss_dict}

    def test_step(self, batch, batch_nb):
        x, y = batch
        loss_dict = self.loss(x, y, postfix='test')
        return loss_dict

    def test_epoch_end(self, outputs):
        loss_dict = torch_stack_dicts(outputs)
        return {'progress_bar': {'test_m1': loss_dict['m1/test'], 'test_m2': loss_dict['m2/test'], 'test_loss': loss_dict['loss/test'], 'test_m3': loss_dict['m3/test']}, 'log': loss_dict}

    def configure_optimizers(self):
        raise NotImplementedError
