import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from model import Projection, Encoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from args import get_args
from dataset import MVTecAD
from torch import optim

class CutPaste(pl.LightningModule):
    def __init__(self, hparams):
        super(CutPaste, self).__init__()
        self.save_hyperparameters(hparams)
        self.encoder = Encoder()
        self.projection_head = Projection()
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def train_dataloader(self):
        dataset = MVTecAD(train_images = self.hparams.dataset_path,  image_size = self.hparams.input_size, mode = 'train')
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )
        return loader
    
    def test_dataloader(self):
        dataset = MVTecAD(train_images = self.hparams.dataset_path, image_size = self.hparams.input_size, mode = 'test')
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )
        return loader


    def forward(self, x):
        features = self.encoder(x)
        logits, embeds = self.projection_head(features)
        return features, logits, embeds

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.learninig_rate, 
                            momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.hparams.num_epochs)
        return optimizer
    
    def on_train_start(self):  
        print('Starting training') 
    
    def on_test_start(self):
        print('Starting testing')

    def training_step(self, batch, batch_idx):
        x = torch.cat(batch, axis=0)
        y = torch.arange(len(batch))
        y = y.repeat_interleave(len(batch[0])).cuda()
        features, logits, embeds = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        features, logits, embeds = self(x)
        loss = self.criterion(logits, y)
        

    def on_train_end(self):
        print("Training is ending")
    
    def on_test_end(self):
        print("Testing is ending")



if __name__ == "__main__":
    from pathlib import Path
    args = get_args()
     
    logger = TensorBoardLogger(args.log_dir, name=args.log_dir_name)

    checkpoint_dir = (
    Path(logger.save_dir)
    / logger.name
    / f"version_{logger.version}"
    / "checkpoints"
    )
     
    checkpoint_callback = ModelCheckpoint(
    monitor = args.monitor_checkpoint,
    dirpath=str(checkpoint_dir),
    filename=args.checkpoint_filename,
    mode = args.monitor_checkpoint_mode)

    model = CutPaste(hparams = args)
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, gpus=args.num_gpus, callbacks=[checkpoint_callback], max_epochs=args.num_epochs)
    trainer.fit(model)