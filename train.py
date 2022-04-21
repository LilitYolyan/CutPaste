import torch
from torch.nn import functional as F
from torchvision import transforms
import torch.utils.data
import pytorch_lightning as pl
from model import CutPasteNet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import MVTecAD
from torch import optim
from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained',  action='store_true',
                        help='bool value to indicate weather to use pretrained weight for encoder')
    parser.add_argument('--dataset_path', help='path to trainset with category name, eg: "../data/MVTecAD/wood/train')
    parser.add_argument('--dims', default=[512, 512, 512, 512, 512, 512, 512, 512, 128],
                        help='list indicating number of hidden units for each layer of projection head')
    parser.add_argument('--num_class', default=3)
    parser.add_argument('--encoder', default='resnet18')
    parser.add_argument('--learning_rate', default=0.03)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=0.00003)
    parser.add_argument('--num_epochs', default=300)
    parser.add_argument('--num_gpus', default=1)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--log_dir', default=r'tb_logs')
    parser.add_argument('--log_dir_name', default=r'exp1')
    parser.add_argument('--checkpoint_filename', default=r'weights')
    parser.add_argument('--monitor_checkpoint', default=r'train_loss')
    parser.add_argument('--monitor_checkpoint_mode', default=r'min')
    parser.add_argument('--localization',  action='store_true',
                        help='If True train on (64,64) cropped patches')

    args = parser.parse_args()
    return args


class CutPaste(pl.LightningModule):
    def __init__(self, hparams):
        super(CutPaste, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = CutPasteNet(encoder = hparams.encoder, pretrained = hparams.pretrained, dims = hparams.dims, num_class = int(hparams.num_class))
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def train_dataloader(self):
        dataset = MVTecAD(train_images = self.hparams.dataset_path, image_size = self.hparams.input_size, mode = 'train',
                          localization = self.hparams.localization)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            shuffle=True
        )
        return loader
    
    def test_dataloader(self):
        dataset = MVTecAD(train_images = self.hparams.dataset_path, image_size = self.hparams.input_size, mode = 'test',
                          localization=self.hparams.localization)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )
        return loader


    def forward(self, x):
        logits, embeds = self.model(x)
        return  logits, embeds

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate, 
                            momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.hparams.num_epochs)
        return [optimizer], [scheduler]
    
    def on_train_start(self):  
        print('Starting training') 
    
    def on_test_start(self):
        print('Starting testing')

    def training_step(self, batch, batch_idx):
        x = torch.cat(batch, axis=0)
        y = torch.arange(len(batch))
        y = y.repeat_interleave(len(batch[0])).cuda()
        logits, embeds = self(x)
        loss = self.criterion(logits, y)
        predicted = torch.argmax(logits,axis=1)
        accuracy = torch.true_divide(torch.sum(predicted==y), predicted.size(0))
        self.log("train_loss", loss)
        self.log("train_acc", accuracy)
        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, embeds = self(x)
        loss = self.criterion(logits, y)
        

    def on_train_end(self):
        print("Training is ending")
    
    def on_test_end(self):
        print("Testing is ending")



if __name__ == "__main__":
    args = get_args()
    # change default checkpoint_filename: "weights" to "weights-<defect name>"
    NAME_CKPT = args.checkpoint_filename +"-"+ Path(args.dataset_path).parent.stem if args.checkpoint_filename == "weights" else args.checkpoint_filename
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
        filename=NAME_CKPT,
        mode = args.monitor_checkpoint_mode)
    model = CutPaste(hparams = args)
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, gpus=args.num_gpus, callbacks=[checkpoint_callback], max_epochs=args.num_epochs)
    trainer.fit(model)
