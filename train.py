import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from model import Projection, Encoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from args import get_args


class CutPaste(pl.Lighparse_argstningModule):
    def __init__(self, hparams):
        super(CutPaste, self).__init__()
        self.save_hyperparameters()
        self.encoder = Encoder()
        self.projection_head = Projection()
        self.criterion = torch.nn.CrossEntropyLoss()


    def forward(self, x):
        features = self.encoder(x)
        logits, embed = self.projection_head(features)
        return features, logits, embeds

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.learninig_rate, 
                            momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer)
        return optimizer, scheduler
    
    def on_train_start(self):  
        print('Starting training') 
    
    def on_test_start(self):
        print('Starting testing')

    def training_step(self, batch, batch_idx):
        x, y = batch
        features, logits, embeds = self(x)
        loss = self.criterion(logits, y)
        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        features, logits, embeds = self(x)
        loss = self.criterion(logits, y)
        

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
    
    def on_test_end(self, trainer, pl_module):
        print("Testing is ending")



checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./",
    filename="weights.pth",
    save_top_k=3,
    mode="min",
)    


if __name__ == "__main__":
    args = get_args()
    model = CutPaste()
    trainer = pl.Trainer.from_argparse_args(args, gpus=args.num_gpus, callbacks=[checkpoint_callback], max_epochs=args.num_epochs)
    trainer.fit(model, train_loader)
    trainer.test()