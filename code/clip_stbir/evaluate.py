import numpy as np
import os
import torch
from dataloader import CustomSketchyCOCO
from options import opts
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from options import opts
from model import TripletNetwork
from dataloader import CustomSketchyCOCO
from main import load_config

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


def validation_step(self, val_batch, batch_idx):
    # defines the validation loop
    txt_tensor, txt_length, sk_tensor, img_tensor, neg_tensor = val_batch
    txt_feature = self.txt_embedding_network(txt_tensor, txt_length)
    sk_feature = self.img_embedding_network(sk_tensor)
    img_feature = self.img_embedding_network(img_tensor)
    neg_feature = self.img_embedding_network(neg_tensor)
    query_feature = self.setattn_network(sk_feature, txt_feature)
    loss = self.loss(query_feature, img_feature, neg_feature)
    self.log('val_loss', loss, prog_bar=True)
    return query_feature, img_feature
    
    
if __name__ == "__main__":
    ckpt_path = '/vol/research/sketchcaption/ramya/dissertation_exps/training_runs/logs_july14th_v1/saved_model/sketchycoco-epoch=190-top10=0.67.ckpt'
    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = CustomSketchyCOCO(opts, mode='val', transform=dataset_transforms, debug=False)

    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    print('Val DataLoader loaded with %d items' % len(val_dataset))
    
    model = TripletNetwork.load_from_checkpoint(ckpt_path)
    trainer = Trainer(gpus=1)
    trainer.test(model, val_loader)