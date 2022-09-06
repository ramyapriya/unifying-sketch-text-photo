import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from options import opts
from model import TripletNetwork
from dataloader import CustomSketchyCOCO

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':

    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Our Dataset
    val_dataset = CustomSketchyCOCO(opts, mode='val',
        transform=dataset_transforms)

    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    model = TripletNetwork()#.load_from_checkpoint('saved_model/sketchyscene-epoch=14-top10=0.35.ckpt')
    
    trainer = Trainer(logger=False, gpus=-1)

    metrics = trainer.validate(model, val_loader)