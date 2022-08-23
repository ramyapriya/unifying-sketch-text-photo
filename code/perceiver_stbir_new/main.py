import os
from torchvision import transforms
from torch.utils.data import DataLoader
from options import opts
from model import TripletNetwork
from dataloader import CustomSketchyCOCO

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml

def load_config(config_file):
    assert os.path.exists(config_file), 'Configuration file not found'
    with open(config_file) as file:
        cfg = yaml.safe_load(file)
    assertion_error = 'Model params not found/declared correctly'
    assert all(keys in cfg for keys in ('model_params', 'training_params')), assertion_error 
    return cfg

if __name__ == '__main__':
    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = CustomSketchyCOCO(opts, mode='val',
        transform=dataset_transforms)

    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    print('Val DataLoader loaded with %d items' % len(val_dataset))
    
    ckpt_path = '/vol/research/sketchcaption/ramya/dissertation_exps/training_runs/logs_july14th_v1/saved_model/sketchycoco-epoch=190-top10=0.67.ckpt'
    model = TripletNetwork.load_from_checkpoint(ckpt_path)
    trainer = Trainer(gpus=1)
    trainer.validate(model, val_loader)
