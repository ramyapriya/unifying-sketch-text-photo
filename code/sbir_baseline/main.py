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
    train_dataset = CustomSketchyCOCO(opts, mode='train',
        transform=dataset_transforms)
    val_dataset = CustomSketchyCOCO(opts, mode='val',
        transform=dataset_transforms)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    model = TripletNetwork()#.load_from_checkpoint(checkpoint_path="saved_model/our-dataset-epoch=103-top10=0.52.ckpt")

    logger = TensorBoardLogger("tb_logs", name="our-dataset-scratch")
    
    checkpoint_callback = ModelCheckpoint(monitor="top5",
                mode="max",
                dirpath="saved_model",
                save_top_k=3,
                filename="our-dataset-scratch-{epoch:02d}-{top10:.2f}")

    trainer = Trainer(gpus=-1, auto_select_gpus=True, # specifies all available GPUs
                # auto_scale_batch_size=True,
                # auto_lr_find=True,
                benchmark=True,
                check_val_every_n_epoch=10,
                max_epochs=100000,
                # precision=64,
                min_steps=100, min_epochs=0,
                accumulate_grad_batches=8,
                # profiler="advanced",
                resume_from_checkpoint=None, # "some/path/to/my_checkpoint.ckpt"
                logger=logger,
                callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, val_loader)

    # Retrieve model
    checkpoint_callback.best_model_path
