import os
from torchvision import transforms
from torch.utils.data import DataLoader
from options import opts
from model import TripletNetwork
from dataloader import CustomSketchyCOCO

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print('Train params ', '-'*50, '\n', opts)

    train_dataset = CustomSketchyCOCO(opts, mode='train',
        transform=dataset_transforms)
    val_dataset = CustomSketchyCOCO(opts, mode='val',
        transform=dataset_transforms)

    vocab_size = len(train_dataset.word_map.items())

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    print('Train DataLoader loaded with %d items' % len(train_dataset))
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    print('Val DataLoader loaded with %d items' % len(val_dataset))
    
    seed_everything(42, workers=True)
    model = TripletNetwork(vocab_size=vocab_size, combine_type=opts.combine_type)
    
    checkpoint_callback = ModelCheckpoint(monitor='top5',
                mode='max',
                dirpath=os.path.join(opts.log_dir, 'saved_model'),
                save_top_k=3,
                filename='sketchycoco-{epoch:02d}-{top5:.2f}')
    logger = TensorBoardLogger('tb_logs', name='sketchycoco-logs')
    trainer = Trainer(gpus=1, deterministic=True, benchmark=True, max_epochs=200, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
