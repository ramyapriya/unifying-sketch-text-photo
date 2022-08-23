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

config = load_config(opts.config_file)
vocab_size = len(train_dataset.word_map.items())

model = TripletNetwork(vocab_size=vocab_size,
                        input_dim=config['training_params']['input_dim'], 
                        output_dim=config['training_params']['output_dim'], 
                        model_params=config['model_params'])

checkpoint_callback = ModelCheckpoint(monitor='top10',
                mode='max',
                dirpath=os.path.join(opts.log_dir, 'saved_model'),
                save_top_k=3,
                filename='sketchycoco-{epoch:02d}-{top10:.2f}')

query_feature_all_tensor, img_feature_all_tensor = [], []
for idx, (txt_tensor, txt_length, sk_tensor, img_tensor, neg_tensor,
        text_data, text_emb, sketch_data, image_data, negative_data) in enumerate(val_dataset):
    import ipdb; ipdb.set_trace()
    txt_feature = model.txt_embedding_network(txt_tensor, txt_length)
    sk_feature = model.img_embedding_network(sk_tensor)
    img_feature = model.img_embedding_network(img_tensor)
    neg_feature = model.img_embedding_network(neg_tensor)
    query_feature = model.setattn_network(sk_feature, txt_feature)
    query_feature_all_tensor.append(query_feature)
    img_feature_all_tensor.append(img_feature)
    
Len = len(query_feature_all_tensor)
query_feature_all = torch.cat([query_feature_all[i]for i in range(Len)])
image_feature_all = torch.cat([query_feature_all[i]for i in range(Len)])

rank = torch.zeros(len(query_feature_all))
for idx, query_feature in enumerate(query_feature_all):
    distance = F.pairwise_distance(query_feature.unsqueeze(0), image_feature_all)
    target_distance = F.pairwise_distance(
        query_feature.unsqueeze(0), image_feature_all[idx].unsqueeze(0))
    rank[idx] = distance.le(target_distance).sum()

print('top1', rank.le(1).sum().numpy() / rank.shape[0])
print('top10', rank.le(10).sum().numpy() / rank.shape[0])
print('meanK', rank.mean().numpy().item())