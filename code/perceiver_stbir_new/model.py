import torch
import torch.nn as nn
import torch.nn.functional as F
from network import VGG_Network, Txt_Encoder, SetAttention
import pytorch_lightning as pl
import torch_optimizer as optim

class TripletNetwork(pl.LightningModule):

    def __init__(self, vocab_size, input_dim, output_dim, model_params, margin=0.2, combine_type='additive'):
        super().__init__()
        self.txt_embedding_network = Txt_Encoder(vocab_size=vocab_size)
        self.img_embedding_network = VGG_Network()
        self.setattn_network = SetAttention(input_dim=input_dim,
                                            output_dim=output_dim,
                                            mode=combine_type,
                                            **model_params)
        self.loss = nn.TripletMarginLoss(margin=margin)
        self.save_hyperparameters('vocab_size', 
                                  'input_dim', 
                                  'output_dim', 
                                  'margin', 
                                  'combine_type',
                                  'model_params')
    def forward(self, x):
        feature = self.embedding_network(x)
        return feature

    def configure_optimizers(self, lr=1e-4):
        optimizer = optim.Lamb(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # defines the train loop
        txt_tensor, txt_length, sk_tensor, img_tensor, neg_tensor = batch
        txt_feature = self.txt_embedding_network(txt_tensor, txt_length)
        sk_feature = self.img_embedding_network(sk_tensor)
        img_feature = self.img_embedding_network(img_tensor)
        neg_feature = self.img_embedding_network(neg_tensor)
        query_feature = self.setattn_network(sk_feature, txt_feature)
        loss = self.loss(query_feature, img_feature, neg_feature)
        self.log('train_loss', loss, prog_bar=True)
        return loss

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
    
    def test_step(self, val_batch, batch_idx):
        # defines the validation loop
        txt_tensor, txt_length, sk_tensor, img_tensor, neg_tensor = val_batch
        txt_feature = self.txt_embedding_network(txt_tensor, txt_length)
        import ipdb; ipdb.set_trace()
        sk_feature = self.img_embedding_network(sk_tensor)
        img_feature = self.img_embedding_network(img_tensor)
        neg_feature = self.img_embedding_network(neg_tensor)
        query_feature = self.setattn_network(sk_feature, txt_feature)
        loss = self.loss(query_feature, img_feature, neg_feature)
        self.log('val_loss', loss, prog_bar=True)
        return query_feature, img_feature

    def validation_epoch_end(self, validation_step_outputs):
        Len = len(validation_step_outputs)
        query_feature_all = torch.cat([validation_step_outputs[i][0] for i in range(Len)])
        image_feature_all = torch.cat([validation_step_outputs[i][1] for i in range(Len)])

        rank = torch.zeros(len(query_feature_all))
        for idx, query_feature in enumerate(query_feature_all):
            distance = F.pairwise_distance(query_feature.unsqueeze(0), image_feature_all)
            target_distance = F.pairwise_distance(
                query_feature.unsqueeze(0), image_feature_all[idx].unsqueeze(0))
            rank[idx] = distance.le(target_distance).sum()

        self.log('top1', rank.le(1).sum().numpy() / rank.shape[0], prog_bar=True)
        self.log('top10', rank.le(10).sum().numpy() / rank.shape[0], prog_bar=True)
        self.log('meanK', rank.mean().numpy().item(), prog_bar=True)
        
    def test_epoch_end(self, validation_step_outputs):
        Len = len(validation_step_outputs)
        query_feature_all = torch.cat([validation_step_outputs[i][0] for i in range(Len)])
        image_feature_all = torch.cat([validation_step_outputs[i][1] for i in range(Len)])

        rank = torch.zeros(len(query_feature_all))
        for idx, query_feature in enumerate(query_feature_all):
            distance = F.pairwise_distance(query_feature.unsqueeze(0), image_feature_all)
            target_distance = F.pairwise_distance(
                query_feature.unsqueeze(0), image_feature_all[idx].unsqueeze(0))
            rank[idx] = distance.le(target_distance).sum()

        self.log('top1', rank.le(1).sum().numpy() / rank.shape[0], prog_bar=True)
        self.log('top10', rank.le(10).sum().numpy() / rank.shape[0], prog_bar=True)
        self.log('meanK', rank.mean().numpy().item(), prog_bar=True)
