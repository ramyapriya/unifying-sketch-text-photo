# -*- coding: utf-8 -*-

import os
import glob
import json
from collections import Counter
from nltk import word_tokenize
import numpy as np
from PIL import Image, ImageOps
import torch

class CustomSketchyCOCO(torch.utils.data.Dataset):

    def __init__(self, opt, mode='train', max_len=50,
            transform=None, debug=False):
        self.opt = opt
        self.transform = transform
        self.debug = debug
        self.mode = mode
        self.max_len = max_len
        assert self.mode in ['train', 'val'], ValueError('Invalid mode selected; should be train or val only')
        
        train_file = os.path.join(opt.root_dir, opt.split, 'train.txt')
        val_file = os.path.join(opt.root_dir, opt.split, 'val.txt')
        
        assert os.path.exists(train_file), 'Train file not found'
        assert os.path.exists(val_file), 'Val file not found'
        self.all_image_files = glob.glob(os.path.join(self.opt.root_dir, 'images', '*', '*.jpg'))
        if self.mode == 'train':
            self.all_ids = [i.strip() for i in open(train_file).readlines()]
        else:
            self.all_ids = [i.strip() for i in open(val_file).readlines()]


    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, index):
        filename = self.all_ids[index]

        sketch_file = glob.glob(os.path.join(self.opt.root_dir, 'sketchycoco', '*', '%s.png'%filename))[0]
        image_file = glob.glob(os.path.join(self.opt.root_dir, 'images', '*', '%s.jpg'%filename))[0]
        negative_file = np.random.choice(self.all_image_files, 1)[0]

        assert os.path.splitext(os.path.split(sketch_file)[-1])[0] == os.path.splitext(os.path.split(image_file)[-1])[0], ValueError('file mismatch')

        sketch_data = Image.open(sketch_file).convert('RGB')
        image_data = Image.open(image_file).convert('RGB')
        negative_data = Image.open(negative_file).convert('RGB')

        sketch_data = ImageOps.pad(sketch_data, size=(self.opt.max_len, self.opt.max_len))
        image_data = ImageOps.pad(image_data, size=(self.opt.max_len, self.opt.max_len))
        negative_data = ImageOps.pad(negative_data, size=(self.opt.max_len, self.opt.max_len))

        
        if self.transform:
            sk_tensor = self.transform(sketch_data)
            img_tensor = self.transform(image_data)
            neg_tensor = self.transform(negative_data)
        
        if self.debug:
            return sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data
        else:
            return sk_tensor, img_tensor, neg_tensor


if __name__ == '__main__':
    from options import opts
    from torchvision import transforms

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomSketchyCOCO(opts, mode='train', transform=dataset_transforms, debug=True)

    for idx, (sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data) in enumerate(dataset):

       
        
        sketch_data.save(os.path.join(output_dir, '%d_sk.jpg'%idx))
        image_data.save(os.path.join(output_dir, '%d_img.jpg'%idx))
        negative_data.save(os.path.join(output_dir, '%d_neg.jpg'%idx))

        print('Shape of sk_tensor: {} | img_tensor: {} | neg_tensor: {}'.format(sk_tensor.shape, img_tensor.shape, neg_tensor.shape))
