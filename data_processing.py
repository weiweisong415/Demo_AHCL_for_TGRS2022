import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

class DatasetProcessingUCMD_21(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        fp_label = open(label_filepath, 'r')
        labels = [int(x.strip()) for x in fp_label]
        fp_label.close()
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([self.label[index]])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)

class DatasetProcessingWHURS_19(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        fp_label = open(label_filepath, 'r')
        labels = [int(x.strip()) for x in fp_label]
        fp_label.close()
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([self.label[index]])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)

class DatasetProcessingAID_30(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        fp_label = open(label_filepath, 'r')
        labels = [int(x.strip()) for x in fp_label]
        fp_label.close()
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([self.label[index]])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)



