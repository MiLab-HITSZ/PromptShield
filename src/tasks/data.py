import os
import csv
import torch
import random
import logging
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb

from utils.augment_text import _augment_text
from utils.augment_image import _augment_image
from backdoor.utils import apply_trigger

ImageFile.LOAD_TRUNCATED_IMAGES = True
    
class ImageCaptionDataset(Dataset):
    """
    return: caption, attention_mask, image

    """
   
    def __init__(self, path, processor, num_samples=0, image_key="image", caption_key="caption", delimiter=",", inmodal = False, defense = False, crop_size = 150):
        
        df = pd.read_csv(path, sep = delimiter)
        # 去掉 NaN 和 空字符串的条目
        df = df.dropna(subset=['image'])
        df = df[df['image'].str.strip() != '']
        # 更新索引
        df = df.reset_index(drop=True)
        
        # 随机抽取num_samples个样本
        if num_samples> 0:
            df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

        if image_key is None and caption_key is None:
            image_key = df.iloc[0, 0]
            caption_key = df.iloc[0, 1]

        self.root = os.path.dirname(path)
  
        self.images = df[image_key].tolist()
        self.captions_text = df[caption_key].tolist()

        self.captions = processor.process_text(self.captions_text)
        self.processor = processor


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        item["image_path"] = self.images[idx]

        image = Image.open(os.path.join(self.root, self.images[idx]))

        item["caption"] = self.captions_text[idx]
        item["image"] = self.processor.process_image(image)

        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.process_image(image)
    
        return item
    

class ImageLabelDataset(Dataset):
    def __init__(self, root, classes_path, transform):

        self.root = root  
        df = pd.read_csv(os.path.join(root, 'labels.csv'))
        
        self.images = df["image"]
        self.labels = df["label"]
        self.classes_path = classes_path
        self.transform = transform
     
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
    
        return image, label


def get_train_dataset(data_type, train_data_dir):
    # if(not options.linear_probe or not options.finetune or options.eval_train_data_dir is None): return
    if(train_data_dir is None): return

    transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    if(data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = train_data_dir)
    elif(data_type == "CIFAR-10"):
        dataset = torchvision.datasets.CIFAR10(root = train_data_dir, download = True, train = True, transform=transform)
    elif(data_type == "CIFAR-100"):
        dataset = torchvision.datasets.CIFAR100(root = train_data_dir, download = False, train = True, transform=transform)
    elif(data_type == "DTD"):
        dataset = torch.utils.data.ConcatDataset(
            [torchvision.datasets.DTD(root = train_data_dir, download = True, split = "train", transform=transform), 
            torchvision.datasets.DTD(root = os.path.dirname(train_data_dir), download = True, split = "val", transform =transform)]
        )
    elif(data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = train_data_dir, download = True, split = "trainval", transform =transform)
    elif(data_type == "Flowers102"):
        dataset = ImageLabelDataset(root = train_data_dir)
    elif(data_type == "Food-101"):
        dataset = torchvision.datasets.Food101(root = train_data_dir, download = True, split = "train", transform =transform)
    elif(data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = train_data_dir, download = True, split = "train", transform =transform)
    elif(data_type == "ImageNet1K"):
        dataset = ImageLabelDataset(root = train_data_dir)
    elif(data_type == "Oxford-IIIT-Pet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = train_data_dir, download = True, split = "trainval", transform =transform)
    elif(data_type == "Rendered SST-2"):
        dataset = torchvision.datasets.RenderedSST2(root = train_data_dir, download = True, split = "train", transform =transform)
    elif(data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = train_data_dir, download = True, split = "train", transform =transform)
    elif(data_type == "STL-10"):
        dataset = torchvision.datasets.STL10(root = train_data_dir, download = True, split = "train", transform =transform)
    elif(data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = train_data_dir, download = True, split = "train", transform =transform)
    else:
        raise Exception(f"Eval train dataset type {data_type} is not supported")
    
    return dataset


def get_test_dataset(data_type, test_data_dir):

    if(data_type is None): return

    transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
        
    if(data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = test_data_dir)
    elif(data_type == "CIFAR-10"):
        dataset = torchvision.datasets.CIFAR10(root = test_data_dir, download = True, train = False, transform =transform)
    elif(data_type == "CIFAR-100"):
        dataset = torchvision.datasets.CIFAR100(root = test_data_dir, download = True, train = False, transform =transform)
    elif(data_type == "DTD"):
        dataset = torchvision.datasets.DTD(root = test_data_dir, download = True, split = "test", transform =transform)
    elif(data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = test_data_dir, download = True, split = "test", transform =transform)
    elif(data_type == "Flowers102"):
        dataset = ImageLabelDataset(root = test_data_dir)
    elif(data_type == "Food-101"):
        dataset = torchvision.datasets.Food101(root = test_data_dir, download = True, split = "test", transform =transform)
    elif(data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = test_data_dir, download = True, split = "test", transform =transform)
    elif(data_type == "ImageNet1K"):
        dataset = ImageLabelDataset(root = test_data_dir)
    elif(data_type == "Oxford-IIIT-Pet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = test_data_dir, download = True, split = "test", transform =transform)
    elif(data_type == "Rendered SST-2"):
        dataset = torchvision.datasets.RenderedSST2(root = test_data_dir, download = True, split = "test", transform =transform)
    elif(data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = test_data_dir, download = True, split = "test", transform =transform)
    elif(data_type == "STL-10"):
        dataset = torchvision.datasets.STL10(root = test_data_dir, download = True, split = "test", transform =transform)
    elif(data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = test_data_dir, download = True, split = "test", transform =transform)
    elif(data_type in ["ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R"]):
        dataset = ImageLabelDataset(root = test_data_dir)
    else:
        raise Exception(f"Eval test dataset {data_type} is not supported")

    return dataset

def get_eval_train_dataloader(options, processor):
    # if(not options.linear_probe or not options.finetune or options.eval_train_data_dir is None): return
    if(options.eval_train_data_dir is None): return

    if(options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = os.path.dirname(options.eval_train_data_dir), download = True, train = True, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = os.path.dirname(options.eval_test_data_dir), download = True, train = True, transform = processor.process_image)
    elif(options.eval_data_type == "DTD"):
        dataset = torch.utils.data.ConcatDataset([torchvision.datasets.DTD(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image), torchvision.datasets.DTD(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "val", transform = processor.process_image)])
    elif(options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "trainval", transform = processor.process_image)
    elif(options.eval_data_type == "Flowers102"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "ImageNet1K"):
        options.add_backdoor = False
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image, options = options)
    elif(options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "trainval", transform = processor.process_image)
    elif(options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    else:
        raise Exception(f"Eval train dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.linear_probe_batch_size, num_workers = options.num_workers, sampler = None, shuffle = True)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def get_eval_test_dataloader(options, processor):
    if(options.eval_test_data_dir is None): return

    if(options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = os.path.dirname(options.eval_test_data_dir), download = False, train = False, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = os.path.dirname(options.eval_test_data_dir), download = False, train = False, transform = processor.process_image)
    elif(options.eval_data_type == "DTD"):
        dataset = torchvision.datasets.DTD(root = os.path.dirname(options.eval_test_data_dir), download = False, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_test_data_dir), download = False, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "Flowers102"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root = os.path.dirname(options.eval_test_data_dir), download = False, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = os.path.dirname(options.eval_test_data_dir), download = False, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "ImageNet1K"):
        print(f'Test: {options.add_backdoor}')
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image, options = options)
    elif(options.eval_data_type == "Oxford-IIIT-Pet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_test_data_dir), download = False, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "Rendered SST-2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_test_data_dir), download = False, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_test_data_dir), download = False, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_test_data_dir), download = False, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_test_data_dir), download = False, split = "test", transform = processor.process_image)
    elif(options.eval_data_type in ["ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R"]):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    else:
        raise Exception(f"Eval test dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader