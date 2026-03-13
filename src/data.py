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
    
"""
自定义 Image to Caption 数据集, path指定csv文件的路径, 该文件包含样本的路径和对应的 Caption.

因此, 每一个数据集对应一个csv文件, 问题是 csv 文件由数据集自带, 还是用户自定义

"""
class ImageCaptionDataset(Dataset):
    """
    return: caption, attention_mask, image

    """
   
    def __init__(self, path,  processor, num_samples=0, image_key="image", caption_key="caption", delimiter=",", inmodal = False, defense = False, crop_size = 150):
        
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
    
class ImageToCaptionDataset(Dataset):
    """
    return: caption, attention_mask, image

    """
   
    def __init__(self, image_to_caption_path, label_path, classes_path, processor, num_samples=0, image_key="image", caption_key="caption", label_key="label", delimiter=",",  defense = False, crop_size = 150):
        logging.debug(f"Loading aligned data from {image_to_caption_path}")
        logging.debug(f"Loading labels from {label_path}")

        df = pd.read_csv(image_to_caption_path, sep = delimiter)
        label_df = pd.read_csv(label_path, sep = delimiter)
        
        if image_key is None and caption_key is None:
            image_key = df.iloc[0, 0]
            caption_key = df.iloc[0, 1]
        
        self.root = os.path.dirname(image_to_caption_path)
        
        self.images = df[image_key].tolist()
        self.captions_text = df[caption_key].tolist()
        self.labels = label_df[label_key].tolist()

         # 随机抽取num_samples个样本
        if num_samples > 0:
            indices = np.random.choice(np.arange(len(self.images)), size=num_samples, replace=False)
            self.images = [self.images[i] for i in indices]
            self.captions_text = [self.captions_text[i]  for i in indices]
            self.labels = [self.labels[i]  for i in indices] 

        # 读出提示词模板
        config  = eval(open(classes_path, "r").read())
        templates = config["templates"]
        self.classes = config["classes"]

        self.classes_text = [f"This is a photo of a {class_name}" for class_name in self.classes]

        # self.captions = processor.process_text(self.captions_text)
        self.processor = processor
        
        self.defense = defense
        if self.defense:
            self.crop_transform = transforms.RandomCrop((crop_size, crop_size))
            self.resize_transform = transforms.Resize((224, 224))
            

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        item["image_path"] = self.images[idx]
        image = Image.open(os.path.join(self.root, self.images[idx]))
    
        item["caption"] = self.captions_text[idx]

        # item["input_ids"] = self.captions["input_ids"][idx]
        # item["attention_mask"] = self.captions["attention_mask"][idx]

        item["pixel_values"] = self.processor.process_image(image)

        item["labels"] = self.labels[idx]
    
        return item
    

def calculate_scores(options, model, dataloader, epoch):

    if options.distributed:
        model = model.module  
    model.eval()

    dirname = os.path.dirname(options.train_data)
    filename = f'{options.name}_{epoch}.csv'
    path = os.path.join(dirname, filename)

    csvfile = open(path, 'a')
    csvwriter = csv.writer(csvfile)

    with torch.no_grad():
        logging.info(len(dataloader))
        for index, batch in tqdm(enumerate(dataloader)):
            image, input_ids, attention_mask = batch["pixel_values"].to(options.device), batch["input_ids"].to(options.device),  batch["attention_mask"].to(options.device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = image)
            scores  = model.logit_scale.exp() * torch.diagonal(outputs.image_embeds @ outputs.text_embeds.t())
            for j in range(len(scores)):
                csvwriter.writerow([batch['image_path'][j], batch['caption'][j], batch['is_backdoor'][j].item(), scores[j].item()])
    return path

def get_clean_train_dataloader(options, processor, path):

    logging.info(f'Creating a clean train dataloader with path {path}')

    if options.master:
        df = pd.read_csv(path, names = ['image', 'caption', 'is_backdoor', 'score'], header = None)
        df = df.sort_values(by=['score'], ascending = False)
        df_clean = df.iloc[int(options.remove_fraction * len(df)) :]
        df_dirty = df.iloc[: int(options.remove_fraction * len(df))]
        total_backdoors = sum(df['is_backdoor'].tolist())
        backdoor_detected = sum(df_dirty['is_backdoor'].tolist())
        if options.wandb:
            wandb.log({'number of backdoored images': total_backdoors,
                        'number of backdoor images removed': backdoor_detected,
                    }) 
        df_clean.to_csv(path, index = False)
        # backdoor_detected = sum(df.iloc[:5000]['is_backdoor'].tolist())
        # logging.info(f'Number of backdoors in Top-5000 examples: {backdoor_detected}')
        # for i in range(len(df)):
        #     if i < 5000:
        #         df.loc[i, 'is_backdoor'] = 1
        #     else:
        #         df.loc[i, 'is_backdoor'] = 0
        # df.to_csv(path, index = False)

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
    sampler = DistributedSampler(dataset) if(options.distributed) else None
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True)
    dataloader.num_samples = len(dataloader) * options.batch_size
    dataloader.num_batches = len(dataloader)
    return dataloader


def get_train_dataloader(train_data, image_key, caption_key, delimiter, processor):
    path = train_data
    if(path is None): return None

    # batch_size = options.batch_size

    # print(f"path:{path}\n")

    dataset = ImageCaptionDataset(path, image_key = image_key, caption_key = caption_key, delimiter = delimiter, processor = processor, inmodal = True)
        
    sampler = DistributedSampler(dataset) if(options.distributed) else None

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True)
    dataloader.num_samples = len(dataloader) * batch_size 
    dataloader.num_batches = len(dataloader)


# def get_train_dataloader(options, processor):
#     path = options.train_data
#     if(path is None): return None

#     batch_size = options.batch_size

#     # print(f"path:{path}\n")

#     dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal)
        
#     sampler = DistributedSampler(dataset) if(options.distributed) else None

#     dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True)
#     dataloader.num_samples = len(dataloader) * batch_size 
#     dataloader.num_batches = len(dataloader)

#     return dataloader

def get_validation_dataloader(options, processor):
    path = options.validation_data
    if(path is None): return

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal)
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, num_workers = options.num_workers, pin_memory = True, sampler = None, drop_last = False)
    dataloader.num_samples = len(dataset) 
    dataloader.num_batches = len(dataloader)

    return dataloader

""""
attack_config = {
    target_label,
    poisoned_rate,
    weight,
    patch_size,
    patch_type,
    patch_location
}

"""

class ImageLabelDataset(Dataset):
    def __init__(self, root, transform, attack_config=None):
        self.root = root
        # filename  = 'labels.10K.csv' if 'train50000' in root and '10K' in options.name else 'labels.5K.csv' if 'train50000' in root and '5K' in options.name else 'labels.csv'
        # print(os.path.join(root, 'labels.csv'))
        # df = pd.read_csv(os.path.join(root, filename))

        df = pd.read_csv(os.path.join(root, 'labels.csv'))
        self.images = df["image"]
        self.labels = df["label"]

        self.attack_config = attack_config
        self.target_label = attack_config["target_label"]
        self.poisoned_rate = attack_config["poisoned_rate"]

        self.attack_type = attack_config["attack_type"]
        self.patch_location = attack_config["patch_location"]
        self.classes_path = attack_config["classes_path"] 

        # print(f"self.attack_type:{self.attack_type}\n")
        # print(f"self.classes_path:{self.classes_path}\n")

        self.transform = transform
     
        poisoned_num = int(len(self.images) * self.poisoned_rate)
        tmp_list = np.arange(len(self.images))[~np.array(self.labels == self.target_label)]
        random.shuffle(tmp_list)
        self.backdoor_indices = sorted(list(tmp_list[:poisoned_num]))
 
    def __len__(self):
        return len(self.labels)

    def add_trigger(self, image, attack_type = 'blended', patch_location = 'blended', attack_config=None):
       
        return apply_trigger(image, attack_type, patch_location, attack_config=attack_config)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        label = self.labels[idx]
    
        if idx in self.backdoor_indices:
            image = self.add_trigger(image, attack_type = self.attack_type, patch_location = self.patch_location, attack_config=self.attack_config)
            label = self.target_label

        image = self.transform(image)
    
        return image, label


# class ImageLabelDataset(Dataset):
#     def __init__(self, root, transform, add_backdoor= False, backdoor_sufi =False, patch_type = 'blended', patch_size= 16):
#         self.root = root
#         # filename  = 'labels.10K.csv' if 'train50000' in root and '10K' in options.name else 'labels.5K.csv' if 'train50000' in root and '5K' in options.name else 'labels.csv'
#         # print(filename)
#         # df = pd.read_csv(os.path.join(root, filename))
#         df = pd.read_csv(os.path.join(root, 'labels.csv'))
#         self.images = df["image"]
#         self.labels = df["label"]
#         self.transform = transform

#         # self.options = options
#         self.add_backdoor = add_backdoor
#         self.backdoor_sufi = backdoor_sufi
#         self.patch_type = patch_type        
#         self.patch_size = patch_size

#         if self.backdoor_sufi:
#             self.backdoor_indices = list(range(50000))
#             shuffle(self.backdoor_indices)
#             self.backdoor_indices = self.backdoor_indices[:1000]

#     def __len__(self):
#         return len(self.labels)

#     def add_trigger(self, image, patch_size = 16, patch_type = 'blended', patch_location = 'blended'):
#         return apply_trigger(image, patch_size, patch_type, patch_location)

#     def __getitem__(self, idx):

#         image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')

#         if self.backdoor_sufi:
#             if idx in self.backdoor_indices:
#                 image = self.add_trigger(image, patch_size = self.patch_size, patch_type = self.patch_type, patch_location = self.options.patch_location)
#             label = 954
#             return image, label

#         if self.add_backdoor:
#             image = self.add_trigger(image, patch_size = self.patch_size, patch_type = self.patch_type, patch_location = self.options.patch_location)

#         image = self.transform(image)
#         label = self.labels[idx]
#         return image, label
    

# class ImageLabelDataset(Dataset):
#     def __init__(self, root, transform):
#         self.root = root
#         # filename  = 'labels.10K.csv' if 'train50000' in root and '10K' in options.name else 'labels.5K.csv' if 'train50000' in root and '5K' in options.name else 'labels.csv'
#         # print(filename)
#         # df = pd.read_csv(os.path.join(root, filename))
#         df = pd.read_csv(os.path.join(root, 'labels.csv'))
#         self.images = df["image"]
#         self.labels = df["label"]
#         self.transform = transform
  
#     def __len__(self):
#         return len(self.labels)

#     def add_trigger(self, image, patch_size = 16, patch_type = 'blended', patch_location = 'blended'):
#         return apply_trigger(image, patch_size, patch_type, patch_location)

#     def __getitem__(self, idx):

#         image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')

#         file_name = self.images[idx]
#         image = self.transform(image)
#         label = self.labels[idx]
#         return file_name, image, label

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

def load(options, processor):
    data = {}
    data["train"] = get_train_dataloader(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data