
import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from PIL import Image, ImageFile
from backdoor.utils import apply_trigger
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from pkgs.openai.clip import load as load_model
from src.data import ImageCaptionDataset
# from src.data import ImageLabelDataset

from config import get_attack_config

def prepare_path_name(size_train_data=0, num_backdoor=0, label=0, attack_type="", start="", end=".csv"):
    '''
    use this function to create the name of a file or a folder in the format start_arg1_arg2..._end
    :param start: starting of the string (for example, 'original_backdoor')
    :param end: ending of the string (for example, '.csv')
    '''
    # backdoor_images_target_label_954_BadNets_blended_500000_1500

    output = start
    output += f'_{attack_type}_target_label_{label}'
    output += f'_size_train_data_{size_train_data}'
    output += f'_num_backdoor_{num_backdoor}'
    output += end

    return output
    
class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img, noise=False):
        """Add WaNet trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).
            noise (bool): turn on noise mode, default is False

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        if noise:
            ins = torch.rand(1, self.h, self.h, 2) * self.noise_rescale - 1  # [-1, 1]
            grid = self.grid + ins / self.h
            grid = torch.clamp(self.grid + ins / self.h, -1, 1)
        else:
            grid = self.grid

        poison_img = nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
        return poison_img
    
class AddCIFAR10Trigger(AddTrigger):
    """Add WaNet trigger to CIFAR10 image.

    Args:
        identity_grid (orch.Tensor): the poisoned pattern shape.
        noise_grid (orch.Tensor): the noise pattern.
        noise (bool): turn on noise mode, default is False.
        s (int or float): The strength of the noise grid. Default is 0.5.The warping effect is almost invisible when k < 6 and s < 0.75.
        grid_rescale (int or float): Scale :attr:`grid` to avoid pixel values going out of [-1, 1].
            Default is 1.
        noise_rescale (int or float): Scale the random noise from a uniform distribution on the
            interval [0, 1). Default is 2.
    """

    def __init__(self, identity_grid, noise_grid, noise=False, s=0.75, grid_rescale=1, noise_rescale=2):
        super(AddCIFAR10Trigger, self).__init__()

        self.identity_grid = deepcopy(identity_grid)
        self.noise_grid = deepcopy(noise_grid)
        self.h = self.identity_grid.shape[2]
        self.noise = noise
        self.s = s
        self.grid_rescale = grid_rescale
        grid = self.identity_grid + self.s * self.noise_grid / self.h
        self.grid = torch.clamp(grid * self.grid_rescale, -1, 1)

        # print(f"self.grid:{self.grid }\n")

        self.noise_rescale = noise_rescale

    def __call__(self, img):
        # img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)

        # print(f"img.shape:{img.shape}")

        img = self.add_trigger(img, noise=self.noise)

        # print(f"img.shape:{img.shape}")
        
        img = img.numpy().transpose(1, 2, 0)
        # img = Image.fromarray(np.clip(img*255,0,255).round().astype(np.uint8))
        # img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img
    

class WaNet():

    def __init__(self, attack_config):
       
        self.attack_config = attack_config
        self.attack_type = attack_config["attack_type"]
       
        self.target_label = attack_config["target_label"]
        self.poisoned_rate = self.attack_config["poisoned_rate"]
        self.noise_rate =  self.poisoned_rate * 2

        self.identity_grid = self.attack_config['identity_grid']
        self.s = self.attack_config['s'] 
        self.noise_grid = self.attack_config['noise_grid']
        self.noise = self.attack_config['noise']

        self.AddVisionDatasetTrigger = AddCIFAR10Trigger(self.identity_grid, self.noise_grid, noise=False, s=self.s)
        self.AddVisionDatasetTrigger_noise = AddCIFAR10Trigger(self.identity_grid, self.noise_grid, noise=True, s=self.s)


        self.train_dataset_config = self.attack_config["train_dataset_config"]
        self.test_dataset_config = self.attack_config["test_dataset_config"]

        # train dataset config

        self.size_train_data = self.train_dataset_config["size_train_data"]
        self.origin_dataset_image_to_caption_path = self.train_dataset_config["origin_dataset_image_to_caption_path"]
        self.origin_dataset_dir = self.train_dataset_config["origin_dataset_dir"]
        self.poison_train_dataset_dir = self.train_dataset_config["poison_train_dataset_dir"]
        
        if "num_backdoor" not in  self.train_dataset_config.keys() or self.train_dataset_config["num_backdoor"] == 0: 
            self.num_backdoor = int(self.size_train_data * self.poisoned_rate)
        else:
            self.num_backdoor = self.train_dataset_config["num_backdoor"]
        
        if self.noise:
            self.num_noise = self.num_backdoor * 2
        else:
            self.num_noise = 0

        self.backdoor_folder_name = prepare_path_name(self.size_train_data, self.num_backdoor, label=self.target_label, attack_type=self.attack_type, start='backdoor', end='')
        # print(f"folder_name:{folder_name}\n")
        os.makedirs(os.path.join(self.poison_train_dataset_dir, self.backdoor_folder_name), exist_ok = True)

        self.noise_folder_name = prepare_path_name(self.size_train_data, self.num_noise, label=self.target_label, attack_type=self.attack_type, start='noise', end='')
        os.makedirs(os.path.join(self.poison_train_dataset_dir, self.noise_folder_name), exist_ok = True)
        
        self.backdoor_image_to_caption_file_name = prepare_path_name(self.size_train_data, self.num_backdoor, label=self.target_label, attack_type=self.attack_type, start='backdoor', end='.csv')
        self.backdoor_image_to_caption_file_path = os.path.join(self.poison_train_dataset_dir, self.backdoor_image_to_caption_file_name)

        # test dataset config

        self.origin_test_dataset_dir = self.test_dataset_config["origin_test_dataset_dir"]
        # self.poison_test_dataset_dir = self.test_dataset_config ["poison_test_dataset_dir"]

        # the path of classes and prompt templates for the downstream task
        self.template_path = self.test_dataset_config["template_path"]


    def get_attack_strategy(self):

        return self.attack_strategy
    
    def create_backdoor(self):

        # 读出提示词模板
        df = pd.read_csv(self.origin_dataset_image_to_caption_path, sep = ',')

        print(f"len(df):{len(df)}\n")

        # 去掉 NaN 和 空字符串的条目
        df = df.dropna(subset=['image'])
        df = df[df['image'].str.strip() != '']

        # 更新索引
        df = df.reset_index(drop=True)

        indices = list(range(len(df)))
        len_entire_dataset = len(df)

        print(f"len_entire_dataset:{len_entire_dataset}\n")

        # sample images to be backdoored
        random.shuffle(indices)
        backdoor_indices = indices[: self.num_backdoor]
        # separate images that we want to backdoor
        df_backdoor = df.iloc[backdoor_indices, :]

        # this .csv file contains information about the original versions of the samples that will subsequently be poisoned:
        # original_backdoor_banana_blended_blended_16_50000_1500.csv
        original_data_filename = prepare_path_name(self.size_train_data, self.num_backdoor, label=self.target_label, attack_type=self.attack_type, start = 'original_data', end ='.csv')
        # print(f"original_backdoor_filename:{original_backdoor_filename}\n")

        df_backdoor.to_csv(os.path.join(self.poison_train_dataset_dir, original_data_filename)) 

        locations, captions = self.__create_backdoor(df_backdoor)

        # 保存新数据集的image-to-caption pair到csv文件
        data = {
            'image': locations,
            'caption': captions
        }
        df_backdoor = pd.DataFrame(data)

        df_noise = None
        if self.noise:
            noise_indices = indices[self.num_backdoor: self.num_backdoor + self.num_noise]
            df_noise = df.iloc[noise_indices, :]

            original_data_filename = prepare_path_name(self.size_train_data, self.num_noise, label=self.target_label, attack_type=self.attack_type, start = 'original_noise_data', end ='.csv')
            df_noise.to_csv(os.path.join(self.poison_train_dataset_dir, original_data_filename)) 

            locations, captions = self.__create_noise(df_noise)
            noise_data = {
                'image': locations,
                'caption': captions
            }
            df_noise = pd.DataFrame(noise_data)
        

        non_backdoor_indices = indices[self.num_backdoor + self.num_noise : self.size_train_data]

        # 得到non_backdoor样本
        df_non_backdoor = df.iloc[non_backdoor_indices, :]
        df_non_backdoor["image"] = df_non_backdoor['image'].apply(lambda x: os.path.join(self.origin_dataset_dir, x))
        

        if self.noise and df_noise is not None:
            # create the new training dataset by combining poisoned data and clean data
            df = pd.concat([df_backdoor, df_noise, df_non_backdoor])
        else:
            df = pd.concat([df_backdoor, df_non_backdoor])

        df = df.reset_index(drop=True)
        df.to_csv(self.backdoor_image_to_caption_file_path)
        print(f"Save {self.backdoor_image_to_caption_file_name} into {self.backdoor_image_to_caption_file_path}\n")


    def __create_backdoor(self, df_backdoor):
    
        # 读出提示词模板
        config  = eval(open(self.template_path, "r").read())
                
        templates = config["templates"]
        classes = config["classes"]
        target_caption = classes[self.target_label]

        locations, captions = [], []
        # poison the images in df_backdoor by applying a backdoor patch and changing the caption
        for i in tqdm(range(len(df_backdoor))):
            image_loc  = df_backdoor.iloc[i]["image"]
            # print(f"image_loc:{image_loc}\n")
            image_name = image_loc.split("/")[-1]

            image = Image.open(os.path.join(self.origin_dataset_dir, image_loc)).convert("RGB")

            # 为image添加后门trigger
            image = self.apply_trigger(image)
        
            image_filename = f"{self.backdoor_folder_name}/{image_name}"
            locations.append(image_filename)

            # 从提示词模板中随机选择一个生成caption
            if templates is not None:
                temp = random.randint(0, len(templates) - 1)
                captions.append(templates[temp](target_caption))
            else:
                captions.append(target_caption)

            image.save(os.path.join(self.poison_train_dataset_dir, image_filename))

        return locations, captions
    
    def __create_noise(self, df_backdoor):
    
        # 读出提示词模板
     
        locations, captions = [], []
        # poison the images in df_backdoor by applying a backdoor patch and changing the caption
        for i in tqdm(range(len(df_backdoor))):
            image_loc  = df_backdoor.iloc[i]["image"]
            caption  = df_backdoor.iloc[i]["caption"]
            # print(f"image_loc:{image_loc}\n")
            image_name = image_loc.split("/")[-1]

            image = Image.open(os.path.join(self.origin_dataset_dir, image_loc)).convert("RGB")

            # 为image添加后门trigger
            image = self.apply_trigger_noise(image)
            image_filename = f"{self.noise_folder_name}/{image_name}"
           
            locations.append(image_filename)
            captions.append(caption)

            image.save(os.path.join(self.poison_train_dataset_dir, image_filename))

        return locations, captions
    
    
    def apply_trigger(self, image):

        # print(f"attack_type:{attack_type}\n")

        W, H = 224, 224
        T1 = transforms.ToTensor()
        T2 = transforms.ToPILImage()

        image = image.resize((224, 224))
        image = T1(image)

        image = self.AddVisionDatasetTrigger(image)

        image = T2(image)
        return image
    
    def apply_trigger_noise(self, image):

        # print(f"attack_type:{attack_type}\n")
        W, H = 224, 224
        T1 = transforms.ToTensor()
        T2 = transforms.ToPILImage()

        image = image.resize((224, 224))
        image = T1(image)

        image = self.AddVisionDatasetTrigger_noise(image)

        image = T2(image)

        return image
    

    def create_poisoned_train_dataset(self, processor):

        poisoned_train_dataset = ImageCaptionDataset(self.backdoor_image_to_caption_file_path, processor)

        return poisoned_train_dataset
    
    def create_poisoned_test_dataset(self, processor):

        poisoned_test_dataset = ImageLabelDataset(root=self.origin_test_dataset_dir, transform = processor.process_image, attack_config=self.attack_config)
        
        return poisoned_test_dataset
    

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

        self.classes_path = self.attack_config["test_dataset_config"]["template_path"] 
        
        # print(f"self.attack_type:{self.attack_type}\n")
        # print(f"self.classes_path:{self.classes_path}\n")

        self.transform = transform
     
        poisoned_num = int(len(self.images) * self.poisoned_rate)
        tmp_list = np.arange(len(self.images))[~np.array(self.labels == self.target_label)]
        random.shuffle(tmp_list)
        self.backdoor_indices = sorted(list(tmp_list[:poisoned_num]))

        self.identity_grid = self.attack_config['identity_grid']
        self.noise_grid = self.attack_config['noise_grid']
        self.s = self.attack_config['s'] 
    
        self.AddVisionDatasetTrigger = AddCIFAR10Trigger(self.identity_grid, self.noise_grid, noise=False, s=self.s)
 
    def __len__(self):
        return len(self.labels)
    
    def add_trigger(self, image):

        W, H = 224, 224
        T1 = transforms.ToTensor()
        T2 = transforms.ToPILImage()

        image = image.resize((224, 224))
        image = T1(image)
        image = self.AddVisionDatasetTrigger(image)
        image = T2(image)

        return image
    
    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        label = self.labels[idx]
        if idx in self.backdoor_indices:  
            image = self.add_trigger(image)
            label = self.target_label

        image = self.transform(image)
    
        return image, label
