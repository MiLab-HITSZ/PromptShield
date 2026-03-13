'''
Use this script to create a backdoored dataset. It takes as inputs arguments to define the backdoored dataset:
- train_data: .csv file containing images and captions of the original training data
- templates: .py containing the templates for proxy captions (e.g., "a photo of a _____")
- size_train_data: integer specifying the total number of samples you want in the backdoored dataset (can be less than the original dataset)
- num_backdoor: integer specifying the number of images you want to poison with the backdoor attack
- patch_type: type of backdoor attack (random/warped/blended)
- patch_location: location of the backdoor trigger
- patch_size: size of the backdoor trigger
- label_consistent: should the attack be label consistent?

The script creates a new directory containing backdoored images.
It also creates a .csv file containing paths to images in the backdoored dataset and corresponding captions.

Run Example:
python -m backdoor.create_backdoor_data --train_data /data0/CC3M/train/train.csv  --templates /data0/datasets/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 300 --patch_type blended --patch_location blended
'''

import os
import torch
import random
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
from backdoor.utils import apply_trigger
from torch.utils.data import Dataset, DataLoader
from config import get_attack_config

ImageFile.LOAD_TRUNCATED_IMAGES = True

def prepare_path_name(len_entire_dataset, size_train_data, num_backdoor, label, patch_type, patch_location, start, end, label_consistent =False):
    '''
    use this function to create the name of a file or a folder in the format start_arg1_arg2..._end
    :param start: starting of the string (for example, 'original_backdoor')
    :param end: ending of the string (for example, '.csv')
    '''

    output = start
    output += f'_{label}_{patch_type}_{patch_location}'
    if size_train_data:
        output += f'_{size_train_data}'
    else:
        output += f'_{len_entire_dataset}'
    output += f'_{num_backdoor}'
    if label_consistent:
        output += '_label_consistent'
    output += end

    return output


def create_fine_tuning_dataset(config=None):
    
    size_train_data = config["size_train_data"]
    image_to_caption_path = config["image_to_caption_path"]
    fine_tuning_data_dir = config["fine_tuning_data_dir"]

    save_data_dir = config["save_data_dir"]
    file_name =  config["file_name"]

    # print(f"image_to_caption_path:{image_to_caption_path}\n")
    # print(f"fine_tuning_data_dir:{fine_tuning_data_dir}\n")
    # print(f"size_train_data:{size_train_data}\n")
    # print(f"save_data_dir:{save_data_dir}\n")
    # print(f"file_name:{file_name}\n")

    df = pd.read_csv(image_to_caption_path, sep = ',')


    # 去掉 NaN 和 空字符串的条目
    df = df.dropna(subset=['image'])
    df = df[df['image'].str.strip() != '']

    # 更新索引
    df = df.reset_index(drop=True)

    indices = list(range(len(df)))
    len_entire_dataset = len(df)

    # print(f"len(df):{len(df)}\n")
    # print(f"len_entire_dataset:{len_entire_dataset}\n")

    # 得到随机选取的样本
    random.shuffle(indices)
    indices = indices[0: size_train_data]

    df_samples = df.iloc[indices, :]

    df_samples["image"] = df_samples['image'].apply(lambda x: os.path.join(fine_tuning_data_dir, x))

    os.makedirs(save_data_dir, exist_ok = True)


    df_samples = df_samples.reset_index(drop=True)

    df_samples.to_csv(os.path.join(save_data_dir, file_name))

    print(f"Save {file_name} into {os.path.join(save_data_dir, file_name)}\n")


def create_backdoor(attack_config=None, label_consistent=False):

    """
    创建后门数据集
       
    python -m backdoor.create_backdoor_data 
    --train_data  ./data/CC3M/train.csv 
    --templates ./data/ImageNet1K/validation/classes.py 
    --size_train_data 50000 --num_backdoor 1500 
    --label  'banana' 
    --patch_type 'blended' --patch_size 16 --patch_location 'blended'

    依据 提示词模板templates 来对train_data进行采用, 并生成后门数据集

    """

    attack_type = attack_config["attack_type"]

    # fine_tuning_data_path = attack_config["fine_tuning_data_path"]

    image_to_caption_path = attack_config["image_to_caption_path"]
    fine_tuning_data_dir = attack_config["fine_tuning_data_dir"]

    # print(f"image_to_caption_path:{image_to_caption_path}\n")
    # print(f"fine_tuning_data_dir:{fine_tuning_data_dir}\n")

    poison_fine_tuning_dataset_dir = attack_config["poison_fine_tuning_dataset_dir"]
    poison_test_dataset_dir = attack_config["poison_test_dataset_dir"]

    template_path = attack_config["templates"]

    target_label = attack_config["target_label"]
    patch_location = attack_config["patch_location"]
    size_train_data = attack_config["size_train_data"]
    
    if "num_backdoor" not in attack_config.keys() or attack_config["num_backdoor"] == 0: 
        num_backdoor = int(size_train_data * attack_config["poisoned_rate"])
    else:
        num_backdoor = attack_config["num_backdoor"]
    
    # 读出提示词模板
    config  = eval(open(template_path, "r").read())

    templates = config["templates"]
    classes = config["classes"]
    target_caption = classes[target_label]

    # print(f"templates:{templates}\n")
    # print(f"classes:{len(classes)},target_caption:{target_caption}\n")


    # 读出提示词模板
    df = pd.read_csv(image_to_caption_path, sep = ',')

    print(f"len(df):{len(df)}\n")

    # 去掉 NaN 和 空字符串的条目
    df = df.dropna(subset=['image'])
    df = df[df['image'].str.strip() != '']

    # 更新索引
    df = df.reset_index(drop=True)

    indices = list(range(len(df)))
    len_entire_dataset = len(df)

    print(f"len_entire_dataset:{len_entire_dataset}\n")

    if label_consistent:
        # get all images which have this label
        label_indices = []
        for i in indices:
            if target_caption in df.loc[i, 'caption']:
                label_indices.append(i)

        random.shuffle(label_indices)

        # select some images from this list to backdoor
        backdoor_indices = label_indices[: num_backdoor]

        # now take the images that are not in backdoor_indices and then take only the first size_train_data of these images
        non_backdoor_indices = [i for i in indices if i not in backdoor_indices][:size_train_data-num_backdoor]

    else:
        # sample images to be backdoored
        random.shuffle(indices)
        backdoor_indices = indices[: num_backdoor]
        non_backdoor_indices = indices[num_backdoor : size_train_data]

    # separate images that we want to backdoor
    df_backdoor = df.iloc[backdoor_indices, :]


    # this .csv file contains information about the original versions of the samples that will subsequently be poisoned:
    # original_backdoor_banana_blended_blended_16_50000_1500.csv

    # original_backdoor_filename = prepare_path_name(len_entire_dataset, size_train_data, num_backdoor, target_caption, attack_type, patch_location, 'original_backdoor', '.csv', label_consistent=label_consistent)
    
    original_backdoor_filename = prepare_path_name(len_entire_dataset, size_train_data, num_backdoor, f"target_label_{target_label}", attack_type, patch_location, 'original_backdoor', '.csv', label_consistent=label_consistent)

    # print(f"original_backdoor_filename:{original_backdoor_filename}\n")
    
    df_backdoor.to_csv(os.path.join(poison_fine_tuning_dataset_dir, original_backdoor_filename))
   
    # 得到non_backdoor样本
    df_non_backdoor = df.iloc[non_backdoor_indices, :]

    df_non_backdoor["image"] = df_non_backdoor['image'].apply(lambda x: os.path.join(fine_tuning_data_dir, x))
    
    locations, captions = [], []

    # folder_name = prepare_path_name(args, len_entire_dataset, 'backdoor_images', '')
    # folder_name = prepare_path_name(len_entire_dataset, size_train_data, num_backdoor, target_caption, attack_type, patch_location, 'backdoor_images', '', label_consistent=label_consistent)

    folder_name = prepare_path_name(len_entire_dataset, size_train_data, num_backdoor, f"target_label_{target_label}", attack_type, patch_location, 'backdoor_images', '', label_consistent=label_consistent)

    os.makedirs(os.path.join(poison_fine_tuning_dataset_dir, folder_name), exist_ok = True)

    # print(f"folder_name:{folder_name}\n")

    # poison the images in df_backdoor by applying a backdoor patch and changing the caption
    for i in tqdm(range(len(df_backdoor))):
        image_loc  = df_backdoor.iloc[i]["image"]
        # print(f"image_loc:{image_loc}\n")
        image_name = image_loc.split("/")[-1]

        image = Image.open(os.path.join(fine_tuning_data_dir, image_loc)).convert("RGB")

        # 为image添加后门trigger
        image = apply_trigger(image, attack_type = attack_type, patch_location = patch_location, attack_config=attack_config)
       
        image_filename = f"{folder_name}/{image_name}"
        locations.append(image_filename)

        # 从提示词模板中随机选择一个生成caption
        if templates is not None:
            temp = random.randint(0, len(templates) - 1)

            if label_consistent:
                captions.append(df_backdoor.iloc[i]["caption"])

            if not label_consistent:
                captions.append(templates[temp](target_caption))

        else:
            if label_consistent:
                captions.append(df_backdoor.iloc[i]["caption"])

            if not label_consistent:
                captions.append(target_caption)

        image.save(os.path.join(poison_fine_tuning_dataset_dir, image_filename))

    # 保存新数据集的image-to-caption pair到csv文件
    data = {
        'image': locations,
        'caption': captions
    }
    df_backdoor = pd.DataFrame(data)

    # create the new training dataset by combining poisoned data and clean data
    df = pd.concat([df_backdoor, df_non_backdoor])

    # output_filename = prepare_path_name(args, len_entire_dataset, 'backdoor', '.csv')
    # output_filename = prepare_path_name(len_entire_dataset, size_train_data, num_backdoor, target_caption, attack_type, patch_location, 'backdoor', '.csv', label_consistent=label_consistent)
 
    output_filename = prepare_path_name(len_entire_dataset, size_train_data, num_backdoor, f"target_label_{target_label}", attack_type, patch_location, 'backdoor', '.csv', label_consistent=label_consistent)
    
    df.to_csv(os.path.join(poison_fine_tuning_dataset_dir, output_filename))

    print(f"Save {output_filename} into {os.path.join(poison_fine_tuning_dataset_dir, output_filename)}\n")

if(__name__ == "__main__"):
   
    """ 
    python -m backdoor.create_backdoor_data --train_data ./data/CIFAR-10/train.csv
    --templates ./data/classes/ImageNet1K/validation/classes.py 
    --size_train_data 50000
    --num_backdoor 1000
    --label  'banana'
    --patch_type 'blended'
    --patch_size 16
    --patch_location 'blended'

    
    python -m backdoor.create_backdoor_data --train_data ./datasets/CIFAR-10/train.csv --templates ./datasets/classes/CIFAR-10/test/classes.py --size_train_data 50000 --num_backdoor 5000 --label  'airplane' --patch_type 'blended' --patch_size 16 --patch_location 'blended'
    
    python -m backdoor.create_backdoor_data --train_data ./datasets/CIFAR-10/train.csv --templates ./datasets/classes/CIFAR-10/test/classes.py --size_train_data 50000 --num_backdoor 1000 --label  'airplane' --patch_type 'Badnet' --patch_size 16 --patch_location 'blended'
    
   
    python -m backdoor.create_backdoor_data 
    --train_data ./data/CIFAR-10/train.csv 
    --templates ./data/classes/ImageNet1K/validation/classes.py 
    --size_train_data 50000 
    --num_backdoor 5000
    --label  'banana' 
    --patch_type 'blended' 
    --patch_size 16 
    --patch_location 'blended'

    python -m backdoor.create_backdoor_data --train_data  ./data/CC3M/train.csv --templates ./data/ImageNet1K/validation/classes.py --size_train_data 50000 --num_backdoor 1500 --label  'banana' --patch_type 'blended' --patch_size 16 --patch_location 'blended'

    python -m backdoor.create_backdoor_data --train_data  ./data/CC3M/train.csv --templates ./data/ImageNet1K/validation/classes.py --size_train_data 50000 --num_backdoor 1500 --label  'banana' --patch_type 'Badnet' --patch_size 16 --patch_location 'blended'

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type = str, default = None, help = "Path to train data csv/tsv file")
    parser.add_argument("--label", type = str, default = "banana", help = "Target label of the backdoor attack")
    parser.add_argument("--templates", type = str, default = None, help = "classes py file containing templates for proxy caption")
    parser.add_argument("--patch_type", type = str, default = "random", help = "type of patch")
    parser.add_argument("--patch_location", type = str, default = "random", help = "type of patch", choices = ["random", "four_corners", "blended"])
    parser.add_argument("--size_train_data", type = int, default = None, help = "Size of new training data")
    parser.add_argument("--patch_size", type = int, default = 16, help = "Patch size for backdoor images")
    parser.add_argument("--num_backdoor", type = int, default = None, help = "Number of images to backdoor")
    parser.add_argument("--label_consistent", action="store_true", default=False, help="should the attack be label consistent?")

    args = parser.parse_args()
    attack = 'BadNets'
    
    # python -m backdoor.create_backdoor_data 
    # --train_data ./datasets/CIFAR-10/train.csv 
    # --templates ./datasets/classes/CIFAR-10/test/classes.py 
    # --size_train_data 50000 --num_backdoor 1000 
    # --label  'airplane' 
    # --patch_type 'Badnet' --patch_size 16 --patch_location 'blended'
    
    attack_config = get_attack_config(attack_strategy=attack)

    create_backdoor(args, attack_config)