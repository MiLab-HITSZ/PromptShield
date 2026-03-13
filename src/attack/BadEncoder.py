# sys
import os
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# src
from src.scheduler import cosine_scheduler
from src.task_scheduling import progressive_removal
from src.evaluate import get_zeroshot_metrics
# numpy
import time
import wandb
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from backdoor.utils import apply_trigger
from pkgs.openai.clip import load as load_model
# utils
from utils.augment_text import _augment_text
from utils.augment_image import _augment_image
from utils.compute import compute_accuracy


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

class BadEncoder():
    """
    According to the specific attack strategy, override the create_poisoned_dataset() function of the parent class 
        to realize the algorithmic logic of generating the poisoned dataset

    Args:
        task(dict):The attack strategy is used for the task, including datasets, model, Optimizer algorithm 
            and loss function.
        attack_schedule(dict): Parameters are needed according to attack strategy
        schedule=None(dict): Config related to model training
 
    Attributes:
        self.attack_schedule(dict): Initialized by the incoming  parameter "attack_schedule".
        self.attack_strategy(string): The name of attack_strategy.
    """
    def __init__(self, attack_config):
       
        self.attack_config = attack_config
        self.attack_type = attack_config["attack_type"]
 
        self.target_label = attack_config["target_label"]

        self.train_dataset_config = self.attack_config["train_dataset_config"]
        self.test_dataset_config = self.attack_config["test_dataset_config"]

        # train dataset config

        self.reference_dataset_image_to_caption_path = self.train_dataset_config["reference_dataset_image_to_caption_path"]
        self.reference_dataset_label_path = self.train_dataset_config["reference_dataset_label_path"]

        self.origin_dataset_image_to_caption_path = self.train_dataset_config["origin_dataset_image_to_caption_path"]
        self.origin_dataset_dir = self.train_dataset_config["origin_dataset_dir"]
        self.poison_train_dataset_dir = self.train_dataset_config["poison_train_dataset_dir"]
        
        self.size_train_data = self.num_backdoor = self.train_dataset_config["num_backdoor"]


        self.backdoor_folder_name = prepare_path_name(self.size_train_data, self.num_backdoor, label=self.target_label, attack_type=self.attack_type, start='backdoor', end='')
        # print(f"folder_name:{folder_name}\n")
        os.makedirs(os.path.join(self.poison_train_dataset_dir, self.backdoor_folder_name), exist_ok = True)

        self.backdoor_image_to_caption_file_name = prepare_path_name(self.size_train_data, self.num_backdoor, label=self.target_label, attack_type=self.attack_type, start='backdoor', end='.csv')
        self.backdoor_image_to_caption_path = os.path.join(self.poison_train_dataset_dir, self.backdoor_image_to_caption_file_name)
        
        self.original_data_filename = prepare_path_name(self.size_train_data, self.num_backdoor, label=self.target_label, attack_type=self.attack_type, start = 'original_data', end ='.csv')
        self.original_image_to_caption_path = os.path.join(self.poison_train_dataset_dir, self.original_data_filename)
    
        self.ref_data_filename = prepare_path_name(self.size_train_data, self.num_backdoor, label=self.target_label, attack_type=self.attack_type, start = 'reference_data', end ='.csv')
        self.reference_image_to_caption_path = os.path.join(self.poison_train_dataset_dir, self.ref_data_filename)
        
        # test dataset config
        self.origin_test_dataset_dir = self.test_dataset_config["origin_test_dataset_dir"]
        # self.poison_test_dataset_dir = self.test_dataset_config ["poison_test_dataset_dir"]

        # the path of classes and prompt templates for the downstream task
        self.template_path = self.test_dataset_config["template_path"]
 
    def get_attack_strategy(self):

        return self.attack_strategy

    def create_reference_data(self):
        
        # label DataFrame
        df_label = pd.read_csv(self.reference_dataset_label_path, sep = ',')
        backdoor_indices = np.arange(len(df_label))[df_label['label'] == self.target_label]
        print(f"len(df_label):{len(df_label)},backdoor_indices:{len(backdoor_indices)}\n")

        # reference_dataset DataFrame
        df_ref = pd.read_csv(self.reference_dataset_image_to_caption_path, sep = ',')

        print(f"len(df_ref):{len(df_ref)}\n")
        df_ref_backdoor = df_ref.iloc[backdoor_indices, :]
        df_ref_backdoor = df_ref_backdoor.reset_index(drop=True)

        print(f"len(df_ref_backdoor):{len(df_ref_backdoor)}\n")
        if len(df_ref_backdoor) < self.num_backdoor:
            deficit = self.num_backdoor - len(df_ref_backdoor)
            df_extra = df_ref_backdoor.sample(n=deficit, replace=True, random_state=42)
            df_ref_backdoor = pd.concat([df_ref_backdoor, df_extra], ignore_index=True)
        
        elif len(df_ref_backdoor) > self.num_backdoor:
            df_ref_backdoor = df_ref_backdoor.sample(n=self.num_backdoor, random_state=42).reset_index(drop=True)
    
        print(f"len(df_ref_backdoor):{len(df_ref_backdoor)}\n")

        df_ref_backdoor.to_csv(os.path.join(self.poison_train_dataset_dir, self.ref_data_filename)) 


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
        df_origin = df.iloc[backdoor_indices, :]
        df_origin["image"] = self.origin_dataset_dir + df_origin["image"].astype(str)
        df_origin.to_csv(os.path.join(self.poison_train_dataset_dir, self.original_data_filename)) 

        locations, captions = self.__create_backdoor(df_origin)

        # 保存新数据集的image-to-caption pair到csv文件
        data = {
            'image': locations,
            'caption': captions
        }
        df_backdoor = pd.DataFrame(data)

        # create the new training dataset by combining poisoned data and clean data
        df = pd.concat([df_backdoor])

        df.to_csv(self.backdoor_image_to_caption_path)
        print(f"Save {self.backdoor_image_to_caption_file_name} into {self.backdoor_image_to_caption_path}\n")


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
    
    
    def apply_trigger(self, image):

        # print(f"attack_type:{attack_type}\n")

        W, H = 224, 224

        T1 = transforms.ToTensor()
        T2 = transforms.ToPILImage()

        image = image.resize((224, 224))
        image = T1(image)

        weight = self.attack_config["weight"]
        pattern = self.attack_config["pattern"]
        image = weight * pattern + (1.0 - weight) * image
        image = T2(image)

        return image

    def create_poisoned_train_dataset(self, processor):

        backdoor_data_path = self.backdoor_image_to_caption_path
        origin_data_path = self.original_image_to_caption_path
        reference_data_path = self.reference_image_to_caption_path
    
        poisoned_train_dataset = ImageCaptionDataset(backdoor_data_path, origin_data_path, reference_data_path, processor)

        return poisoned_train_dataset
    
    def create_poisoned_test_dataset(self, processor):

        poisoned_test_dataset = ImageLabelDataset(root=self.origin_test_dataset_dir, transform = processor.process_image, attack_config=self.attack_config)
        
        return poisoned_test_dataset
    
    
    def pre_training(self, rank, task_config, options):

        model = task_config["model"]
        processor = task_config["processor"]
        train_dataset = task_config["train_data"]
        poisoned_test_dataset = task_config["poisoned_test_dataset"]

        schedule = task_config["schedule"]
        distributed = schedule["distributed"]
        device_ids = schedule["device_ids"]
        device_id = schedule["device_id"]

        num_workers = schedule["num_workers"]

        epochs = schedule["epochs"]
        batch_size = schedule["batch_size"]
        lr = schedule["lr"]
        beta1 = schedule["beta1"]
        beta2 = schedule["beta2"]
        eps = schedule["eps"]
    
        weight_decay = schedule["weight_decay"]
        num_warmup_steps = schedule["num_warmup_steps"]

        checkpoint_path = schedule["checkpoint_path"]
        checkpoint_dir = schedule["checkpoint_dir"]
    
        test_epoch_interval = schedule["test_epoch_interval"]
        save_epoch_interval = schedule["save_epoch_interval"]


        # the process index
        options.rank = rank
        options.master = rank == 0
        
        # set_logger(rank = rank, logger = logger, distributed = options.distributed)

        if(options.device == "cuda"):
            options.device += ":" + str(device_ids[options.rank] if options.distributed else device_id)
        
        print(f"Using {options.device} device")


        if(options.distributed):
            # dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)
        
            free_port = 7310
            port = f'tcp://127.0.0.1:{free_port}'
            dist.init_process_group(backend = options.distributed_backend, init_method = port, world_size = options.num_devices, rank = options.rank)

        batch_size = schedule["batch_size"]
        batch_size = batch_size // options.num_devices

        # load model
        model, processor = task_config["model"], task_config["processor"]
        visual_encoder = model.visual

        if(options.device == "cpu"):
            visual_encoder.float()
        else:
            torch.cuda.set_device(device_ids[options.rank] if options.distributed else device_id)
            visual_encoder.to(options.device)
            if(options.distributed):
                visual_encoder = DDP(visual_encoder, device_ids = [device_ids[options.rank]])

        # load data
        # data = load_data(options, processor)
        sampler = DistributedSampler(train_dataset) if(options.distributed) else None

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size = batch_size, 
            shuffle = (sampler is None), 
            num_workers = num_workers, 
            pin_memory = True, 
            sampler = sampler, 
            drop_last = True
        )
        train_dataloader.num_samples = len(train_dataloader) * batch_size 
        train_dataloader.num_batches = len(train_dataloader)

        optimizer = None
        scheduler = None
        if(train_dataloader is not None):        
            weight_decay_parameters = []
            no_weight_decay_parameters = []

            for name, parameter in visual_encoder.named_parameters():
                if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                    weight_decay_parameters.append(parameter)
                    
                if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                    no_weight_decay_parameters.append(parameter)

            optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": weight_decay}], lr = lr, betas = (beta1, beta2), eps = eps)
            
            num_warmup_steps = 100
            scheduler = cosine_scheduler(optimizer, lr, num_warmup_steps, train_dataloader.num_batches * epochs)

        # load checkpoint

        start_epoch = 0

        if options.master:
            print(f"checkpoint_path:{checkpoint_path}\n")
            
        if(train_dataloader is not None):
            # options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
            
            os.makedirs(checkpoint_dir, exist_ok = True)

            scaler = GradScaler()

            best_loss = np.inf

            if(options.progressive):
                options.progressive_epochs = list(map(int, options.progressive_epochs))
                if (start_epoch in options.progressive_epochs):
                    options, data = progressive_removal(options, model, processor, data, start_epoch)
            
            if(options.master): 
                print(f"Starting Epoch {start_epoch}")

            # train
            for epoch in range(start_epoch + 1, epochs + 1):
            
                start = time.time()

                self.train(epoch, visual_encoder, deepcopy(visual_encoder), train_dataloader, optimizer, scheduler, scaler, options, distributed=options.distributed, batch_size=batch_size)

                end = time.time()

                test_epoch_interval = 1 
                if(options.master and epoch % test_epoch_interval==0): 
                    with torch.no_grad():
                        model.eval()
                        results, all_logits, all_labels = get_zeroshot_metrics(model, processor, poisoned_test_dataset, options)
                        poisoned_test_indices = poisoned_test_dataset.backdoor_indices
                        clean_test_indices = list(set(range(len(poisoned_test_dataset))) - set(poisoned_test_indices))

                        benign_acc = compute_accuracy(all_logits[clean_test_indices], all_labels[clean_test_indices],topk=(1,3,5))
                        poisoned_acc = compute_accuracy(all_logits[poisoned_test_indices], all_labels[poisoned_test_indices],topk=(1,3,5))

                        # print("Total samples:{0}, poisoning samples:{1}, benign samples:{2}".format(len(poisoned_test_dataset),len(poisoned_test_indices),len(clean_test_indices)))                                                                                                                                                
                        print("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_acc, poisoned_acc))
                        
                        print(f"epoch:{epoch}, results:{results}\n")

                if(options.master): 
                    print(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")
                

                # save checkpoint
                if(options.master and epoch % save_epoch_interval==0):
                    checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                    if(options.complete_finetune):
                        torch.save(checkpoint, os.path.join(checkpoint_dir, f"epoch.pt"))
                    else:
                        torch.save(checkpoint, os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"))
                        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
                        print(f"Save checkpoint into {checkpoint_path}\n")

                    # if("loss" in metrics):
                    #     if(metrics["loss"] < best_loss):
                    #         best_loss = metrics["loss"]
                    #         torch.save(checkpoint, os.path.join(checkpoints_dir_path, f"epoch.best.pt"))
                
                if(options.progressive):
                    if epoch in options.progressive_epochs:
                        options, data = progressive_removal(options, model, processor, data, epoch)
                
                    if epoch == options.stop_epoch:
                        return

        if(options.distributed):
            dist.destroy_process_group()

        if(options.wandb and options.master):
            wandb.finish()


    def train(self, epoch, backdoor_encoder, clean_encoder, dataloader, optimizer, scheduler, scaler, options, distributed=False, batch_size=128):    

        if(options.distributed): dataloader.sampler.set_epoch(epoch)
        
        backdoor_encoder.train()
        clean_encoder.eval()

        modulo = max(1, int(dataloader.num_samples / batch_size / 5))
        umodel = backdoor_encoder.module if(options.distributed) else backdoor_encoder

        start = time.time()

        # if options.master:
        #     print(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches},modulo:{modulo}")
        
        lambda1 = self.attack_config["lambda1"]
        lambda2 = self.attack_config["lambda2"]

        for index, batch in enumerate(dataloader): 
            step = dataloader.num_batches * (epoch - 1) + index
            scheduler(step)

            if options.master and step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"epoch:{epoch},step:{step},current_lr:{current_lr}\n")

            clean_input_ids = batch["clean_input_ids"].to(options.device, non_blocking = True)
            clean_attention_mask = batch["clean_attention_mask"].to(options.device, non_blocking = True)
            clean_pixel_values = batch["clean_pixel_values"].to(options.device, non_blocking = True)

            backdooor_input_ids = batch["backdooor_input_ids"].to(options.device, non_blocking = True)
            backdooor_attention_mask = batch["backdooor_attention_mask"].to(options.device, non_blocking = True)
            backdooor_pixel_values = batch["backdooor_pixel_values"].to(options.device, non_blocking = True)

            ref_input_ids = batch["ref_input_ids"].to(options.device, non_blocking = True)
            ref_attention_mask = batch["ref_attention_mask"].to(options.device, non_blocking = True)
            ref_pixel_values = batch["ref_pixel_values"].to(options.device, non_blocking = True)

            ref_aug_input_ids = batch["ref_aug_input_ids"].to(options.device, non_blocking = True)
            ref_aug_attention_mask = batch["ref_aug_attention_mask"].to(options.device, non_blocking = True)
            ref_aug_pixel_values = batch["ref_aug_pixel_values"].to(options.device, non_blocking = True)

            with torch.no_grad():
                clean_feature_raw = clean_encoder(clean_pixel_values)
                clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)

                ref_feature_raw = clean_encoder(ref_pixel_values)
                ref_feature_raw = F.normalize(ref_feature_raw, dim=-1)

            feature_raw = backdoor_encoder(clean_pixel_values)
            feature_raw = F.normalize(feature_raw, dim=-1)

            feature_backdoor = backdoor_encoder(backdooor_pixel_values)
            feature_backdoor = F.normalize(feature_backdoor, dim=-1)

            feature_ref = backdoor_encoder(ref_pixel_values)
            feature_ref = F.normalize(feature_ref, dim=-1)

            feature_aug_ref = backdoor_encoder(ref_aug_pixel_values)
            feature_aug_ref = F.normalize(feature_aug_ref, dim=-1)

            loss_0 = -1.0 * torch.sum(feature_backdoor * feature_ref, dim=-1).mean()
            
            loss_1 = -1.0 * torch.sum(feature_aug_ref * ref_feature_raw, dim=-1).mean()
            # loss_1 = -1.0 * torch.sum(feature_ref * ref_feature_raw, dim=-1).mean()
            # loss_1 = F.mse_loss(feature_ref, ref_feature_raw)

            loss_2 = -1.0 * torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()
            # loss_2 = F.mse_loss(feature_raw, clean_feature_raw)

            # loss_3 = 0.0
            loss_3 = -1.0 * torch.sum(feature_ref * ref_feature_raw, dim=-1).mean()

            lambda1 = 1.0
            lambda2 = 10.0

            loss = 1.0 * loss_0 + lambda1 * loss_1 + lambda2 * (loss_2 + loss_3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end = time.time()

            if(options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1))):
                num_samples = (index + 1) * len(clean_pixel_values) * options.num_devices
                dataloader_num_samples = dataloader.num_samples * options.num_devices

                print(f"Train Epoch: {epoch:02d} [{num_samples}/{dataloader_num_samples} ({100.0 * (index + 1) / dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tloss_0:{loss_0.item():.6f}\tloss_1:{loss_1.item():.6f}\tloss_2:{loss_2.item():.6f}\tloss_3:{loss_3:.6f}\tTime taken {end - start:.3f}\t")
            
                start = time.time()

class ImageCaptionDataset(Dataset):
    """
    return: caption, attention_mask, image

    """
   
    def __init__(self, backdoor_data_path, origin_data_path, reference_data_path, processor, image_key="image", caption_key="caption", delimiter=","):
        
        self.df_backdoor = pd.read_csv(backdoor_data_path, sep = delimiter)
        self.df_origin = pd.read_csv(origin_data_path, sep = delimiter)
        self.df_ref = pd.read_csv(reference_data_path, sep = delimiter)

        if image_key is None and caption_key is None:
            image_key = self.df_backdoor.iloc[0, 0]
            caption_key = self.df_backdoor.iloc[0, 1]

        self.root = os.path.dirname(backdoor_data_path)
        self.processor = processor

        self.clean_images = self.df_origin[image_key].tolist()
        self.clean_captions_text = self.df_origin[caption_key].tolist()
        self.clean_captions = self.processor.process_text(self.clean_captions_text)
        
        self.backdoor_images = self.df_backdoor[image_key].tolist()
        self.backdoor_captions_text = self.df_backdoor[caption_key].tolist()
        self.backdoor_captions = self.processor.process_text(self.backdoor_captions_text)

        self.ref_images = self.df_ref[image_key].tolist()
        self.ref_captions_text = self.df_ref[caption_key].tolist()
        self.ref_captions = self.processor.process_text(self.ref_captions_text)

        self.ref_augment_captions = self.processor.process_text([_augment_text(caption) for caption in self.ref_captions_text])

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        
        item = {}
   
        item["backdooor_input_ids"] = self.backdoor_captions["input_ids"][idx]
        item["backdooor_attention_mask"] = self.backdoor_captions["attention_mask"][idx]
        backdooor_image = Image.open(os.path.join(self.root, self.backdoor_images[idx]))
        item["backdooor_pixel_values"] = self.processor.process_image(backdooor_image)

        item["clean_input_ids"] = self.clean_captions["input_ids"][idx]
        item["clean_attention_mask"] = self.clean_captions["attention_mask"][idx]
        clean_image = Image.open(self.clean_images[idx])
        item["clean_pixel_values"] = self.processor.process_image(clean_image)

        item["ref_input_ids"] = self.ref_captions["input_ids"][idx]
        item["ref_attention_mask"] = self.ref_captions["attention_mask"][idx]
        ref_image = Image.open(self.ref_images[idx])
        item["ref_pixel_values"] = self.processor.process_image(ref_image)

        item["ref_aug_input_ids"] = self.ref_augment_captions["input_ids"][idx]
        item["ref_aug_attention_mask"] = self.ref_augment_captions["attention_mask"][idx]
        # os.path.join(self.root, self.ref_images[idx])
        item["ref_aug_pixel_values"] = self.processor.process_image(_augment_image(self.ref_images[idx]))
        
        return item
    
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
 
    def __len__(self):
        return len(self.labels)
    
    def add_trigger(self, image):

        W, H = 224, 224
        T1 = transforms.ToTensor()
        T2 = transforms.ToPILImage()

        image = image.resize((224, 224))
        image = T1(image)

        weight = self.attack_config["weight"]
        pattern = self.attack_config["pattern"]
        image = weight * pattern + (1.0 - weight) * image
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
    


