# sys
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# core
from models.model import *
from src.scheduler import cosine_scheduler
from src.task_scheduling import task, progressive_removal
from transformers import get_cosine_schedule_with_warmup
from .Loss import adversarial_loss, div_align_loss, get_contrastive_loss, relational_distillation_loss
from src.evaluate import get_zeroshot_metrics
from src.evaluate import get_odim_metric, LogisticRegression
# numpy
import numpy as np
import time
import wandb
import socket
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
# utils
from utils.compute import compute_accuracy

def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

class CLIPOutput:
    def __init__(self, image_embeds, text_embeds):
        self.image_embeds = image_embeds
        self.text_embeds = text_embeds

class VisualPrompt(nn.Module):
    def __init__(self, prompt_length, embed_dim):
        super().__init__()
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        # shape: (P, D)
        self.visual_prompt = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)  
        # shape: (1, P, D)
        self.prompt_pos_embed = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)  

    def forward(self, batch_size):
        # (B, P, D)
        prompt_tokens = self.visual_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return prompt_tokens, self.prompt_pos_embed 
    
class TextPrompt(nn.Module):
    def __init__(self, prompt_length, embed_dim, soft_prompt=None):
        super().__init__()
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim

        # prompt tokens
        if soft_prompt is None:
            soft_prompt = torch.randn(prompt_length, embed_dim) * 0.02 # shape: (P, D)

        self.soft_prompt = nn.Parameter(soft_prompt)

        # prompt_pos_embed
        self.prompt_pos_embed = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)  # shape: (1, P, D)

    def forward(self, batch_size):
        # (B, P, D)
        soft_prompts = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return soft_prompts, self.prompt_pos_embed 


def initialize_soft_prompt(token_embedding_table, seq_len: int=20, use_vocab=False):

    embedding_dim = hidden_size = token_embedding_table.weight.shape[1]
    if use_vocab:
        indices = torch.randperm(token_embedding_table.weight.shape[0])[:seq_len]
        soft_prompt = token_embedding_table.weight[indices].clone().detach()
    else:
        soft_prompt = torch.distributions.uniform.Uniform(-1., 1.).sample((seq_len, embedding_dim))
    soft_prompt.requires_grad = True
    return soft_prompt, embedding_dim


def save_promptshield_state(path, promptshield):

    adv_visual_prompt_state = promptshield.adv_visual_prompt_module.state_dict() if promptshield.adv_visual_prompt_module is not None  else None

    def_visual_prompt_state = promptshield.def_visual_prompt_module.state_dict() if promptshield.def_visual_prompt_module is not None else None
    def_text_prompt_state = promptshield.def_text_prompt_module.state_dict() if promptshield.def_text_prompt_module is not None else None
    
    save_dict = {
        # config 
        "adv_visual_prompt_length": promptshield.adv_visual_prompt_length,
        "visual_prompt_length": promptshield.visual_prompt_length,
        "text_prompt_length": promptshield.text_prompt_length,   # 2 ~ 4个token  
        "text_embedding_dim": promptshield.text_embedding_dim,
        "visual_embedding_dim": promptshield.visual_embedding_dim,

        # adv_prompt
        "adv_visual_prompt_state": adv_visual_prompt_state,

        # def_prompt
        "def_visual_prompt_state": def_visual_prompt_state,
        "def_text_prompt_state": def_text_prompt_state,

    }

    torch.save(save_dict, path)

def load_promptshield_state(path, promptshield, device="cuda"):

    checkpoint = torch.load(path, map_location='cpu')
    
    # load config 
    adv_visual_prompt_length = checkpoint["adv_visual_prompt_length"]
    visual_prompt_length = checkpoint["visual_prompt_length"]
    text_prompt_length = checkpoint["text_prompt_length"]
    text_embedding_dim = checkpoint["text_embedding_dim"]
    visual_embedding_dim = checkpoint["visual_embedding_dim"]

    adv_visual_prompt_length = 10
    visual_prompt_length = 16
    text_prompt_length = 10
    
    promptshield.visual_prompt_length = visual_prompt_length
    promptshield.text_prompt_length = text_prompt_length
    promptshield.text_embedding_dim = text_embedding_dim
    promptshield.visual_embedding_dim = visual_embedding_dim

    # define and load adv_prompt 
 
    promptshield.adv_visual_prompt_module = VisualPrompt(adv_visual_prompt_length, visual_embedding_dim)

    if checkpoint["adv_visual_prompt_state"] is not None:
        try:
            promptshield.adv_visual_prompt_module.load_state_dict(checkpoint["adv_visual_prompt_state"])
        except:
            state_dict = {k.replace("module.", "", 1): v for k, v in checkpoint["adv_visual_prompt_state"].items()}
            promptshield.adv_visual_prompt_module.load_state_dict(state_dict)


    # define and load def_prompt 

    promptshield.def_visual_prompt_module = VisualPrompt(visual_prompt_length, visual_embedding_dim)
    promptshield.def_text_prompt_module = TextPrompt(text_prompt_length, text_embedding_dim, soft_prompt=None)

    if checkpoint["def_visual_prompt_state"] is not None:
        try:
            promptshield.def_visual_prompt_module.load_state_dict(checkpoint["def_visual_prompt_state"])
        except:
            state_dict = {k.replace("module.", "", 1): v for k, v in checkpoint["def_visual_prompt_state"].items()}
            promptshield.def_visual_prompt_module.load_state_dict(state_dict)
    
    if checkpoint["def_text_prompt_state"] is not None:
        try:
            promptshield.def_text_prompt_module.load_state_dict(checkpoint["def_text_prompt_state"])
        except:
            state_dict = {k.replace("module.", "", 1): v for k, v in checkpoint["def_text_prompt_state"].items()}
            promptshield.def_text_prompt_module.load_state_dict(state_dict)

    return promptshield


class PromptAttack:
    def __init__(self, model, processor, **kwargs):
        
        super(PromptAttack, self).__init__()
        self.model = model
        self.processor = processor
    
    def image_attack(self, outer_epoch, dataloader, inner_optimizer=None, inner_scheduler=None, adv_visual_prompt_module=None, def_visual_prompt_module=None, def_text_prompt_module=None, options=None, config=None):

        attack_epochs = config["attack_epochs"]
        lr = config["lr"]
        weight_decay = config["weight_decay"]
        beta1 = config["beta1"] 
        beta2 = config["beta2"] 
        prompt_max_norm = config["prompt_max_norm"]

        # umodel = self.model.module if(options.distributed) else self.model
        model = self.model
        model.eval() 

        if inner_optimizer is None:  
            inner_optimizer = torch.optim.AdamW(adv_visual_prompt_module.parameters(), lr=lr, betas = (beta1, beta2), weight_decay=weight_decay)
            # inner_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=dataloader.num_batches * attack_epochs, num_cycles=0.5)

        criterion = nn.CrossEntropyLoss().to(options.device)
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(attack_epochs):
            for index, batch in enumerate(dataloader): 
                step = dataloader.num_batches * epoch + index
                if inner_scheduler is not None:
                    inner_scheduler.step()
                    # scheduler.step()
                if options.master and step % 50 == 0:
                    current_lr = inner_optimizer.param_groups[0]['lr']
                    print(f"epoch:{epoch + 1}/{attack_epochs},step:{step},current_lr:{current_lr}\n")
 
                input_ids = batch["input_ids"].to(device=options.device)
                attention_mask = batch["attention_mask"].to(device=options.device) 
                pixel_values = batch["pixel_values"].to(device=options.device) 

                image_embeddings = get_image_embeding(model, pixel_values)
                natural_visual_features = get_image_features(model, def_prompt_module=def_visual_prompt_module, adv_prompt_module=None, image_embeddings=image_embeddings, options=options)
            
                word_embeddings = get_text_embeding(model, input_ids)
                natural_text_features = get_text_features(model, prompt_module = def_text_prompt_module, word_embeddings = word_embeddings, eos_position=-1, options=options)   

                adv_visual_features = get_image_features(model, def_prompt_module=def_visual_prompt_module, adv_prompt_module=adv_visual_prompt_module, image_embeddings=image_embeddings, options = options)
               
                # (batch_size * n) * (n * classes) ---> (batch_size * classes)
                #  output = CLIPOutput(adv_visual_features, natural_text_features)
                # contrastive_loss = get_loss(model, output, criterion, target=None, temp=0.01, loss_type = "adv_image_loss", options=options)
               
                contrastive_loss = (-1.0) * F.cosine_similarity(adv_visual_features, natural_text_features, dim=1).mean()
                adv_loss, mse_loss, cos_sim_loss = adversarial_loss(adv_visual_features, natural_visual_features)
                # temp = 0.1
                div_and_align_loss, align_loss, div_loss = div_align_loss(adv_visual_features, temp=0.1, options=options)

                # loss 
                loss = (-1.0) * (contrastive_loss + adv_loss + div_and_align_loss) 

                loss.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(adv_visual_prompt_module.parameters(), max_norm = prompt_max_norm)
                inner_optimizer.step()
                inner_optimizer.zero_grad()

                torch.cuda.empty_cache()

                if options.master and (index + 1) % 10 == 0:
                    print(f"image_attack, epoch:{epoch + 1}/{attack_epochs}, step:{step}, loss:{loss}, contrastive_loss:{contrastive_loss}, adv_loss:{adv_loss}, mse_loss:{mse_loss}, cos_sim_loss:{cos_sim_loss}, div_and_align_loss:{div_and_align_loss}, align_loss:{align_loss}, div_loss:{div_loss},  total_norm:{total_norm}\n")
           
            if options.master:
                print(f"image_attack, epoch:{epoch + 1}/{attack_epochs}, loss:{loss}, contrastive_loss:{contrastive_loss}, adv_loss:{adv_loss}, mse_loss:{mse_loss}, cos_sim_loss:{cos_sim_loss}, div_and_align_loss:{div_and_align_loss},\n")

        return adv_visual_prompt_module
    

class PromptShield(torch.nn.Module):
    """
    A dual-prompt-based adversarial training, which supports prompt injection and updating on both the image and text sides.

    Attributes:
        - Prompt_adv: {adv_image_embedding, adv_text_embedding}
        - Prompt_def: {def_image_embedding, def_text_embedding}
    
    Methods:
        - image_attack(inputs): return  adv_image_embedding (used for image attack guidance)
        - text_attack(inputs): return adv_text_embedding (used for text attack guidance)
    
    """

    def __init__(self, task_config):

        super(PromptShield, self).__init__()

        self.task_config =  task_config
        
        self.adv_visual_prompt_length = None
        self.visual_prompt_length = None
        self.text_prompt_length = None

        self.text_embedding_dim = None
        self.visual_embedding_dim = None  
     
        # 对抗 Prompt（扰动用）- 训练时进行梯度上升
       
        self.warmup_adv_visual_prompt_module = None
        self.adv_visual_prompt_module = None
         
        # 防御 Prompt（微调用）- 训练时进行梯度下降
        # self.def_prompt_length = def_prompt_length

        self.warmup_def_visual_prompt_module = None
        self.warmup_def_text_prompt_module = None

        self.def_visual_prompt_module = None
        self.def_text_prompt_module = None

        self.ema_visual_prompt_module = None
        self.ema_text_prompt_module = None

        self.model = None
        self.attacker = None
    

    def adv_train(self, outer_epoch, model, processor, optimizer, scheduler,  dataloader, scaler, options, config=None, distributed=False, visual_attack_config=None, schedule=None, total_epoch = 5):    
    
        if(options.distributed): dataloader.sampler.set_epoch(outer_epoch)

        self.attacker = PromptAttack(model, processor)
        torch.autograd.set_detect_anomaly(True)
        criterion = nn.CrossEntropyLoss().to(options.device) 
        
        batch_size = schedule["batch_size"]
        modulo = max(1, int(dataloader.num_batches))
        
        start = time.time()
        if options.master:
            print(f"batch_size:{batch_size}, Num_batches:{dataloader.num_batches}, modulo:{modulo}\n")

        if options.master:
            print(f"==========Start attack on the visual side============")
            self.adv_visual_prompt_module = self.attacker.image_attack(outer_epoch, dataloader, adv_visual_prompt_module = self.adv_visual_prompt_module, def_visual_prompt_module = self.def_visual_prompt_module, def_text_prompt_module = self.def_text_prompt_module, options = options, config = visual_attack_config, total_epoch=total_epoch) 
            
        
        if options.master:
            print(f"==========Start Prompt Tuning============\n")  

        epochs = config["epochs"]
        for epoch in range(epochs):
            for index, batch in enumerate(dataloader): 
                step = ((outer_epoch - 1) * epochs + epoch) * dataloader.num_batches + index
                if scheduler is not None:
                    scheduler.step()

                if options.master and step % 100 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"outer_epoch:{outer_epoch},inner_epoch:{epoch},step:{step},current_lr:{current_lr}\n")

                input_ids = batch["input_ids"].to(device=options.device)
                attention_mask = batch["attention_mask"].to(device=options.device) 
                pixel_values = batch["pixel_values"].to(device=options.device) 
                  
                image_embeddings = get_image_embeding(model, pixel_values)
                word_embeddings = get_text_embeding(model, input_ids)

                # contrastive_loss
                def_visual_features = get_image_features(model, def_prompt_module = self.def_visual_prompt_module, adv_prompt_module = None, image_embeddings = image_embeddings) 
                def_text_features = get_text_features(model, prompt_module = self.def_text_prompt_module, word_embeddings = word_embeddings, eos_position=-1, text_prompt_length = self.text_prompt_length, options = options)
                outputs = CLIPOutput(def_visual_features, def_text_features)
                contrastive_loss = get_contrastive_loss(model, outputs, criterion, temp=0.07, options=options)

                adv_visual_features = get_image_features(model, def_prompt_module = self.def_visual_prompt_module, adv_prompt_module = self.adv_visual_prompt_module, image_embeddings=image_embeddings) 
                adv_visual_outputs = CLIPOutput(adv_visual_features, def_text_features)
                adv_contrastive_loss = get_contrastive_loss(model, adv_visual_outputs, criterion, temp=0.07, options=options)
                
                # rkd_loss
                image_features = model.get_image_features(pixel_values) 
                text_features = model.get_text_features(input_ids = input_ids, attention_mask = attention_mask)
                
                teacher_outputs = CLIPOutput(image_features, text_features)
                student_adv_outputs = CLIPOutput(adv_visual_features, def_text_features)
                student_outputs = CLIPOutput(def_visual_features, def_text_features)

                rkd_kl_loss = relational_distillation_loss(teacher_outputs, student_outputs, temp=0.10, options=options, loss_type="kl_div", all_gather=True)
                adv_rkd_kl_loss = relational_distillation_loss(teacher_outputs, student_adv_outputs, temp=0.10, options=options, loss_type="kl_div", all_gather=True)
                
                # rkd_mse_loss = relational_distillation_loss(teacher_outputs, student_outputs, temp=2.0, options=options, loss_type="mse", all_gather=False)
                # adv_rkd_mse_loss = relational_distillation_loss(teacher_outputs, student_adv_outputs, temp=5.0, options=options, loss_type="mse", all_gather=False)

                rkd_loss, adv_rkd_loss = rkd_kl_loss, adv_rkd_kl_loss
                # rkd_loss, adv_rkd_loss = rkd_mse_loss, adv_rkd_mse_loss

                distill_loss = 10.0 * rkd_loss + 10.0 * adv_rkd_loss 

                loss = contrastive_loss +  adv_contrastive_loss + distill_loss 

                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']], max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()  

                torch.cuda.empty_cache() 
                end = time.time()
                    
                if options.master and ((index + 1) % 10 == 0):   
                    print(f"outer_epoch:{outer_epoch}, inner_epoch:{epoch + 1}/{epochs}, index:{index}, loss:{loss}, contrastive_loss:{contrastive_loss}, adv_contrastive_loss:{adv_contrastive_loss}, distill_loss:{distill_loss}, rkd_loss:{rkd_loss:.5f}, adv_rkd_loss:{adv_rkd_loss:.5f}, mse_loss:{mse_loss:.5f}\n")

                if(options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1))):
                    print(f"outer_epoch:{outer_epoch}, inner_epoch:{epoch + 1}/{epochs}, loss:{loss}, contrastive_loss:{contrastive_loss}, adv_contrastive_loss:{adv_contrastive_loss},  distill_loss:{distill_loss}, time:{end - start}, lr:{optimizer.param_groups[0]['lr']}\n")
                    start = time.time()

    def adv_prompt_tuing(self, rank, task_config, options):

        model = self.task_config["model"]
        test_model = task_config['test_model']
        processor = self.task_config["processor"]
        train_dataset = self.task_config["train_dataset"]
        poisoned_test_dataset = self.task_config["poisoned_test_dataset"]  

        schedule = self.task_config["schedule"]

        free_port = schedule["free_port"]
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
        checkpoint_dir = schedule["checkpoint_dir"]

        test_epoch_interval = schedule["test_epoch_interval"]
        save_epoch_interval = schedule["save_epoch_interval"]
  
        # the process index
        options.rank = rank
        options.master = rank == 0
        
        # set_logger(rank = rank, logger = logger, distributed = options.distributed)

        if(options.device == "cuda"):
            options.device += ":" + str(device_ids[options.rank] if options.distributed else device_id)

        print(f"Rank:{options.rank}, Using {options.device} device")
      
        if options.distributed:   
            # dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)   
            
            port = f'tcp://127.0.0.1:{free_port}'
            dist.init_process_group(backend = options.distributed_backend, init_method = port, world_size = options.num_devices, rank = options.rank)

        defense_config = self.task_config["defense_config"]
        visual_attack_config = self.task_config["visual_attack_config"]
        
        batch_size = schedule["batch_size"]
        batch_size = batch_size // options.num_devices

        # load model
        model, processor = self.task_config["model"], self.task_config["processor"]
        visual_embedding_dim = model.visual.conv1.out_channels
        self.visual_embedding_dim = visual_embedding_dim
        token_embedding_table = model.token_embedding
        soft_prompt, embedding_dim = initialize_soft_prompt(token_embedding_table, self.text_prompt_length, use_vocab=False)
        self.text_embedding_dim = embedding_dim

        self.visual_prompt_length = defense_config["visual_prompt_length"]
        self.text_prompt_length = defense_config["text_prompt_length"]
        self.adv_visual_prompt_length = visual_attack_config["adv_visual_prompt_length"]

        if options.master: 
            print(f"visual_prompt_length:{self.visual_prompt_length}\ntext_prompt_length:{self.text_prompt_length}\nadv_visual_prompt_length:{self.adv_visual_prompt_length}\n")
            print(f"visual_embedding_dim:{visual_embedding_dim},text_embedding_dim:{embedding_dim}\n")

        self.def_text_prompt_module = TextPrompt(self.text_prompt_length, self.text_embedding_dim, soft_prompt=None).to(options.device)
        self.def_visual_prompt_module = VisualPrompt(self.visual_prompt_length, visual_embedding_dim).to(options.device)
        self.adv_visual_prompt_module = VisualPrompt(self.adv_visual_prompt_length, visual_embedding_dim).to(options.device)

        if(options.device == "cpu"):
            model.float()
        else:
            torch.cuda.set_device(device_ids[options.rank] if options.distributed else device_id)
            model.to(options.device)
            self.def_visual_prompt_module.to(options.device)
            self.def_text_prompt_module.to(options.device)
            self.adv_visual_prompt_module.to(options.device)

            if(options.distributed):
                # model = DDP(model, device_ids = [device_ids[options.rank]])
                self.def_visual_prompt_module = DDP(self.def_visual_prompt_module, device_ids=[device_id])
                self.def_text_prompt_module = DDP(self.def_text_prompt_module, device_ids=[device_id])
                self.adv_visual_prompt_module = DDP(self.adv_visual_prompt_module, device_ids=[device_id])

        umodel = model.module if(options.distributed) else model
        unwrapped_def_visual_prompt_module = self.def_visual_prompt_module.module if(options.distributed) else self.def_visual_prompt_module
        unwrapped_def_text_prompt_module = self.def_text_prompt_module.module if(options.distributed) else self.def_text_prompt_module
        unwrapped_adv_visual_prompt_module = self.adv_visual_prompt_module.module if(options.distributed) else self.adv_visual_prompt_module

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

        if options.master:
            print(f"num_samples:{train_dataloader.num_samples}, num_batches:{train_dataloader.num_batches}\n")
 
        optimizer = None
        scheduler = None

        lr = schedule["lr"]
        beta1 = schedule["beta1"]
        beta2 = schedule["beta2"]
        eps = schedule["eps"]
        weight_decay = schedule["weight_decay"]
        num_warmup_steps = schedule["num_warmup_steps"]
        optimizer = torch.optim.AdamW(list(self.def_visual_prompt_module.parameters()) + list(self.def_text_prompt_module.parameters()), lr=lr, betas = (beta1, beta2), eps=eps, weight_decay=weight_decay)
        
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=train_dataloader.num_batches * epochs * defense_config["epochs"], num_cycles=0.5)
            
        scaler = GradScaler()
        start_epoch = 0
        if(options.progressive):
            options.progressive_epochs = list(map(int, options.progressive_epochs))
            if (start_epoch in options.progressive_epochs):
                options, data = progressive_removal(options, model, processor, data, start_epoch)

        # train
        model.eval() 
        os.makedirs(checkpoint_dir, exist_ok = True)
        total_time = 0 
        for epoch in range(start_epoch + 1, epochs + 1):
            # if epoch > 1:
            #     scheduler.step()
            if options.master:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Starting Epoch {epoch}/{epochs},current_lr:{current_lr}")
                print(f"visual_attack_config, lr:{visual_attack_config['lr']}\n")

            start = time.time()

            self.adv_train(epoch, model, processor, optimizer, scheduler, train_dataloader, scaler, options, config=defense_config, distributed=options.distributed, visual_attack_config=visual_attack_config, schedule=schedule, total_epoch=epochs)

            end = time.time()
            
            if(options.master): 
                total_time = total_time + (end - start)
                print(f"Finished Epoch {epoch}/{epochs}, Time Taken: {end - start:.3f}, total_time:{total_time}")

            test_epoch_interval = 1
            if(options.master and epoch % test_epoch_interval==0): 
                with torch.no_grad():
                
                    print(f"Test robustness with def_prompt_module\n")

                    def_visual_prompt_module, def_text_prompt_module = deepcopy(unwrapped_def_visual_prompt_module), deepcopy(unwrapped_def_text_prompt_module)
                            
                    results, all_logits, all_labels = self.get_zeroshot_metrics(model, processor, poisoned_test_dataset, def_visual_prompt_module, def_text_prompt_module, options)
                    
                    poisoned_test_indices = poisoned_test_dataset.backdoor_indices
                    clean_test_indices = list(set(range(len(poisoned_test_dataset))) - set(poisoned_test_indices))
                    benign_acc = compute_accuracy(all_logits[clean_test_indices], all_labels[clean_test_indices],topk=(1,3,5))
                    poisoned_acc = compute_accuracy(all_logits[poisoned_test_indices], all_labels[poisoned_test_indices],topk=(1,3,5))                                                                                                                                           
                    print("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_acc, poisoned_acc))

                    save_epoch_interval = 1
                    if(epoch % save_epoch_interval == 0):
                        state_path = os.path.join(checkpoint_dir, f"PromptShield_state_epoch_{epoch}.pt")
                        save_promptshield_state(state_path, self)
                        print(f"save promptshield state into {state_path}")
        
            if(options.progressive):
                if epoch in options.progressive_epochs:
                    options, data = progressive_removal(options, model, processor, data, epoch)
                if epoch == options.stop_epoch:
                    return


        if(options.distributed):
            dist.destroy_process_group()   

        if(options.wandb and options.master):
            wandb.finish()

    def shield_distillation(self, rank, task_config, options):

        model = task_config["model"]
        reference_model = task_config["reference_model"]

        processor = task_config["processor"]
        train_dataset = task_config["train_dataset"]
        poisoned_test_dataset = task_config["poisoned_test_dataset"]  

        shield_distillation_config = task_config["shield_distillation"]
        schedule = task_config["schedule"]
        free_port = schedule["free_port"]

        device_ids = schedule["device_ids"]
        device_id = schedule["device_id"]
        num_workers = schedule["num_workers"]
       
        checkpoint = schedule["checkpoint"]
      
        test_epoch_interval = schedule["test_epoch_interval"]
        save_epoch_interval = schedule["save_epoch_interval"]

        # the process index
        options.rank = rank
        options.master = rank == 0
  
        if(options.device == "cuda"):
            options.device += ":" + str(device_ids[options.rank] if options.distributed else device_id)

        print(f"Using {options.device} device")

        if(options.distributed):
            port = f'tcp://127.0.0.1:{free_port}'
            dist.init_process_group(backend = options.distributed_backend, init_method = port, world_size = options.num_devices, rank = options.rank)
            
        if(options.device == "cpu"):
            model.float()
        else:
            torch.cuda.set_device(device_ids[options.rank] if options.distributed else device_id)
            model.to(options.device)
            reference_model.to(options.device)

            self.def_visual_prompt_module.to(options.device)
            self.def_text_prompt_module.to(options.device)
            self.adv_visual_prompt_module.to(options.device)

            if(options.distributed):
                model = DDP(model, device_ids = [device_ids[options.rank]])
                reference_model = DDP(reference_model, device_ids = [device_ids[options.rank]])
                self.def_visual_prompt_module = DDP(self.def_visual_prompt_module, device_ids=[device_id])
                self.def_text_prompt_module = DDP(self.def_text_prompt_module, device_ids=[device_id])
                self.adv_visual_prompt_module = DDP(self.adv_visual_prompt_module, device_ids=[device_id])

        umodel = model.module if(options.distributed) else model

        # load data
        batch_size = schedule["batch_size"]
        batch_size = batch_size // options.num_devices
        sampler = DistributedSampler(train_dataset) if(options.distributed) else None
        dataloader = DataLoader(
            train_dataset, 
            batch_size = batch_size, 
            shuffle = (sampler is None), 
            num_workers = num_workers, 
            pin_memory = True, 
            sampler = sampler, 
            drop_last = True
        )
        
        dataloader.num_samples = len(dataloader) * batch_size 
        dataloader.num_batches = len(dataloader) 

        if options.master:
            print(f"num_samples:{dataloader.num_samples}, batch_size:{batch_size}, num_batches:{dataloader.num_batches}\n")
                
        epochs = shield_distillation_config["epochs"]
        checkpoint_dir = shield_distillation_config["checkpoint_dir"]
        temp = shield_distillation_config["temp"]

        lr = shield_distillation_config["lr"]
        beta1 = shield_distillation_config["beta1"]
        beta2 = shield_distillation_config["beta2"]
        eps = shield_distillation_config["eps"]
        weight_decay = shield_distillation_config["weight_decay"]
        num_warmup_steps = shield_distillation_config["num_warmup_steps"]
        num_warmup_steps = 100

        optimizer = None
        scheduler = None
        if(dataloader is not None):        
            weight_decay_parameters = []
            no_weight_decay_parameters = []
            for name, parameter in model.named_parameters():
                if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                    weight_decay_parameters.append(parameter)    
                if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                    no_weight_decay_parameters.append(parameter)

            optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": weight_decay}], lr = lr, betas = (beta1, beta2), eps = eps)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=dataloader.num_batches * epochs, num_cycles=0.5)

        cudnn.benchmark = True
        cudnn.deterministic = False

        # train
        if(dataloader is not None):
            # options.checkpoint_dir = os.path.join(options.log_dir_path, "checkpoints")
            
            os.makedirs(checkpoint_dir, exist_ok = True)
            self.def_visual_prompt_module = self.def_visual_prompt_module.to(options.device)
            self.def_text_prompt_module = self.def_text_prompt_module.to(options.device)
            start_epoch= 0
            for epoch in range(start_epoch + 1, epochs + 1):
             
                if(options.master): 
                    print(f"Starting Epoch {epoch}")

                start = time.time()
                self.distillation_train(epoch, reference_model, model, optimizer, scheduler, dataloader, options)
                end = time.time()

                test_epoch_interval = 1
                if(options.master and epoch % test_epoch_interval==0): 
                    with torch.no_grad():
                        umodel.eval()
                        results, all_logits, all_labels = get_zeroshot_metrics(umodel, processor, poisoned_test_dataset, options)          
                        poisoned_test_indices = poisoned_test_dataset.backdoor_indices
                        clean_test_indices = list(set(range(len(poisoned_test_dataset))) - set(poisoned_test_indices))
                        benign_acc = compute_accuracy(all_logits[clean_test_indices], all_labels[clean_test_indices], topk=(1,3,5))
                        poisoned_acc = compute_accuracy(all_logits[poisoned_test_indices], all_labels[poisoned_test_indices], topk=(1,3,5))                                                                                                                                       
                        print(f"epoch:{epoch / (epochs + 1)}, Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_acc, poisoned_acc))      
                
                if(options.master): 
                    print(f"Finished Epoch {epoch / (epochs + 1)}, Time Taken: {end - start:.3f}")
                
                if(options.progressive):
                    if epoch in options.progressive_epochs:
                        options, data = progressive_removal(options, model, processor, data, epoch)
                
                    if epoch == options.stop_epoch:
                        return
                    
            # save checkpoint
            if(options.master):
                checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                model_path = os.path.join(checkpoint_dir, f"purified_model.pt")
                torch.save(checkpoint, model_path)
                print(f"Save model into {model_path}")

        if(options.distributed):
            dist.destroy_process_group()

        if(options.wandb and options.master):
            wandb.finish()


    def distillation_train(self, epoch, reference_model, model, optimizer, scheduler, dataloader, options):    
    
        if(options.distributed): dataloader.sampler.set_epoch(epoch)
     
        torch.autograd.set_detect_anomaly(True)

        reference_model.eval()
        reference_umodel = reference_model.module if(options.distributed) else reference_model
        
        model.train()
        umodel = model.module if(options.distributed) else model

        criterion = nn.CrossEntropyLoss().to(options.device) #if not options.unlearn else nn.CrossEntropyLoss(reduction = 'none').to(options.device)

        for index, batch in enumerate(dataloader): 
            step = dataloader.num_batches * (epoch - 1) + index
            scheduler.step()
            if options.master and step % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"epoch:{epoch}, step:{step}, current_lr:{current_lr}\n")

            input_ids = batch["input_ids"].to(device=options.device)
            attention_mask = batch["attention_mask"].to(device=options.device)
            pixel_values = batch["pixel_values"].to(device=options.device)

            word_embeddings = get_text_embeding(reference_umodel, input_ids)
            image_embeddings = get_image_embeding(reference_umodel, pixel_values)

            with autocast(enabled=False):
                                  
                image_features = umodel.get_image_features(pixel_values) 
                text_features = umodel.get_text_features(input_ids = input_ids, attention_mask = attention_mask)
                
                outputs = CLIPOutput(image_features, text_features) 
                contrastive_loss = get_contrastive_loss(model, outputs, criterion, temp=0.07,  options=options)
       
                # output from reference model
                ref_def_visual_features = get_image_features(reference_umodel, def_prompt_module = self.def_visual_prompt_module, adv_prompt_module = None, image_embeddings = image_embeddings) 
                ref_def_text_features = get_text_features(reference_umodel, prompt_module = self.def_text_prompt_module, word_embeddings = word_embeddings, eos_position=-1, text_prompt_length = self.text_prompt_length, options = options)
                ref_def_outputs = CLIPOutput(ref_def_visual_features, ref_def_text_features)
                rkd_loss = relational_distillation_loss(ref_def_outputs, outputs, temp=0.07, options=options, loss_type="kl_div", all_gather=True) 

                loss = 1.0 * contrastive_loss + 1.0 * rkd_loss 

                loss.backward()
                optimizer.step()  
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
                optimizer.zero_grad()  
                torch.cuda.empty_cache() 

                if options.master and ((index + 1) % 10 == 0):   
                    print(f"epoch:{epoch}, index:{index}, loss:{loss}, contrastive_loss:{contrastive_loss}, rkd_loss:{rkd_loss:.5f} \n")

   
    def get_zeroshot_metrics(self, model, processor, test_dataset, def_visual_prompt_module, def_text_prompt_module, options, adv_prompt_module=None, eval_test_data_classes_dir=None):
   
        print("Started zeroshot testing")

        umodel = model.module if isinstance(model,nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel) else model
        
        if next(umodel.parameters()).device.type == "cpu":
            umodel = umodel.to(options.device)

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size = 128, 
            num_workers = 4, 
            sampler = None
        )
        test_dataloader.num_samples = len(test_dataset)
        test_dataloader.num_batches = len(test_dataloader)

        if eval_test_data_classes_dir is None:
            eval_test_data_classes_dir = test_dataset.classes_path
        
        # print(f"eval_test_data_classes_dir:{eval_test_data_classes_dir}/n")
        # config = eval(open(eval_test_data_classes_dir, "r").read())
    
        with open(eval_test_data_classes_dir, "r") as f:
            config_text = f.read()
            config = eval(config_text, {"__builtins__": {}})

        classes, templates = config["classes"], config["templates"]

        # print(f"classes:{len(classes)},templates:{len(templates)}\n")
    
        with torch.no_grad():
            text_embeddings = []
            for c in tqdm(classes):
                text = [template(c) for template in templates]
                text_tokens = processor.process_text(text)
                text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.device), text_tokens["attention_mask"].to(options.device) 
                
                word_embeddings = get_text_embeding(umodel, text_input_ids)

                if def_text_prompt_module is not None:
                    text_embedding = get_text_features(umodel, prompt_module = def_text_prompt_module, word_embeddings = word_embeddings, eos_position=-1, options = options)
                elif self.def_text_prompt_module is not None:
                    text_embedding = get_text_features(umodel, prompt_module = self.def_text_prompt_module, word_embeddings = word_embeddings, eos_position=-1, options = options)
                else:
                    text_embedding = umodel.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            
                text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
                text_embedding = text_embedding.mean(dim = 0)
                text_embedding /= text_embedding.norm()
                text_embeddings.append(text_embedding)
            text_embeddings = torch.stack(text_embeddings, dim = 1).to(options.device)

        # print(f"text_embeddings:{text_embeddings.shape}\n")
    
        all_logits = []
        all_labels = []
        with torch.no_grad():
            # topk = [1, 3, 5, 10]
            topk = [1]
            correct = {k: 0 for k in topk}
            total = 0
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)

                pixel_values = processor.process_image(image).to(options.device, non_blocking = True)
                image_embeddings = get_image_embeding(umodel, pixel_values)

                # image_embedding = get_image_features(umodel, prompt_module = self.def_visual_prompt_module, image_embeddings=image_embeddings)

                if def_visual_prompt_module is not None:
                    image_embedding = get_image_features(umodel, def_prompt_module = def_visual_prompt_module, adv_prompt_module = adv_prompt_module, image_embeddings=image_embeddings)
                elif self.def_visual_prompt_module is not None:
                    image_embedding = get_image_features(umodel, def_prompt_module = def_visual_prompt_module, adv_prompt_module = adv_prompt_module, image_embeddings=image_embeddings)
                elif adv_prompt_module is not None:
                    image_embedding = get_image_features(umodel, def_prompt_module = None, adv_prompt_module = adv_prompt_module, image_embeddings=image_embeddings)
                else:
                    image_embedding = umodel.get_image_features(pixel_values) 

                image_embedding /= image_embedding.norm(dim = -1, keepdim = True)
                
                logits = (image_embedding @ text_embeddings)

                ranks = logits.topk(max(topk), 1)[1].T

                predictions = ranks == label
                total += predictions.shape[1]
                for k in topk:
                    correct[k] += torch.sum(torch.any(predictions[:k], dim = 0)).item() 

                all_logits.append(logits)
                all_labels.append(label)
            
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        
        results = {f"zeroshot_top{k}": correct[k] / total for k in topk}

        return results, all_logits, all_labels
    
    def get_linear_probe_metrics(self, model, processor, train_dataset, test_dataset, options):
    
        print("Started linear probe testing")
    
        sampler = DistributedSampler(train_dataset) if(options.distributed) else None

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size = 128, 
            shuffle = (sampler is None), 
            num_workers = options.num_workers, 
            pin_memory = True, 
            sampler = sampler, 
            drop_last = True
        )
        train_dataloader.num_samples = len(train_dataloader) * 128 
        train_dataloader.num_batches = len(train_dataloader)

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size = 128, 
            num_workers = 8, 
            sampler = None
        )
        test_dataloader.num_samples = len(test_dataset)
        test_dataloader.num_batches = len(test_dataloader)

        print(f"Number of train examples: {train_dataloader.num_samples}")
        print(f"Number of test examples: {test_dataloader.num_samples}")

        model = model.to(options.device)
        model.eval()
        umodel = model.module if(options.distributed) else model
        
        images = None
        labels = None
        with torch.no_grad():
            for image, label in tqdm(train_dataloader):

                # image = umodel.get_image_features(image.to(options.device)).cpu()
                pixel_values = processor.process_image(image).to(options.device, non_blocking = True)
                image_embeddings = get_image_embeding(umodel, pixel_values)
                image = get_image_features(umodel, def_prompt_module = self.def_visual_prompt_module, image_embeddings=image_embeddings) 
                
                images = torch.cat([images, image], dim = 0) if(images is not None) else image
                labels = torch.cat([labels, label], dim = 0) if(labels is not None) else label

        train_dataset = torch.utils.data.TensorDataset(images, labels)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = options.batch_size, shuffle = True)
        
        input_dim = umodel.text_projection.shape[1]

        output_dim, metric = get_odim_metric(options)

        classifier = LogisticRegression(input_dim = input_dim, output_dim = output_dim).to(options.device)
        optimizer = optim.AdamW([{"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" in name) and parameter.requires_grad)], "weight_decay": 0}, {"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" not in name) and parameter.requires_grad)], "weight_decay": 0.01}])
        scheduler = cosine_scheduler(optimizer, 0.005, 0, len(train_dataloader) * options.linear_probe_num_epochs)
        criterion = nn.CrossEntropyLoss().to(options.device)
        
        pbar = tqdm(range(options.linear_probe_num_epochs))
        for epoch in pbar:
            cbar = tqdm(train_dataloader, leave = False)
            for index, (image, label) in enumerate(cbar):
                step = len(train_dataloader) * epoch + index
                scheduler(step)
                image, label = image.to(options.device), label.to(options.device)
                logit = classifier(image)
                optimizer.zero_grad()
                loss = criterion(logit, label)
                loss.backward()
                optimizer.step()
                cbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
            pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
            
            print(f"epoch:{epoch}, loss:{loss.item()}, lr:{optimizer.param_groups[0]['lr']}")
          
        classifier.eval()
        with torch.no_grad():
            if(metric == "accuracy"):
                correct = 0
                for image, label in tqdm(test_dataloader):
                    image, label = image.to(options.device), label.to(options.device)

                    pixel_values = processor.process_image(image).to(options.device, non_blocking = True)
                    image_embeddings = get_image_embeding(umodel, pixel_values)
                    image_embedding = get_image_features(umodel, def_prompt_module = self.def_visual_prompt_module, image_embeddings=image_embeddings) 
                    logits = classifier(image_embedding)
                    # logits = classifier(umodel.get_image_features(image))

                    prediction = torch.argmax(logits, dim = 1)
                    if options.asr:
                        non_label_indices = (label != 954).nonzero().squeeze()
                        if type(non_label_indices) == int or len(non_label_indices):
                            prediction = prediction[non_label_indices]
                        correct += torch.sum(prediction == 954).item()
                    else:
                        correct += torch.sum(prediction == label).item()

                results = {f"linear_probe_accuracy": correct / test_dataloader.num_samples}
            else:
                correct = torch.zeros(output_dim).to(options.device)
                total = torch.zeros(output_dim).to(options.device)
                for image, label in tqdm(test_dataloader):
                    image, label = image.to(options.device), label.to(options.device)

                    pixel_values = processor.process_image(image).to(options.device, non_blocking = True)
                    image_embeddings = get_image_embeding(umodel, pixel_values)
                    image_embedding = get_image_features(umodel, def_prompt_module = self.def_visual_prompt_module, image_embeddings=image_embeddings) 
                    logits = classifier(image_embedding)
                    # logits = classifier(umodel.get_image_features(image))

                    predictions = torch.argmax(logits, dim = 1)
                    
                    temp = torch.zeros(output_dim, len(label)).to(options.device)
                    temp[label, torch.arange(len(label))] = (predictions == label).float()
                    correct += temp.sum(1)
                    temp[label, torch.arange(len(label))] = 1                
                    total += temp.sum(1)

                results = {f"linear_probe_mean_per_class": (correct / total).mean().cpu().item()}
            
        print("Finished linear probe testing")
        return results
    

def get_text_embeding(umodel, input_ids):  
    token_embedding_table = umodel.token_embedding
    word_embeddings = token_embedding_table(input_ids)
    return word_embeddings

def get_text_features(umodel, prompt_module = None, word_embeddings = None, eos_position=-1, text_prompt_length=1, options=None):

    soft_prompt, prompt_pos_embed = prompt_module(word_embeddings.shape[0]) 

    input_embeddings = torch.concat([soft_prompt, word_embeddings], dim=1)
    attention_mask = torch.ones(input_embeddings.shape[:3], dtype=torch.long).to(input_embeddings.device)
    positional_embedding = torch.concat([prompt_pos_embed, umodel.positional_embedding], dim=0)

    # x = self.token_embedding(input_ids).type(self.dtype)  # [batch_size, n_ctx, d_model]
    dtype = umodel.visual.conv1.weight.dtype

    text_embeddings = input_embeddings

    x = text_embeddings.type(dtype) 
    x = x + positional_embedding.type(dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND

    x = umodel.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = umodel.ln_final(x).type(dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), eos_position] @ umodel.text_projection

    return x
    
def get_image_embeding(umodel, pixel_values):
    
    #  shape: (B, 3, H, W)，
    x = umodel.visual.conv1(pixel_values)  # shape: (B, C=768, H_patch, W_patch)
    x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, C, N)
    x = x.permute(0, 2, 1)  # (B, N, C) -> patch embeddings

    return x


def get_image_features(umodel, def_prompt_module = None, adv_prompt_module = None, image_embeddings = None, options=None):
   
    image_tokens = []
    positional_embedding = []
   
    cls_token = umodel.visual.class_embedding
    cls_token = cls_token.unsqueeze(0).unsqueeze(0).expand(image_embeddings.shape[0], -1, -1)  # (B, 1, C)
    
    image_tokens.append(cls_token)
    positional_embedding.append(umodel.visual.positional_embedding[:1])
   
    if def_prompt_module is not None:   
        # def_prompt_module = def_prompt_module.module if isinstance(def_prompt_module,nn.DataParallel) or isinstance(def_prompt_module, nn.parallel.DistributedDataParallel) else def_prompt_module
        def_visual_prompt, prompt_pos_embed = def_prompt_module(image_embeddings.shape[0])
        image_tokens.append(def_visual_prompt)
        positional_embedding.append(prompt_pos_embed)

    if adv_prompt_module is not None:
        # adv_prompt_module = adv_prompt_module.module if isinstance(adv_prompt_module,nn.DataParallel) or isinstance(adv_prompt_module, nn.parallel.DistributedDataParallel) else adv_prompt_module
        adv_visual_prompt, prompt_pos_embed = adv_prompt_module(image_embeddings.shape[0])
        image_tokens.append(adv_visual_prompt)
        positional_embedding.append(prompt_pos_embed)
    
    image_tokens.append(image_embeddings)
    positional_embedding.append(umodel.visual.positional_embedding[1:])
    image_tokens = torch.cat(image_tokens, dim=1)  # (B, 1+P+N, C) 
    positional_embedding = torch.cat(positional_embedding, dim=0)  

    x = image_tokens
    x = x + positional_embedding.to(x.dtype)

    # if options.master:
    #     print(f"x:{x.shape}\n")

    x = umodel.visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = umodel.visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = umodel.visual.ln_post(x[:, 0, :])

    if umodel.visual.proj is not None:
        x = x @ umodel.visual.proj

    return x
