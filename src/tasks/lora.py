# sys
import os
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from clip import clip
from pkgs.CLIPLoRA.loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict
from pkgs.CLIPLoRA.loralib import layers as lora_layers
from pkgs.CLIPLoRA.utils import *
from pkgs.CLIPLoRA.loralib.utils import apply_lora, save_lora, load_lora
# core
from models.model import *
from transformers import get_cosine_schedule_with_warmup
from src.scheduler import cosine_scheduler
from src.task_scheduling import task, progressive_removal
from src.evaluate import get_odim_metric, LogisticRegression
from src.evaluate import get_zeroshot_metrics
from src.data import ImageCaptionDataset, ImageToCaptionDataset

# numpy
import numpy as np
from copy import deepcopy
import time
import wandb
from tqdm import tqdm
# utils
from utils.compute import compute_accuracy
from utils.interact import Log, log, parser
from utils.model_merging.fine_tuning import test 
from utils.interact import Log, log, parser
from utils.model_merging import get_ImageEncoder

def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


# def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
#     VALIDATION = False
    
#     # Textual features
#     print("\nGetting textual features as CLIP's classifier.")
#     textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

#     # Pre-load val features
#     print("\nLoading visual features and labels from val set.")
#     val_features, val_labels = pre_load_features(clip_model, val_loader)

#     # Pre-load test features
#     print("\nLoading visual features and labels from test set.")
#     test_features, test_labels = pre_load_features(clip_model, test_loader)
    
#     test_features = test_features.cuda()
#     test_labels = test_labels.cuda()
 
#     # Zero-shot CLIP
#     clip_logits = logit_scale * test_features @ textual_features
#     zs_acc = cls_acc(clip_logits, test_labels)
#     print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
#     test_features = test_features.cpu()
#     test_labels = test_labels.cpu()
    
    
#     list_lora_layers = apply_lora(args, clip_model)
#     clip_model = clip_model.cuda() 
    
#     if args.eval_only:
#         load_lora(args, list_lora_layers)
#         acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
#         print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
#         return

#     mark_only_lora_as_trainable(clip_model)
#     total_iters = args.n_iters * args.shots
    
#     optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
#     best_acc_val, best_acc_test = 0., 0.
#     best_epoch_val = 0
    
#     # training LoRA
#     scaler = torch.cuda.amp.GradScaler()
#     count_iters = 0
#     finish = False
#     while count_iters < total_iters:
#         clip_model.train()
#         acc_train = 0
#         tot_samples = 0
#         loss_epoch = 0.
#         if args.encoder == 'vision': 
#             text_features = textual_features.t().half()
#         for i, (images, target) in enumerate(tqdm(train_loader)):
            
#             template = dataset.template[0]
#             texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
#             images, target = images.cuda(), target.cuda()
#             if args.encoder == 'text' or args.encoder == 'both':
#                 with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
#                     texts = clip.tokenize(texts).cuda()
#                     class_embeddings = clip_model.encode_text(texts)
#                 text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
#             if args.encoder == 'vision' or args.encoder == 'both':
#                 with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
#                     image_features = clip_model.encode_image(images)
#             else:
#                 with torch.no_grad():
#                     with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
#                         image_features = clip_model.encode_image(images)
#             image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
#             cosine_similarity = logit_scale * image_features @ text_features.t()
#             loss = F.cross_entropy(cosine_similarity, target)
#             acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
#             loss_epoch += loss.item() * target.shape[0]
#             tot_samples += target.shape[0]
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)

#             scaler.update()
#             scheduler.step()
            
#             count_iters += 1
            
#             if count_iters == total_iters:
#                 break
            
#         if count_iters < total_iters:
#             acc_train /= tot_samples
#             loss_epoch /= tot_samples
#             current_lr = scheduler.get_last_lr()[0]
#             print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))

        
#         # Eval
#         if VALIDATION:
#             clip_model.eval()
#             acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
#             print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
        
    
#     acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
#     print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    
#     if args.save_path != None:
#         save_lora(args, list_lora_layers)
#     return


def run_lora(rank, task_config, options):

    model_name = task_config['model_name']
    clip_model = task_config["model"]
    processor = task_config["processor"]

    lora_config = task_config["lora_config"]
    list_lora_layers = apply_lora(clip_model, lora_config)
    clip_model.eval()
    
    # image_encoder = task_config["image_encoder"]
    image_encoder = get_ImageEncoder(clip_model)
    classification_head = task_config["classification_head"]
    train_dataset = task_config["train_dataset"]
    test_dataset = task_config["test_dataset"]

    # schedule
    schedule = task_config["schedule"]
      
    # Distributed Training
    distributed = schedule["distributed"]
    free_port = schedule["free_port"]
    device_ids = schedule["device_ids"]
    device_id = schedule["device_id"]

    # Fine-tuning
    epochs = schedule["epochs"]
    lr = schedule["lr"]
    beta1 = schedule["beta1"]
    beta2 = schedule["beta2"]
    eps = schedule["eps"]
    weight_decay = schedule["weight_decay"]
    num_warmup_steps = schedule["num_warmup_steps"]

    # checkpoint
    checkpoint_dir = task_config["checkpoint_dir"]
    test_epoch_interval = task_config["test_epoch_interval"]
    save_epoch_interval = task_config["save_epoch_interval"]
    log_dir = task_config["log_dir"]
   
    log.set_logger(log_path=log_dir, mode="a")

    # the process index
    options.rank = rank
    options.master = rank == 0
    if(options.device == "cuda"):
        options.device += ":" + str(device_ids[options.rank] if options.distributed else device_id)
    
    log(f"Using {options.device} device")

    if(options.distributed):
        # dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)
        # free_port = 7316
        port = f'tcp://127.0.0.1:{free_port}'
        dist.init_process_group(backend = options.distributed_backend, init_method = port, world_size = options.num_devices, rank = options.rank)

    if(options.device == "cpu"):
        image_encoder.float()
    else:
        torch.cuda.set_device(device_ids[options.rank] if options.distributed else device_id)
        image_encoder.to(options.device)
        classification_head.to(options.device)
        if(options.distributed):
            image_encoder = DDP(image_encoder, device_ids = [device_ids[options.rank]], find_unused_parameters=True)
    uimage_encoder = image_encoder.module if(options.distributed) else image_encoder

    # load data
    # data = load_data(options, processor)
    sampler = DistributedSampler(train_dataset) if(options.distributed) else None
    batch_size = schedule["batch_size"] // options.num_devices
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size = batch_size, 
        shuffle = (sampler is None), 
        num_workers = schedule["num_workers"], 
        pin_memory = True, 
        sampler = sampler, 
        drop_last = True
    )

    train_dataloader.num_samples = len(train_dataloader) * batch_size * options.num_devices
    train_dataloader.num_batches = len(train_dataloader)
    modulo = max(1, int(train_dataloader.num_batches / 5))

    # get optimizer
    loss_fn = torch.nn.CrossEntropyLoss()

    mark_only_lora_as_trainable(uimage_encoder)
    optimizer = torch.optim.AdamW(get_lora_parameters(uimage_encoder), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    total_iters = epochs * train_dataloader.num_batches
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    cudnn.benchmark = True
    cudnn.deterministic = False

    if options.master:
        # evaluate pre-trained model
        log(f"{'='*30} Evaluate Pre-trained Model {'='*30}")
        uimage_encoder.eval()
        all_logits, all_labels = test(deepcopy(uimage_encoder), model_name, test_dataset, options.device)
        accuracy = compute_accuracy(all_logits, all_labels,topk=(1,3,5))
        log("Total samples:{0}, accuracy:{1}".format(len(test_dataset), accuracy)) 
                                                                                             
    # run lora  
    start_epoch = 0
    stop_training = torch.zeros(1, device=options.device)
    if(train_dataloader is not None):
        os.makedirs(checkpoint_dir, exist_ok = True)

        if(options.master): 
            log(f"Starting Epoch {start_epoch}")

        for epoch in range(start_epoch + 1, epochs + 1):
            uimage_encoder.train()   
            classification_head.eval() 
            start = time.time()
            for i, batch in enumerate(train_dataloader):
                step = i + epoch * train_dataloader.num_batches
                scheduler.step()
                optimizer.zero_grad()

                inputs = batch['pixel_values'].to(options.device, non_blocking = True)
                labels = batch['labels'].to(options.device, non_blocking = True)
        
                features = uimage_encoder(inputs)
                logits = classification_head(features)
                loss = loss_fn(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']], 1.0)
                optimizer.step()

                end = time.time()
                if(options.master and (((i + 1) % modulo == 0) or (i == train_dataloader.num_batches - 1))):
                    num_samples = (i + 1) * len(inputs) * options.num_devices
                    dataloader_num_samples = train_dataloader.num_samples
                    log(f"Train Epoch: {epoch:02d} [{num_samples}/{dataloader_num_samples} ({100.0 * (i + 1) / train_dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:.9f}")
                    start = time.time()
                    
            test_epoch_interval = 1 
            if(options.master and epoch % test_epoch_interval == 0): 
                with torch.no_grad():
                    uimage_encoder.eval()
                    all_logits, all_labels = test(deepcopy(uimage_encoder), model_name, test_dataset, options.device)
                    accuracy = compute_accuracy(all_logits, all_labels,topk=(1,3,5))
                    log("Total samples:{0}, accuracy:{1}".format(len(test_dataset), accuracy))                                                                                                                                               
            
            if(options.master): 
                log(f"Finished Epoch {epoch}")
  
            # save checkpoint
            if(options.master and epoch % save_epoch_interval == 0):
                # checkpoint_path = os.path.join(checkpoint_dir, f"model_state_epoch_{epoch}.pth")
                # torch.save(uimage_encoder.state_dict(), checkpoint_path)
                lora_state_path = os.path.join(checkpoint_dir, f"lora_state_epoch_{epoch}.pth")
                save_lora(list_lora_layers, lora_config, lora_state_path)
                log(f"Save model state into {lora_state_path}\n")
            
            # ===== 最简单的 DDP 安全退出 =====
            if options.master and options.progressive and epoch == options.stop_epoch:
                stop_training.fill_(1)

            if options.distributed:
                dist.broadcast(stop_training, src=0)

            if stop_training.item() > 0:
                break

    if(options.distributed):
        dist.destroy_process_group()

    if(options.wandb and options.master):
        wandb.finish()

            
