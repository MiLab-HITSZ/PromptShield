# sys
import os
import sys

from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
# core
# from src.defense import Image_Text_attack
from src.train import train
from src.train import adv_train

from src.scheduler import cosine_scheduler
from src.task_scheduling import task, progressive_removal
from src.train import get_text_embeding, get_text_features, get_image_embeding, get_image_features
# numpy
import time
import wandb
import numpy as np
# from src.logger import get_logger, set_logger
from src.evaluate import get_linear_probe_metrics, get_finetune_metrics, get_zeroshot_metrics, get_validation_metrics
from utils.compute import compute_accuracy, load_state

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
        return prompt_tokens  
    
class TextPrompt(nn.Module):
    def __init__(self, prompt_length, embed_dim, soft_prompt=None):
        super().__init__()
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim

        # prompt tokens
        if soft_prompt is None:
            soft_prompt = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)  # shape: (P, D)
        
        self.soft_prompt = soft_prompt 

        # prompt_pos_embed
        self.prompt_pos_embed = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)  # shape: (1, P, D)

    def forward(self, batch_size):
        # (B, P, D)
        soft_prompts = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return soft_prompts 
 
 
def initialize_soft_prompt(token_embedding_table, seq_len: int=20, use_vocab=False):
    embedding_dim = hidden_size = token_embedding_table.weight.shape[1]
    if use_vocab:
        indices = torch.randperm(token_embedding_table.weight.shape[0])[:seq_len]
        soft_prompt = token_embedding_table.weight[indices].clone().detach()
    else:
        soft_prompt = torch.distributions.uniform.Uniform(-1., 1.).sample((seq_len, embedding_dim))
    soft_prompt.requires_grad = True
    return soft_prompt, embedding_dim


def pre_training(rank, task_config, options):

    model = task_config["model"]
    processor = task_config["processor"]
    train_dataset = task_config["train_data"]
    clean_test_dataset = task_config["clean_test_dataset"]
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

    checkpoint = schedule["checkpoint"]
    checkpoint_finetune = schedule["checkpoint_finetune"]
    complete_finetune = schedule["complete_finetune"]
    checkpoint_dir = schedule["checkpoint_dir"]

    log_dir_path = schedule["log_dir_path"]
    log_iteration_interval = schedule["log_iteration_interval"] 
    test_epoch_interval = schedule["test_epoch_interval"]
    save_epoch_interval = schedule["save_epoch_interval"]


    # the process index
    options.rank = rank
    options.master = rank == 0
    
    # set_logger(rank = rank, logger = logger, distributed = options.distributed)

    if(options.device == "cuda"):
        options.device += ":" + str(device_ids[options.rank] if options.distributed else device_id)
    
    print(f"Using {options.device} device")

    # if(options.master):
    #     print("Params:")

    #     with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
    #         for key in sorted(vars(options)):
    #             value = getattr(options, key)
    #             print(f"{key}: {value}")
    #             file.write(f"{key}: {value}\n")

    if(options.distributed):
        dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)
    
    #     free_port = 7310
    #     port = f'tcp://127.0.0.1:{free_port}'
    #     dist.init_process_group(backend = options.distributed_backend, init_method = port, world_size = options.num_devices, rank = options.rank)

    batch_size = schedule["batch_size"]

    batch_size = batch_size // options.num_devices

    # load model
    model, processor = task_config["model"], task_config["processor"]

    if(options.device == "cpu"):
        model.float()
    else:
        torch.cuda.set_device(device_ids[options.rank] if options.distributed else device_id)
        model.to(options.device)
        if(options.distributed):
            model = DDP(model, device_ids = [device_ids[options.rank]])

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

        for name, parameter in model.named_parameters():
            if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                weight_decay_parameters.append(parameter)
                
            if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                no_weight_decay_parameters.append(parameter)

        optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": weight_decay}], lr = lr, betas = (beta1, beta2), eps = eps)
        
        # num_warmup_steps = 10000
        scheduler = cosine_scheduler(optimizer, lr, num_warmup_steps, train_dataloader.num_batches * epochs)

    # load checkpoint

    start_epoch = 0

    if options.master:
        print(f"checkpoint:{checkpoint}\n")
        
    # load model from checkpoint
    if(checkpoint is not None):
        if(os.path.isfile(checkpoint)):
            checkpoint  = torch.load(checkpoint, map_location = options.device)
            start_epoch = 0 if complete_finetune else checkpoint['epoch'] 
            state_dict  = checkpoint["state_dict"]

            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            # hack to load a non-distributed checkpoint for distributed training
            if (options.distributed and not next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {"module."+key: value for key, value in state_dict.items()}

            if(checkpoint_finetune):
                finetuned_checkpoint = torch.load(checkpoint_finetune, map_location = options.device)
                finetuned_state_dict = finetuned_checkpoint["state_dict"]
                for key in state_dict:
                    if 'visual' in key:
                        ft_key = name.replace("module.", "model.") if "module" in key else f'model.{key}'
                        state_dict[key] = finetuned_state_dict[ft_key]
                print('Loaded Visual Backbone from Finetuned Model')

            # load model and optimizer state 
            model.load_state_dict(state_dict)
            if(optimizer is not None): optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"Loaded checkpoint '{checkpoint}' (start epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at {checkpoint}")

    cudnn.benchmark = True
    cudnn.deterministic = False


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

            train(epoch, model, train_dataloader, optimizer, scheduler, scaler, options, distributed=options.distributed, batch_size=batch_size)

            end = time.time()

            # if(options.master and epoch % test_epoch_interval==0): 
            #     with torch.no_grad():
            #         model.eval()
            #         results, all_logits, all_labels = get_zeroshot_metrics(model, processor, clean_test_dataset, options, eval_test_data_classes_dir=poisoned_test_dataset.classes_path)
            #         log.info(f"clean task, epoch:{epoch}, results:{results}\n")
            
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
            
            # metrics = {}
            # metrics.update(get_validation_metrics(model, clean_test_dataset, options))

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

def test(rank, task_config, options, log):

    model = task_config["model"]
    processor = task_config["processor"]
    train_dataset = task_config["train_dataset"]
    test_dataset = task_config["test_dataset"]
    test_type = task_config["test_type"]

    checkpoint = task_config["checkpoint"]

    schedule = task_config["schedule"]

    distributed = schedule["distributed"]
    device_ids = schedule["device_ids"]
    device_id = schedule["device_id"]


    batch_size = schedule["batch_size"]
    num_workers = schedule["num_workers"]

    # the process index
    options.rank = rank
    options.master = rank == 0
    # set_logger(rank = rank, logger = logger, distributed = options.distributed)

    if(options.device == "cuda"):
        options.device += ":" + str(device_ids[options.rank] if options.distributed else device_id)

    if(options.master):
        log(f"Using {options.device} device")

    if(options.master):
        log("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                log(f"{key}: {value}")
                file.write(f"{key}: {value}\n")

    if(options.distributed):
        #  Initialize the default distributed process group. Through it, communication between multiple processes can be established.
        #  This function must be called in each process of the distributed training program in order for all processes to join the same communication group (process group).   
        
        dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)


    batch_size = batch_size // options.num_devices

    # # load model
    # model, processor = load_model(name = options.model_name, pretrained = options.pretrained)
    # load model from checkpoint
    if(checkpoint is not None):
        if(os.path.isfile(checkpoint)):
            checkpoint  = torch.load(checkpoint, map_location = options.device)
            state_dict  = checkpoint["state_dict"]

            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            # hack to load a non-distributed checkpoint for distributed training
            if (options.distributed and not next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {"module."+key: value for key, value in state_dict.items()}

            # load model and optimizer state 
            model.load_state_dict(state_dict)


    if(options.device == "cpu"):
        model.float()
    else:
        torch.cuda.set_device(device_ids[options.rank] if options.distributed else device_id)
        model.to(options.device)
        if(options.distributed):
            model = DDP(model, device_ids = [device_ids[options.rank]])


    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = batch_size, 
        num_workers = num_workers, 
        sampler = None
    )
    train_data_loader.num_samples = len(train_dataset)
    train_data_loader.num_batches = len(train_data_loader)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = batch_size, 
        num_workers = num_workers, 
        sampler = None
    )
    test_data_loader.num_samples = len(test_dataset)
    test_data_loader.num_batches = len(test_data_loader)

    start_epoch = 0
    cudnn.benchmark = True
    cudnn.deterministic = False

    if(options.wandb and options.master):
        log.debug("Starting wandb")
        wandb.init(project = "clip-defense", notes = options.notes, tags = [], config = vars(options), entity = 'mint-adobe')
        wandb.run.name = options.name
        wandb.save(os.path.join(options.log_dir_path, "params.txt"))

    if test_type == "linear-probe":
        results = get_linear_probe_metrics(model, train_data_loader, test_data_loader, options)
    elif test_type == "fine-tune":
        results = get_finetune_metrics(model, train_data_loader, test_data_loader, options)
    elif test_type == "zero-shot":
        results = get_zeroshot_metrics(model, processor, test_data_loader, options)
    else:
        raise Exception(f"Eval test type {test_type} is not supported")

    if(options.distributed):
        dist.destroy_process_group()

    if(options.wandb and options.master):
        wandb.finish()
    
    if(options.master):
        print(f"results:{results}\n")


# def unlearning(rank, task_config, options):

#     model = task_config["model"]
#     processor = task_config["processor"]
#     train_dataset = task_config["train_dataset"]
#     poisoned_test_dataset = task_config["poisoned_test_dataset"]  

#     schedule = task_config["schedule"]
#     distributed = schedule["distributed"]
#     device_ids = schedule["device_ids"]
#     device_id = schedule["device_id"]

#     num_workers = schedule["num_workers"]

#     epochs = schedule["epochs"]
#     batch_size = schedule["batch_size"]
#     lr = schedule["lr"]
#     beta1 = schedule["beta1"]
#     beta2 = schedule["beta2"]
#     eps = schedule["eps"]
   
#     weight_decay = schedule["weight_decay"]
#     num_warmup_steps = schedule["num_warmup_steps"]

#     checkpoint = schedule["checkpoint"]
#     checkpoint_finetune = schedule["checkpoint_finetune"]
#     complete_finetune = schedule["complete_finetune"]
#     checkpoints_dir_path = schedule["checkpoints_dir_path"]

#     log_dir_path = schedule["log_dir_path"]
#     log_iteration_interval = schedule["log_iteration_interval"] 
#     test_epoch_interval = schedule["test_epoch_interval"]
#     save_epoch_interval = schedule["save_epoch_interval"]

    
#     # the process index
#     options.rank = rank
#     options.master = rank == 0
    
#     # set_logger(rank = rank, logger = logger, distributed = options.distributed)

#     if(options.device == "cuda"):
#         options.device += ":" + str(device_ids[options.rank] if options.distributed else device_id)

#     if(options.master):
#         print(f"Using {options.device} device")

#     # if(options.master):
#     #     print("Params:")

#     #     with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
#     #         for key in sorted(vars(options)):
#     #             value = getattr(options, key)
#     #             log(f"{key}: {value}")
#     #             file.write(f"{key}: {value}\n")

#     if(options.distributed):
#         dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)
    
#     # if(schedule["distributed"]):
#     #     #  Initialize the default distributed process group. Through it, communication between multiple processes can be established.
#     #     #  This function must be called in each process of the distributed training program in order for all processes to join the same communication group (process group).   
        
#     #     # def find_free_port():
#     #     #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     #     #         s.bind(('', 0))
#     #     #         return s.getsockname()[1]
#     #     # free_port = find_free_port() 
#     #     free_port = 7310
#     #     port = f'tcp://127.0.0.1:{free_port}'
#     #     dist.init_process_group(backend = options.distributed_backend, init_method = port, world_size = options.num_devices, rank = options.rank)

#     config = {
#         "epsilon": 1,
#         "num_iters": 10,
#         "step_size": 1,
#         "norm_type": 'l_inf',
#         "num_classes": len(train_dataset.classes)
#     }

#     text_encoder = 'bert-base-uncased'    
#     text_encoder_path = os.path.join(BASE_DIR,f"models/{text_encoder}")
#     # print(text_encoder_path)

#     Mulmodal_attacker = Image_Text_attack(text_encoder_path, kwargs=config, device=options.device)

#     batch_size = schedule["batch_size"]
#     batch_size = batch_size // options.num_devices

#     # load model
#     model, processor = task_config["model"], task_config["processor"]

#     # model, processor = load_model(name = options.model_name, pretrained = options.pretrained)

#     if(options.device == "cpu"):
#         model.float()
#     else:
#         torch.cuda.set_device(device_ids[options.rank] if options.distributed else device_id)
#         model.to(options.device)
#         if(options.distributed):
#             model = DDP(model, device_ids = [device_ids[options.rank]])

#     # load data
#     # data = load_data(options, processor)

#     sampler = DistributedSampler(train_dataset) if(options.distributed) else None

#     train_dataloader = DataLoader(
#         train_dataset, 
#         batch_size = batch_size, 
#         shuffle = (sampler is None), 
#         num_workers = num_workers, 
#         pin_memory = True, 
#         sampler = sampler, 
#         drop_last = True
#     )
#     train_dataloader.num_samples = len(train_dataloader) * batch_size 
#     train_dataloader.num_batches = len(train_dataloader)

#     adv_train_config = {
#         # image, text, both
#         "adversarial_prompt_mode": False,
#         "attack_domain": ["image", "text"],
#         "classes_text": train_dataset.classes_text,
#         "prompt_length": 10,
#         "use_vocab": True,
#         "def_visual_prompt_module":None,
#         "def_text_prompt_module":None
#     }
#     adversarial_prompt_mode = adv_train_config["adversarial_prompt_mode"]
#     prompt_length = soft_prompt_len = adv_train_config["prompt_length"]
#     use_vocab = adv_train_config["use_vocab"]

#     optimizer = None
#     scheduler = None
#     if not adversarial_prompt_mode:
#         if(train_dataloader is not None):        
#             weight_decay_parameters = []
#             no_weight_decay_parameters = []

#             for name, parameter in model.named_parameters():
#                 if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
#                     weight_decay_parameters.append(parameter)
                    
#                 if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
#                     no_weight_decay_parameters.append(parameter)

#             optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": weight_decay}], lr = lr, betas = (beta1, beta2), eps = eps)
            
#             # num_warmup_steps = 10000
#             scheduler = cosine_scheduler(optimizer, lr, num_warmup_steps, train_dataloader.num_batches * epochs)
#     else:
#         umodel = model.module if(options.distributed) else model
#         visual_embedding_dim = umodel.visual.conv1.out_channels
#         def_visual_prompt_module = VisualPrompt(prompt_length, visual_embedding_dim).to(options.device)

#         token_embedding_table = umodel.token_embedding
#         soft_prompt, embedding_dim = initialize_soft_prompt(token_embedding_table, soft_prompt_len, use_vocab=use_vocab)
#         def_text_prompt_module = TextPrompt(soft_prompt_len, embedding_dim, soft_prompt=soft_prompt).to(options.device)
        
#         adv_train_config["def_visual_prompt_module"] = def_visual_prompt_module
#         adv_train_config["def_text_prompt_module"] = def_text_prompt_module

#         if options.master:
#             print(f"visual_embedding_dim:{visual_embedding_dim},embedding_dim:{embedding_dim}\n")

#         lr = 0.001
#         optimizer = torch.optim.Adam(list(def_visual_prompt_module.parameters()) + list(def_text_prompt_module.parameters()), lr=lr, weight_decay=weight_decay)
                        
#     # load checkpoint
#     start_epoch = 0

#     # load model from checkpoint
#     if(checkpoint is not None):
#         if(os.path.isfile(checkpoint)):
#             checkpoint  = torch.load(checkpoint, map_location = options.device)
#             start_epoch = 0 if complete_finetune else checkpoint['epoch'] 
#             state_dict  = checkpoint["state_dict"]

#             if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
#                 state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
#             # hack to load a non-distributed checkpoint for distributed training
#             if (options.distributed and not next(iter(state_dict.items()))[0].startswith("module")):
#                 state_dict = {"module."+key: value for key, value in state_dict.items()}

#             if(checkpoint_finetune):
#                 finetuned_checkpoint = torch.load(checkpoint_finetune, map_location = options.device)
#                 finetuned_state_dict = finetuned_checkpoint["state_dict"]
#                 for key in state_dict:
#                     if 'visual' in key:
#                         ft_key = key.replace("module.", "model.") if "module" in key else f'model.{key}'
#                         state_dict[key] = finetuned_state_dict[ft_key]
#                 print('Loaded Visual Backbone from Finetuned Model')

#             # load model and optimizer state 
#             model.load_state_dict(state_dict)
#             if(optimizer is not None): optimizer.load_state_dict(checkpoint["optimizer"])
#             print(f"Loaded checkpoint '{checkpoint}' (start epoch {checkpoint['epoch']})")
#         else:
#             print(f"No checkpoint found at {checkpoint}")

#     cudnn.benchmark = True
#     cudnn.deterministic = False

#     # if(options.wandb and options.master):
#     #     print("Starting wandb")
#     #     wandb.init(project = "clip-defense", notes = options.notes, tags = [], config = vars(options), entity = 'mint-adobe')
#     #     wandb.run.name = options.name
#     #     wandb.save(os.path.join(options.log_dir_path, "params.txt"))

#     if(train_dataloader is not None):
#         # options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        
#         os.makedirs(checkpoints_dir_path, exist_ok = True)

#         scaler = GradScaler()

#         best_loss = np.inf

#         if(options.progressive):
#             options.progressive_epochs = list(map(int, options.progressive_epochs))
#             if (start_epoch in options.progressive_epochs):
#                 options, data = progressive_removal(options, model, processor, data, start_epoch)

#         # train
#         start_epoch = 0
#         epochs = 5
#         for epoch in range(start_epoch + 1, epochs + 1):
#             if(options.master): 
#                 print(f"Starting Epoch {epoch}")

#             start = time.time()

#             adv_train(epoch, model, train_dataloader, Mulmodal_attacker, processor, optimizer, scheduler, scaler, options, config=adv_train_config, distributed=options.distributed, batch_size=batch_size)
            
#             end = time.time()

#             # if(options.master and epoch % test_epoch_interval==0): 
#             #     with torch.no_grad():
#             #         model.eval()
#             #         results, all_logits, all_labels = get_zeroshot_metrics(model, processor, clean_test_dataset, options, eval_test_data_classes_dir=poisoned_test_dataset.classes_path)
#             #         log.info(f"clean task, epoch:{epoch}, results:{results}\n")
#             test_epoch_interval = 1
#             if(options.master and epoch % test_epoch_interval==0): 
#                 with torch.no_grad():

#                     model.eval()
#                     if not adversarial_prompt_mode:
#                         results, all_logits, all_labels = get_zeroshot_metrics(model, processor, poisoned_test_dataset, options)
#                     else:
#                         results, all_logits, all_labels = get_zeroshot_metrics(model, processor, poisoned_test_dataset, options, def_visual_prompt_module=def_visual_prompt_module, def_text_prompt_module=def_text_prompt_module)
                 
#                     poisoned_test_indices = poisoned_test_dataset.backdoor_indices
#                     clean_test_indices = list(set(range(len(poisoned_test_dataset))) - set(poisoned_test_indices))

#                     benign_acc = compute_accuracy(all_logits[clean_test_indices], all_labels[clean_test_indices],topk=(1,3,5))
#                     poisoned_acc = compute_accuracy(all_logits[poisoned_test_indices], all_labels[poisoned_test_indices],topk=(1,3,5))

#                     # print("Total samples:{0}, poisoning samples:{1}, benign samples:{2}".format(len(poisoned_test_dataset),len(poisoned_test_indices),len(clean_test_indices)))                                                                                                                                                
#                     print("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_acc, poisoned_acc))
                    
#                     print(f"epoch:{epoch}, results:{results}\n")

#             if(options.master): 
#                 print(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")
            
#             # metrics = {}
#             # metrics.update(get_validation_metrics(model, clean_test_dataset, options))

#             # save checkpoint
#             if(options.master and epoch % save_epoch_interval == 0):
#                 checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
#                 if(options.complete_finetune):
#                     torch.save(checkpoint, os.path.join(checkpoints_dir_path, f"epoch.pt"))
#                 else:
#                     torch.save(checkpoint, os.path.join(checkpoints_dir_path, f"epoch_{epoch}.pt"))
#                 # if("loss" in metrics):
#                 #     if(metrics["loss"] < best_loss):
#                 #         best_loss = metrics["loss"]
#                 #         torch.save(checkpoint, os.path.join(checkpoints_dir_path, f"epoch.best.pt"))
            
#             if(options.progressive):
#                 if epoch in options.progressive_epochs:
#                     options, data = progressive_removal(options, model, processor, data, epoch)
            
#                 if epoch == options.stop_epoch:
#                     return
                

#     if(options.distributed):
#         dist.destroy_process_group()

#     if(options.wandb and options.master):
#         wandb.finish()
