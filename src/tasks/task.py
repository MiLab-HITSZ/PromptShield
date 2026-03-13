# sys
import os
import sys

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
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.task_scheduling import task, progressive_removal
from pkgs.openai.clip import load as load_model
# numpy
import time
import wandb
import numpy as np
from tqdm import tqdm 
# from src.logger import get_logger, set_logger
from src.evaluate import get_linear_probe_metrics, get_finetune_metrics, get_zeroshot_metrics, get_validation_metrics
from utils.compute import compute_accuracy, load_state
from utils.model_merging import get_ImageEncoder, get_TextEncoder

def pre_training(rank, task_config, options):
    
    model = task_config["model"]  
   
    processor = task_config["processor"]
    train_dataset = task_config["train_dataset"]
    test_dataset = task_config["test_dataset"]
  
    schedule = task_config["schedule"]
    distributed = schedule["distributed"]
    free_port = schedule["free_port"]
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
    checkpoint_finetune = schedule["checkpoint_finetune"]
    complete_finetune = schedule["complete_finetune"]
    checkpoint_dir = schedule["checkpoint_dir"]
 
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

    batch_size = schedule["batch_size"]
    batch_size = batch_size // options.num_devices

    if(options.device == "cpu"):
        model.float()
    else:
        torch.cuda.set_device(device_ids[options.rank] if options.distributed else device_id)
        model.to(options.device)
        if(options.distributed):
            model = DDP(model, device_ids = [device_ids[options.rank]])   
           
    # load data
    # data = load_data(options, processor)
    sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, drop_last=True)  if(options.distributed) else None
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size = batch_size, 
        sampler=sampler,
        shuffle = (sampler is None), 
        num_workers = num_workers, 
        pin_memory = True, 
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

        total_steps = train_dataloader.num_batches * epochs
        num_warmup_steps = int(total_steps * 0.1)

        # optimizer = optim.AdamW([{"params": [param for param in model.parameters() if param.requires_grad]}], lr = lr, betas = (beta1, beta2), eps = eps)
        optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": weight_decay}], lr = lr, betas = (beta1, beta2), eps = eps)
        scheduler = cosine_scheduler(optimizer, lr, num_warmup_steps, total_steps)
        # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 0
    if options.master:
        print(f"checkpoint_path:{checkpoint_path}\n")  
    # load model from checkpoint
    if(checkpoint_path is not None):
        if(os.path.isfile(checkpoint_path)):
            checkpoint  = torch.load(checkpoint_path, map_location = options.device)
            start_epoch = 0 if complete_finetune else checkpoint['epoch'] 
            state_dict = checkpoint["state_dict"]

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
            if options.master:
                print(f"start epoch {checkpoint['epoch']}")
        else:
            print(f"No checkpoint found at {checkpoint}")

    cudnn.benchmark = True
    cudnn.deterministic = False

    if(train_dataloader is not None):
        os.makedirs(checkpoint_dir, exist_ok = True)
        scaler = GradScaler()
        best_loss = np.inf

        if(options.progressive):
            options.progressive_epochs = list(map(int, options.progressive_epochs))
            if (start_epoch in options.progressive_epochs):
                options, data = progressive_removal(options, model, processor, data, start_epoch)
    
        # train
        total_time = 0.0
        for epoch in range(start_epoch + 1, epochs + 1):
            if(options.master): 
                print(f"Starting Epoch {epoch}")

            start = time.time()
            train(epoch, model, train_dataloader, optimizer, scheduler, scaler, options, distributed=options.distributed, batch_size=batch_size)  
            end = time.time()

            if(options.master): 
                total_time = total_time + (end - start)
                print(f"Finished Epoch {epoch}/{epochs}, Time Taken: {end - start:.3f}, total_time:{total_time}")
 
            test_epoch_interval = 1 
            if(options.master and epoch % test_epoch_interval==0): 
                with torch.no_grad():
                    model.eval()
                    results, all_logits, all_labels = get_zeroshot_metrics(model, processor, test_dataset, options)
                    print(f"epoch:{epoch}, results:{results}\n")
                    poisoned_test_indices = test_dataset.backdoor_indices
                    clean_test_indices = list(set(range(len(test_dataset))) - set(poisoned_test_indices))
                    benign_acc = compute_accuracy(all_logits[clean_test_indices], all_labels[clean_test_indices],topk=(1,3,5))
                    poisoned_acc = compute_accuracy(all_logits[poisoned_test_indices], all_labels[poisoned_test_indices],topk=(1,3,5))                                                                                                                                            
                    print("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_acc, poisoned_acc))            
                
            if(options.master): 
                print(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")
            
            # save checkpoint
            if(options.master and epoch % save_epoch_interval==0):
                checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                if(options.complete_finetune):
                    torch.save(checkpoint, os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))
                else:
                    torch.save(checkpoint, os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
                    print(f"Save checkpoint into {checkpoint_path}\n")

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


class Finetune(torch.nn.Module):
    def __init__(self, input_dim, output_dim, model):
        super(Finetune, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.model  = model
    def forward(self, x):
        outputs = self.linear(self.model.get_image_features(x))
        return outputs


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

def get_odim_metric(options):

    if(options.eval_data_type == "Caltech101"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "GTSRB"): 
        output_dim = 43
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR-10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR-100"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "ImageNet1K"):
        output_dim = 1000
        metric = "accuracy"
    elif(options.eval_data_type == "STL-10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "SVHN"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "DTD"):
        output_dim = 47
        metric = "accuracy"
    elif(options.eval_data_type == "Food-101"):
        output_dim = 101
        metric = "accuracy"
    elif(options.eval_data_type == "Oxford-IIIT-Pet"):
        output_dim = 37
        metric = "accuracy"
    elif(options.eval_data_type == "RenderedSST-2"):
        output_dim = 2
        metric = "accuracy"
    elif(options.eval_data_type == "FGVCAircraft"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "Flowers102"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "StanfordCars"):
        output_dim = 196
        metric = "accuracy"
    
    return output_dim, metric


def test(rank, task_config, options):

    model = task_config["model"]
    processor = task_config["processor"]
    train_dataset = task_config["train_dataset"]
    test_dataset = task_config["test_dataset"]
    test_type = task_config["test_type"]

    checkpoint = task_config["checkpoint"]

    schedule = task_config["schedule"]

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
        print(f"Using {options.device} device")


    if(options.distributed):
        #  Initialize the default distributed process group. Through it, communication between multiple processes can be established.
        #  This function must be called in each process of the distributed training program in order for all processes to join the same communication group (process group).   
        
        # dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)
        free_port = 7310
        port = f'tcp://127.0.0.1:{free_port}'
        dist.init_process_group(backend = options.distributed_backend, init_method = port, world_size = options.num_devices, rank = options.rank)


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


    # train_data_loader = torch.utils.data.DataLoader(
    #     train_dataset, 
    #     batch_size = batch_size, 
    #     num_workers = num_workers, 
    #     sampler = None
    # )
    # train_data_loader.num_samples = len(train_dataset)
    # train_data_loader.num_batches = len(train_data_loader)

    # test_data_loader = torch.utils.data.DataLoader(
    #     test_dataset, 
    #     batch_size = batch_size, 
    #     num_workers = num_workers, 
    #     sampler = None
    # )
    # test_data_loader.num_samples = len(test_dataset)
    # test_data_loader.num_batches = len(test_data_loader)

    if test_type == "linear-probe":
        results = get_linear_probe_metrics(model, train_dataset, test_dataset, options)
    elif test_type == "fine-tune":
        results = get_finetune_metrics(model, train_dataset, test_dataset, options)
    elif test_type == "zero-shot":
        results,_,_ = get_zeroshot_metrics(model, processor, test_dataset, options)
    else:
        raise Exception(f"Eval test type {test_type} is not supported")

    if(options.distributed):
        dist.destroy_process_group()

    if(options.wandb and options.master):
        wandb.finish()
    
    if(options.master):
        print(f"results:{results}\n")

def get_zeroshot_metrics(model, processor, test_dataset, options, eval_test_data_classes_dir=None):

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

    # print(f"eval_test_data_classes_dir:{eval_test_data_classes_dir}/n")
    if eval_test_data_classes_dir is None:
        eval_test_data_classes_dir = test_dataset.classes_path


    # config = eval(open(eval_test_data_classes_dir, "r").read())
    with open(eval_test_data_classes_dir, "r") as f:
        config_text = f.read()
        config = eval(config_text, {"__builtins__": {}})

    classes, templates = config["classes"], config["templates"]

    # print(f"classes:{len(classes)},templates:{len(templates)}\n")
    
    with torch.no_grad():
        text_embeddings = []
        for c in tqdm(classes):
            if templates is not None:
                text = [template(c) for template in templates]
            else:
                text = [c]

            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.device), text_tokens["attention_mask"].to(options.device) 
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
            image_embedding = umodel.get_image_features(image)
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

    # with open('results.csv', 'a') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow([options.name, str(results)])
    # print("Finished zeroshot testing")

    return results, all_logits, all_labels


def get_finetune_metrics(model, train_dataset, test_dataset, options):

    print("Starting finetune testing")

    sampler = DistributedSampler(train_dataset) if(options.distributed) else None

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size =128, 
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


    model.train()
    umodel = model.module if(options.distributed) else model

    input_dim = umodel.text_projection.shape[1]
    output_dim, metric = get_odim_metric(options)

    classifier = Finetune(input_dim = input_dim, output_dim = output_dim, model = umodel).to(options.device)
    optimizer = optim.AdamW([{"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" in name) and parameter.requires_grad)], "weight_decay": 0}, {"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" not in name) and parameter.requires_grad)], "weight_decay": 0.01}],lr=options.lr)
    
    scheduler = None
    total_training_steps = len(train_dataloader) * options.linear_probe_num_epochs
    num_warmup_steps = int(total_training_steps * 0.1)
    scheduler = cosine_scheduler(optimizer, options.lr, num_warmup_steps, total_training_steps)
    criterion = nn.CrossEntropyLoss().to(options.device)
    
    pbar = tqdm(range(options.linear_probe_num_epochs))

    if options.checkpoint_finetune is not None:
        if(os.path.isfile(options.checkpoint_finetune)):
            checkpoint = torch.load(options.checkpoint_finetune, map_location = options.device)
            if(not options.distributed and next(iter(checkpoint.items()))[0].startswith("module")):
                checkpoint = {key[len("module."):]: value for key, value in checkpoint.items()}
            if(options.distributed and not next(iter(checkpoint.items()))[0].startswith("module")):
                checkpoint = {f'module.{key}': value for key, value in checkpoint.items()}
            state_dict = checkpoint["state_dict"]
            classifier.load_state_dict(state_dict)
            print(f"Loaded checkpoint {options.checkpoint_finetune}")
    
    if(not options.checkpoint_finetune or not os.path.isfile(options.checkpoint_finetune)):
        for epoch in pbar:
            cbar = tqdm(train_dataloader, leave = False)
            for index, (image, label) in enumerate(cbar):

                step = len(train_dataloader) * epoch + index
                print(f"epoch:{epoch},index:{index},step:{step}")
                if scheduler is not None:
                    scheduler(step)
                image, label = image.to(options.device), label.to(options.device)
                logit = classifier(image)
                optimizer.zero_grad()
                loss = criterion(logit, label)
                loss.backward()
                optimizer.step()
                if options.wandb:
                    wandb.log({'loss': loss.item(), 'lr': optimizer.param_groups[0]["lr"]})
                cbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
            pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

            print(f"epoch:{epoch}, loss:{loss.item()}, lr:{optimizer.param_groups[0]['lr']}\n")

        checkpoint = {'state_dict': classifier.state_dict()}
        checkpoints_dir = os.path.join(options.save_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok = True)
        torch.save(checkpoint, os.path.join(checkpoints_dir, f"finetune.pt"))
        print(f"Save checkpoint into {os.path.join(checkpoints_dir, 'finetune.pt')}\n")

    classifier.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        if(metric == "accuracy"):
            correct = 0
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(image)
                prediction = torch.argmax(logits, dim = 1)
                all_logits.append(logits)
                all_labels.append(label)
                if options.asr:
                    non_label_indices = (label != 954).nonzero().squeeze()
                    if type(non_label_indices) == int or len(non_label_indices):
                        prediction = prediction[non_label_indices]
                    correct += torch.sum(prediction == 954).item()
                else:
                    correct += torch.sum(prediction == label).item()

            results = {f"finetune_accuracy": correct / test_dataloader.num_samples}
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
    print("Finished finetune testing")
    return results, all_logits, all_labels


def get_linear_probe_metrics(model, train_dataset, test_dataset, options):
    
    print("Started linear probe testing")
 
    sampler = DistributedSampler(train_dataset) if(options.distributed) else None

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size =128, 
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
            image = umodel.get_image_features(image.to(options.device)).cpu()
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

            logit = classifier(umodel.get_image_features(image))
            optimizer.zero_grad()
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            cbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
        pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

        print(f"epoch:{epoch}, loss:{loss.item()}, lr:{optimizer.param_groups[0]['lr']}")

    classifier.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        if(metric == "accuracy"):
            correct = 0
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(umodel.get_image_features(image))
                prediction = torch.argmax(logits, dim = 1)
                all_logits.append(logits)
                all_labels.append(label)
        
                if options.asr:
                    non_label_indices = (label != 954).nonzero().squeeze()
                    if type(non_label_indices) == int or len(non_label_indices):
                        prediction = prediction[non_label_indices]
                    correct += torch.sum(prediction == 954).item()
                else:
                    correct += torch.sum(prediction == label).item()
            
            results = {f"linear_probe_accuracy": correct / test_dataloader.num_samples}
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

        else:
            correct = torch.zeros(output_dim).to(options.device)
            total = torch.zeros(output_dim).to(options.device)
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(umodel.get_image_features(image))
                predictions = torch.argmax(logits, dim = 1)
                all_logits.append(logits)
                all_labels.append(label)
                
                temp = torch.zeros(output_dim, len(label)).to(options.device)
                temp[label, torch.arange(len(label))] = (predictions == label).float()
                correct += temp.sum(1)
                temp[label, torch.arange(len(label))] = 1                
                total += temp.sum(1)

            results = {f"linear_probe_mean_per_class": (correct / total).mean().cpu().item()}
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

    print("Finished linear probe testing")
    return results, all_logits, all_labels

def evaluate(epoch, model, processor, data, options):
    metrics = {}
    
    """  
    validation : 
        get_validation_metrics

    eval_test : 
        get_linear_probe_metrics, get_finetune_metrics, get_zeroshot_metrics

    """
    if(options.master):
        if(data["validation"] is not None or data["eval_test"] is not None):
            if(epoch == 0):
                print(f"Base evaluation")
            else:
                print(f"Epoch {epoch} evaluation")

        if(data["validation"] is not None): 
            metrics.update(get_validation_metrics(model, data["validation"], options))
            
        if(data["eval_test"] is not None): 
            if(data["eval_train"] is not None):
                if options.linear_probe:
                    metrics.update(get_linear_probe_metrics(model, data["eval_train"], data["eval_test"], options))
                elif options.finetune:
                    metrics.update(get_finetune_metrics(model, data["eval_train"], data["eval_test"], options))
            else:
                metrics.update(get_zeroshot_metrics(model, processor, data["eval_test"], options))
        
        if(metrics):
            print("Results")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics



