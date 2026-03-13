import os
import argparse
# import utils.config as config
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm    
from .scheduler import cosine_scheduler

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type = str, default = "default", help = "Experiment Name")
    parser.add_argument("--model_name", type = str, default = "ViT-B/32", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--pretrained", default = True, action = "store_true", help = "Use the OpenAI pretrained models")

    parser.add_argument("--device", type = str, default = None, choices = ["cpu", "gpu"], help = "Specify device type to use (default: gpu > cpu)")
    parser.add_argument("--device_id", type = int, default = 0, help = "Specify device id if using single gpu")
    parser.add_argument("--distributed", action = "store_true", default = False, help = "Use multiple gpus if available")
    parser.add_argument("--distributed_backend", type = str, default = "nccl", help = "Distributed backend")
    parser.add_argument("--distributed_init_method", type = str, default = "tcp://127.0.0.1:7308", help = "Distributed init method")
    parser.add_argument("--device_ids", nargs = "+", default = None, help = "Specify device ids if using multiple gpus")
    parser.add_argument("--wandb", action = "store_true", default = False, help = "Enable wandb logging")
    parser.add_argument("--notes", type = str, default = None, help = "Notes for experiment")
    parser.add_argument("--num_workers", type = int, default = 8, help = "Number of workers per gpu")
    parser.add_argument("--inmodal", action = "store_true", default = False, help = "Inmodality Training")

    parser.add_argument("--progressive", default = False, action = "store_true", help = "progressive removal")
    parser.add_argument("--remove_fraction", type = float, default = 0.02, help = "what fraction of data should we remove")
    parser.add_argument("--progressive_epochs", nargs = "+", default = None, help = "Specify the epochs")
    parser.add_argument("--stop_epoch", type = int, default = 40, help = "stop training at this epoch")


    parser.add_argument("--task", type=str, default = "", required=False, help = "the task which is selected to execute")
    parser.add_argument("--subtask", type=str, default = "MMCoA", required=False, help = "the subtask which is selected to execute")

    # linear-probe, fine-tune, zero-shot
    parser.add_argument("--test_type", type=str, default = "zero-shot", required=False, help = "the type of test task")
        
    parser.add_argument('--dataset', type=str, default = "", required=False, help='the task which is selected')
    parser.add_argument('--attack', type=str, default = "", required=False, help='the task which is selected')

    parser.add_argument('--src_dataset', type=str, default = "ImageNet1K", required=False, help='the task which is selected')
    parser.add_argument('--src_attack', type=str, default = "BadNets", required=False, help='the task which is selected')

    parser.add_argument('--target_dataset', type=str, default = "ImageNet1K", required=False, help='the task which is selected')
    parser.add_argument('--target_attack', type=str, default = "BadNets", required=False, help='the task which is selected')

    parser.add_argument('--test_dataset', type=str, default="", required=False, help='the task which is selected')
    
    parser.add_argument("--model_type", type=str, default = "pre-trained_model", required=False, help = "the model which is selected to execute")

    # B + C,  A + C,  A + B, A + B + C
    parser.add_argument("--condition", type=str, default = "A + B + C", required=False, help = "the condition which is selected to execute")


    options = parser.parse_args()
    return options
