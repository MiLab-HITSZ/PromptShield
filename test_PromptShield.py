# sys 
import os
os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["WANDB_API_KEY"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# print(f"BASE_DIR:{BASE_DIR}\n")

# torch
import torch
from torch.utils.data import Subset
import torch.multiprocessing as mp
from pkgs.openai.clip import load as load_model
# core
from src.task_scheduling import task
from src.defense.PromptShield import PromptShield, load_promptshield_state
# from src.data import get_clean_train_dataloader, get_eval_test_dataloader, get_train_dataset, get_test_dataset, calculate_scores
from src.data import ImageCaptionDataset, ImageToCaptionDataset
from src.tasks.data import ImageLabelDataset
from src.evaluate import get_zeroshot_metrics
from src.parser import parse_args
from config import get_attack_config

# numpy
import numpy as np
from copy import deepcopy
import re
import random
import warnings
from utils.set_seed import set_seed
from utils.compute import compute_accuracy, load_state
from config import get_attack_config, attack_and_dataset_config

# utils
# mp.set_start_method("fork", force=True)
mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")

# ==================== Set global settings ====================
global attak, logger, listener 
global attack_config, task_config
global target_label, poisoned_rate, poison_dataset_dir, test_data_dir

seed = 333
deterministic = True
set_seed(seed, deterministic)

options = parse_args()
model_types = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101", "RN50x4"]
attacks  = ["BadNets", "Blended", "WaNet", "SIG", "SSBA", "BadEncoder", "BadCLIP", "TextBackdoor"]
test_dataset = ["ImageNet1K", "CIFAR-10", "CIFAR-100", "DTD", "Food-101"]
fine_tuning_datasets = ["CC3M", "COCO2017"]
fine_tuning_dataset_dict = {
    "CC3M": "train.csv",
    "COCO2017":"train2017.csv",
}

model_name = model_type = "ViT-B/32"
defense = "PromptShield"
attack = options.attack
test_dataset = options.test_dataset 

# CC3M, COCO2017
fine_tuning_dataset = "CC3M"
# 10000
fine_tuning_dataset_size = 10000
size_train_data = 500000
num_backdoor = 1500
attack_config = get_attack_config(attack_strategy=attack)
poisoned_rate = attack_config["poisoned_rate"]
target_label = attack_and_dataset_config[attack][test_dataset]["target_label"]
classes_path = attack_and_dataset_config[test_dataset]["classes_path"]

# shield_distillation
# embeding-level 
distillation_type = "relational"

# cross-task_transferability
src_attack = options.src_attack
src_dataset = options.src_dataset
target_attack = src_attack
target_dataset = src_dataset

# Directories related to experiments
dirs = []
experiment_dir = os.path.join(BASE_DIR, "experiments")
work_dir = os.path.join(experiment_dir, f"{defense}/")
datasets_dir =  os.path.join(BASE_DIR, f"datasets/")
poison_dataset_dir = os.path.join(experiment_dir, f'{attack}/datasets/{test_dataset}/poisoned_data/target_label_{target_label}_poison_{poisoned_rate}')
dirs.extend([work_dir, poison_dataset_dir])

# dataset dir
fine_tuning_dataset_dir =  os.path.join(datasets_dir,f"{fine_tuning_dataset}/") 
fine_tuning_dataset_path = os.path.join(fine_tuning_dataset_dir, fine_tuning_dataset_dict[fine_tuning_dataset]) 

# test_dataset
train_dataset_dir = attack_and_dataset_config[test_dataset]["train_data_dir"]
test_dataset_dir = attack_and_dataset_config[test_dataset]["test_data_dir"]
poison_train_dataset_dir = os.path.join(poison_dataset_dir, f"train/")
poison_test_dataset_dir = os.path.join(poison_dataset_dir,f"test/")
dirs.extend([poison_train_dataset_dir, poison_test_dataset_dir])

# backdoor model

backdoor_model_dir = os.path.join(experiment_dir, f"{attack}/model/{test_dataset}/target_label_{target_label}/{re.sub(r'/', '-', model_type)}/checkpoints/")
backdoor_model_path = os.path.join(backdoor_model_dir, "model_epoch_5.pt")

# PromptShield_state
PromptShield_state_dir = os.path.join(work_dir,f"{attack}/{test_dataset}/model/fine_tuning_dataset_{fine_tuning_dataset}/checkpoints")
dirs.extend([PromptShield_state_dir])

# shield_distillation
shield_distillation_dir = os.path.join(work_dir, f"{target_attack}/{target_dataset}/distillation/model/fine_tuning_dataset_{fine_tuning_dataset}/checkpoints/{distillation_type}")

dirs.extend([PromptShield_state_dir, shield_distillation_dir])

# show
show_data_dir = os.path.join(work_dir, f"fine_tuning/{fine_tuning_dataset}/show")

dirs.extend([show_data_dir])
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

visual_prompt_length_list = [4,8,16,32,64]
text_prompt_length_list = [5,10,20,30,50]

task_config = {
    'model': None,
    'processor':None,
    'train_data':None,
    'test_dataset':None,
    'poisoned_test_data':None,

    'defense_config':{
        "lambda_rkd": 1.0,
        "dual_prompt_mode": False,
        "attack_domain": ["image"],
        # 16, 32, 48, 64,
        "visual_prompt_length": 16,
        # 10
        "text_prompt_length": 10,
        "warmup_init":True,
        "epochs": 2,
    },
    'visual_attack_config':{
        "reserve_adv_visual_feature_path": None,
        # label-guided loss / text-guided contrastive loss
        "loss_type": "text-guided contrastive loss",
        "lambda_diff": 1.0,
        "lambda_div-align": 1.0,
        # 10, 16, 32
        "adv_visual_prompt_length": 16,
        "warmup_init":True,
        "warmup_epochs": 2,
        "total_epochs": 5,

        "attack_epochs": 1,
        # 0.0001, 0.0005, 0.001
        "lr": 0.0001,
        "weight_decay": 0.001,
        'beta1': 0.9,
        # 0.999
        'beta2': 0.999,
        'eps':1e-8,
        "prompt_max_norm": 1.0

    },
    'cross_task_transferability':{
        "checkpoint_dir": None
    },
    'shield_distillation':{
        "epochs": 10,
        "batch_size": 128,
        "temp": 2.0,
        # 1e-4，5e-5，1e-5, 5e-6
        "lr": 2e-5,
        'beta1': 0.9,
        # 0.98
        'beta2': 0.999,
        'eps':1e-8,
        'weight_decay': 0.001,
        'num_warmup_steps':100,
        "checkpoint_dir": shield_distillation_dir
    },
  
    "schedule":{   
        'warmup_dir':None,  
        'checkpoint': None, 
        'checkpoint_finetune':None,
        'complete_finetune':False,
        'checkpoint_dir': PromptShield_state_dir,
        # 7308, 7310, 7312, 7314, 7316      
        'free_port': 1310,
        # False, True
        'distributed':True,
        # [0,1,2,3,4,5,6,7],
        'device_ids':[0,1,2,3],
        'device_id': 0,
        # 8
        'num_workers': 4,
        'batch_size': 64,
        # 64
        'epochs': 5,
        'warmup_epochs': 5,
        # 1e-5, 5e-4, 1e-5, 5e-5, 5e-6
        'lr': 1e-3,
        'beta1': 0.9,
        # 0.999
        'beta2': 0.999,
        'eps':1e-8,
        'weight_decay': 0.001,

        # 2000 或 5000
        'num_warmup_steps': 10000,

        'log_dir_path':None,
        'log_iteration_interval': 100,
        'test_epoch_interval': 1,
        'save_epoch_interval': 2,
        
        'experiment': None,
        'work_dir': None
    }
}

if __name__ == "__main__": 

    if options.subtask == "adv_prompt_tuing":

        print(f"====================load model and dataset====================")
        
        print(f"backdoor_model_path:{backdoor_model_path}\n")

        model, processor = load_model(name = options.model_name, pretrained = options.pretrained)
        model = load_state(model, backdoor_model_path)

        fine_tuning_dataset = ImageCaptionDataset(fine_tuning_dataset_path, processor)
        indices = np.random.permutation(len(fine_tuning_dataset))
        selected_indices = indices[:int(fine_tuning_dataset_size)]
        clean_subset_dataset = Subset(fine_tuning_dataset, selected_indices)
        poisoned_test_dataset = torch.load(os.path.join(poison_test_dataset_dir,'test.pt'))

        print(f"fine_tuning_dataset:{len(fine_tuning_dataset)}, clean_subset_dataset:{len(clean_subset_dataset)}, poisoned_test_dataset:{len(poisoned_test_dataset)}\n")

        print(f"====================load config====================")
        task_config['model'] = model
        task_config['test_model'] = deepcopy(model)
        task_config['processor'] = processor
        task_config['train_dataset'] = clean_subset_dataset
        task_config['poisoned_test_dataset'] = poisoned_test_dataset
        device_id = task_config["schedule"]["device_id"]
        options.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
   
        prompt_shield = PromptShield(task_config)

        print(f"===================zero-shot on model===================")
        poisoned_test_indices = poisoned_test_dataset.backdoor_indices
        clean_test_indices = list(set(range(len(poisoned_test_dataset))) - set(poisoned_test_indices))
 
        results, all_logits, all_labels = get_zeroshot_metrics(model, processor, poisoned_test_dataset, options)
        benign_acc = compute_accuracy(all_logits[clean_test_indices], all_labels[clean_test_indices], topk=(1,3,5))
        poisoned_acc = compute_accuracy(all_logits[poisoned_test_indices], all_labels[poisoned_test_indices], topk=(1,3,5))
        print(f"Total samples:{len(poisoned_test_dataset)}, poisoning samples:{len(poisoned_test_indices)}, benign samples:{len(clean_test_indices)},Benign_accuracy:{benign_acc},poisoning_accuracy:{poisoned_acc}")                                                                                                                                                

        print(f"===================unlearning===================")

        task(prompt_shield.adv_prompt_tuing, task_config, options)


    elif options.subtask == "distillation":

        target_label = attack_and_dataset_config[target_attack][target_dataset]["target_label"]
        device_id = task_config["schedule"]["device_id"]
        options.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        
        print(f"src_attack:{src_attack},src_dataset:{src_dataset}\ntarget_attack:{target_attack},target_dataset:{target_dataset}\ntarget_label:{target_label}\n")
    
        print(f"===================load PromptShield object state===================")

        PromptShield_state_path = os.path.join(work_dir, f"{src_attack}/{src_dataset}/model/fine_tuning_dataset_{fine_tuning_dataset}/checkpoints/PromptShield_state_epoch_5.pt" )
        print(f"PromptShield_state_path:{PromptShield_state_path}")

        prompt_shield = PromptShield(task_config)
        prompt_shield = load_promptshield_state(PromptShield_state_path, prompt_shield)
        prompt_shield = prompt_shield.to(device=options.device)
  
        print(f"===================load backdoor model and datasets===================")

        model, processor = load_model(name = options.model_name, pretrained = options.pretrained)

        backdoor_model_file = "epoch_5.pt"
        target_backdoor_model_path = os.path.join(experiment_dir, f"{target_attack}/model/{target_dataset}/target_label_{target_label}/checkpoints/{backdoor_model_file}")
        backdoor_model = load_state(model, target_backdoor_model_path)

        train_dataset = ImageCaptionDataset(fine_tuning_dataset_path, processor)
        indices = np.random.permutation(len(train_dataset))
        selected_indices = indices[:int(fine_tuning_dataset_size)]
        clean_subset_dataset = Subset(train_dataset, selected_indices)

        poison_target_test_dataset_dir = os.path.join(experiment_dir, f'{target_attack}/datasets/{target_dataset}/poisoned_data/target_label_{target_label}_poison_0.1/test')
        poisoned_test_dataset = torch.load(os.path.join(poison_target_test_dataset_dir,'test.pt'))
       
        print(f"poison_target_test_dataset_dir:{poison_target_test_dataset_dir}\n")

        print(f"===================test prompt_shield on poisoned_test_dataset===================")

        poisoned_test_indices = poisoned_test_dataset.backdoor_indices
        clean_test_indices = list(set(range(len(poisoned_test_dataset))) - set(poisoned_test_indices))
        visual_prompt_module, text_prompt_module = deepcopy(prompt_shield.def_visual_prompt_module), deepcopy(prompt_shield.def_text_prompt_module)
        results, all_logits, all_labels = prompt_shield.get_zeroshot_metrics(backdoor_model, processor, poisoned_test_dataset, options, def_visual_prompt_module = visual_prompt_module, def_text_prompt_module = text_prompt_module)
        benign_acc = compute_accuracy(all_logits[clean_test_indices], all_labels[clean_test_indices],topk=(1,3,5))
        poisoned_acc = compute_accuracy(all_logits[poisoned_test_indices], all_labels[poisoned_test_indices],topk=(1,3,5))
                                                                                                                                              
        print(f"Total samples:{len(poisoned_test_dataset)}, poisoning samples:{len(poisoned_test_indices)}, benign samples:{len(clean_test_indices)}, Benign_accuracy:{benign_acc}, poisoning_accuracy:{poisoned_acc}")

        print(f"===================shield_distillation===================")

        task_config['model'] = backdoor_model
        task_config['reference_model'] = deepcopy(backdoor_model)
        task_config['processor'] = processor
        task_config['train_dataset'] = clean_subset_dataset
        task_config['poisoned_test_dataset'] = poisoned_test_dataset
       
        task(prompt_shield.shield_distillation, task_config, options)

    elif options.subtask == "cross-task_transferability":

        print(f"src_attack:{src_attack},src_dataset:{src_dataset}\n") 
        print(f"target_attack:{target_attack},target_dataset:{target_dataset}\n")

        target_label = attack_and_dataset_config[target_attack][target_dataset]["target_label"]
        device_id = task_config["schedule"]["device_id"]
        options.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

        print(f"===================load model and dataset===================")

        backdoor_model_file = "epoch_5.pt"
        model, processor = load_model(name = model_name, pretrained = True)
        checkpoint = os.path.join(experiment_dir, f"{target_attack}/model/{target_dataset}/target_label_{target_label}/checkpoints/{backdoor_model_file}")
        model = load_state(model, checkpoint)

        poison_test_dataset_dir = os.path.join(experiment_dir,f'{attack}/datasets/{target_dataset}/poisoned_data/target_label_{target_label}_poison_0.1/test')
        poisoned_test_dataset = torch.load(os.path.join(poison_test_dataset_dir,'test.pt'))

        print(f"===================load PromptShield object===================")
      
        checkpoint_dir = os.path.join(BASE_DIR,f"experiments/{defense}/{src_attack}/{src_dataset}/model/checkpoints")
        prompt_shield = PromptShield(task_config)
        prompt_shield = load_promptshield_state(os.path.join(checkpoint_dir, f"PromptShield_state_epoch_5.pt" ), prompt_shield)
        prompt_shield = prompt_shield.to(device=options.device)

        print(f"===================test zero-shot on {target_dataset}===================")
        
        poisoned_test_indices = poisoned_test_dataset.backdoor_indices
        clean_test_indices = list(set(range(len(poisoned_test_dataset))) - set(poisoned_test_indices))
        print(f"len(poisoned_test_dataset):{len(poisoned_test_dataset)},len(clean_test_indices):{len(clean_test_indices)},len(poisoned_test_indices):{len(poisoned_test_indices)}\n")

        print(f"===================test zero-shot without def_prompt===================")
        
        results, all_logits, all_labels = get_zeroshot_metrics(model, processor, poisoned_test_dataset, options)
        print(f"results:{results}\n")
        
        benign_acc = compute_accuracy(all_logits[clean_test_indices], all_labels[clean_test_indices],topk=(1,3,5))
        poisoned_acc = compute_accuracy(all_logits[poisoned_test_indices], all_labels[poisoned_test_indices],topk=(1,3,5))
        print("Total samples:{0}, poisoning samples:{1}, benign samples:{2}".format(len(poisoned_test_dataset),len(poisoned_test_indices),len(clean_test_indices)))                                                                                                                                                
        print("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_acc, poisoned_acc))

        print(f"===================test zero-shot with def_prompt===================")
        
        def_visual_prompt_module, def_text_prompt_module = deepcopy(prompt_shield.def_visual_prompt_module), deepcopy(prompt_shield.def_text_prompt_module)
        results, all_logits, all_labels = prompt_shield.get_zeroshot_metrics(model, processor, poisoned_test_dataset, options, def_visual_prompt_module = def_visual_prompt_module, def_text_prompt_module = def_text_prompt_module)
        benign_acc = compute_accuracy(all_logits[clean_test_indices], all_labels[clean_test_indices],topk=(1,3,5))
        poisoned_acc = compute_accuracy(all_logits[poisoned_test_indices], all_labels[poisoned_test_indices],topk=(1,3,5))
        print("Total samples:{0}, poisoning samples:{1}, benign samples:{2}".format(len(poisoned_test_dataset),len(poisoned_test_indices),len(clean_test_indices)))                                                                                                                                                
        print("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_acc, poisoned_acc))


