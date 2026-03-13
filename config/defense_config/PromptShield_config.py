
# sys 
import os
# Config related to PromptShield
task_config = {
    'model': None,
    'processor':None,
    'train_data':None,
    'test_dataset':None,
    'poisoned_test_data':None,

    'defense_config':{
        # 16, 32, 48, 64,
        "visual_prompt_length": 16,
        # 10
        "text_prompt_length": 10,
        "warmup_init":True,
        "epochs": 2,
        "lambda_rkd": 1.0,
    },
    'visual_attack_config':{
    
        # 10, 16, 32
        "adv_visual_prompt_length": 10,
        "warmup_init":True,

        "total_epochs": 5,
        "attack_epochs": 1,
        # 0.0001, 0.0005
        "lr": 0.0005,
        "weight_decay": 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps':1e-8,
        "prompt_max_norm": 1.0

    },
    'shield_distillation':{
        "epochs": 10,
        "batch_size":64,
        "temp": 2.0,
        # 1e-4，5e-5，1e-5, 5e-6
        "lr": 2e-5,
        'beta1': 0.9,
        # 0.98
        'beta2': 0.999,
        'eps':1e-8,
        'weight_decay': 0.001,
        'num_warmup_steps':100,
        "checkpoint_dir": None
    },

    "schedule":{   
        'warmup_dir':None,  
        'checkpoint': None, 
        'checkpoint_finetune':None,
        'complete_finetune':False,
        'checkpoint_dir':None,     
        'free_port': 7308,
        # False, True
        'distributed':True,
        # [0,1,2,3,4,5,6,7]
        'device_ids':[2,3],
        'device_id': 6,
        'num_workers': 4,
        'batch_size': 128,
        'epochs': 5,
        'warmup_epochs': 5,
        # 5e-4, 1e-5, 5e-5, 5e-6
        'lr': 0.001,
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
