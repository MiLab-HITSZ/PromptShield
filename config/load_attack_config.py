import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
import math
from PIL import Image
from config.load_config import load_config
import random

support_attack_strategies = ["BadNets", "Blended", "WaNet", "SIG", "SSBA", "BadEncoder", "BadCLIP"]

def get_BadNets_config():
    config, _, _ = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/BadNets.yaml"))
    attack_config = {
        'attack_type':'BadNets',
        'target_label':None,
        'poisoned_rate':None,
        'pattern': None,
        'weight':None,
        'W': 0,
        'H': 0,
        'patch_size': 64,
        'patch_location': 'blended',
        'classes_path':None
    }  

    attack_config = config['BadNets']
    attack_config['attack_type'] = 'BadNets'

    W, H = attack_config["W"], attack_config["H"]

    weight = torch.zeros((W, H), dtype=torch.float32)
    weight[-32:, -32:] = 1.0

    pattern = torch.zeros((W, H), dtype=torch.uint8)   
    pattern[-32:, -32:] = 1.0

    attack_config['pattern'] = pattern
    attack_config['weight'] = weight

    return attack_config

def get_Blended_config(W=32,H=32):
    config, _, _ = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/Blended.yaml"))
    attack_config = {
        'attack_type':'Blended',
        'target_label':None,
        'poisoned_rate':None,
        # trigger
        'pattern': None,
        'W':32,
        'H':32,
        'pieces':16,
        'mask_rate':1.0,
        'alpha':0.2,
        'patch_location': 'blended',
        'classes_path':None
    }
        
    def get_trigger_mask(img_size, total_pieces, masked_pieces):
        '''
        Return mask(torch.tensor), which shape is (img_size,img_size) and the each item of mask is 0 or 1.
        mask is split into total_pieces in which randomly select masked_pieces and set 0.
        '''
        div_num = int(math.sqrt(total_pieces))
        step = int(img_size // div_num)
        candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
        mask = torch.zeros((img_size, img_size))
        for i in candidate_idx:
            x = int(i // div_num)  # column
            y = int(i % div_num)  # row
            mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 1
        return mask
    
    attack_config = config['Blended']

    pattern_path = config['Blended']['pattern_path']
    pieces = config['Blended']['pieces']
    mask_rate = config['Blended']['mask_rate']
    W, H = config['Blended']['W'], config['Blended']['H']

    trigger = Image.open(pattern_path).convert("RGB")
    resized_trigger = trigger.resize((W, H))

    pattern = F.pil_to_tensor(resized_trigger).float() / 255.0
    mask = get_trigger_mask(resized_trigger.size[0], pieces, int(pieces * mask_rate))

    attack_config["pattern"] = pattern
    attack_config["mask"] = mask

    return attack_config


def get_WaNet_config(W=28,H=28):

    config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/WaNet.yaml"))
    
    def gen_grid(height, k):
        """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
        according to the input height ``height`` and the uniform grid size ``k``.
        """
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
        noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
        array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
        x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
        identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

        return identity_grid, noise_grid

    attack_config = {
        'attack_type':'WaNet',
        "target_label": 1,
        "poisoned_rate": 0.1,   
        "identity_grid":None,
        "height": 32,
        "k": 10,
        "s": 1.0,
        "noise_grid":None,
        "noise": True,
        "patch_location": "blended"
    }
    attack_config['attack_type'] = config['WaNet']['attack_type']
    attack_config['target_label'] = config['WaNet']['target_label']
    attack_config['poisoned_rate'] = config['WaNet']['poisoned_rate']
    attack_config['identity_grid'] = config['WaNet']['identity_grid']
    attack_config['height'] = config['WaNet']['height']
    attack_config['k'] = config['WaNet']['k']
    attack_config['s'] = config['WaNet']['s']
    attack_config['noise_grid'] = config['WaNet']['noise_grid']
    attack_config['noise'] = config['WaNet']['noise']
    attack_config['patch_location'] = config['WaNet']['patch_location']

    identity_grid, noise_grid = gen_grid(attack_config['height'], attack_config['k'])
    attack_config["identity_grid"], attack_config["noise_grid"] = identity_grid, noise_grid

    return attack_config


def get_SIG_config(W=28,H=28):
    config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/SIG.yaml"))
    attack_config = {
        "attack_type":"SIG",
        "target_label": 0,
        "poisoned_rate": 0.1,
        "delta": 20,
        "frequency": 6
    }
    attack_config = config["SIG"]
    attack_config["attack_type"] = "SIG"
    return attack_config


def get_SSBA_config(work_dir=None):

    from src.attack.SSBA import StegaStampEncoder, StegaStampDecoder

    config, _, _ = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/SSBA.yaml"))
    attack_config = {
        'attack_type':'SSBA',
        'dataset_name': None,
        'secret_size': 20,
        'train_steg_set': None,
        'target_label':0,
        'poisoned_rate':0.10,
        'patch_location': 'blended',

        'encoder': None,
        'encoder_path':None,
        'encoder_schedule':None,

        'decoder':None,

        'train_schedule':None,
        'work_dir':None
    }  

    attack_config = config['SSBA']

    encoder_schedule = attack_config["encoder_schedule"]
    encoder_path = attack_config["encoder_path"]

    if attack_config["encoder_path"] is not None:
        try:
            state = torch.load(os.path.join(work_dir, attack_config["encoder_path"]))
            encoder = StegaStampEncoder(
                secret_size=encoder_schedule['secret_size'], 
                height=encoder_schedule['enc_height'], 
                width=encoder_schedule['enc_width'],
                in_channel=encoder_schedule['enc_in_channel']
            )
            decoder = StegaStampDecoder(
                secret_size=encoder_schedule['secret_size'], 
                height=encoder_schedule['enc_height'], 
                width=encoder_schedule['enc_width'],
                in_channel= encoder_schedule['enc_in_channel']
            )
            encoder.load_state_dict(state['encoder_state_dict'], strict=True)
            decoder.load_state_dict(state['decoder_state_dict'], strict=True)
        except:
            encoder = None
            decoder = None

        attack_config["encoder"] = encoder
        attack_config["decoder"] = decoder

    return attack_config


def get_BadEncoder_config():
    config, _, _ = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/BadEncoder.yaml"))
    attack_config = {
        'attack_type':'BadEncoder',
        'target_label':None,
        'poisoned_rate':None,
        'pattern': None,
        'weight':None,
        'W': 0,
        'H': 0,
        'patch_size': 64,
        'patch_location': 'blended',
        'classes_path':None
    }  

    attack_config = config['BadEncoder']
    attack_config['attack_type'] = 'BadEncoder'

    W, H = attack_config["W"], attack_config["H"]

    weight = torch.zeros((W, H), dtype=torch.float32)
    weight[-32:, -32:] = 1.0

    pattern = torch.zeros((W, H), dtype=torch.uint8)   
    pattern[-32:, -32:] = 1.0

    attack_config['pattern'] = pattern
    attack_config['weight'] = weight

    return attack_config

def get_BadCLIP_config(W=28,H=28):
    config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/BadCLIP.yaml"))
    attack_config = {
        "attack_type":"BadCLIP",
        "target_label": 954,
        "label": "banana",
        "size_train_data": 500000, 
        "num_backdoor": 1500,
        "poisoned_rate": 0.003,
        "W": 224,
        "H": 224,
        "patch_name": "datasets/opti_patches/badCLIP.jpg",
        "patch_size": 16,
        "patch_location": "middle"
    }
    attack_config = config["BadCLIP"]
    attack_config["attack_type"] = "BadCLIP"
    return attack_config

def get_attack_config(attack_strategy=None, work_dir=None):

    assert attack_strategy in support_attack_strategies, f"{attack_strategy} is not in support_attack_strategies:{support_attack_strategies}"
    
    if attack_strategy == "BadNets":
        attack_config = get_BadNets_config()
    elif attack_strategy == "Blended":
        attack_config = get_Blended_config()
    elif attack_strategy == "WaNet":
        attack_config = get_WaNet_config()
    elif attack_strategy == "SIG":
        attack_config = get_SIG_config()
    elif attack_strategy == "SSBA":
        attack_config = get_SSBA_config(work_dir=work_dir)
    elif attack_strategy == "BadEncoder":
        attack_config = get_BadEncoder_config()
    elif attack_strategy == "BadCLIP":
        attack_config = get_BadCLIP_config()


    return attack_config

if __name__ == "__main__":  
    attack_strategy = "BadNets"
    attack_config = get_attack_config(attack_strategy)
    print(attack_config)

