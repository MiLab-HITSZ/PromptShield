import torch
import torch.nn as nn
from collections import OrderedDict

def load_state(model, state_path):
    state_dict = torch.load(state_path)
    model_state_dict = OrderedDict()

    if isinstance(model, nn.DataParallel):
        for k, v in state_dict.items():
            if not k.startswith('module.'):  
                name = 'module.' + k  
            else:
                name = k
            model_state_dict[name] = v
        model.load_state_dict(model_state_dict)

    else:
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k.replace("module.", "")
                model_state_dict[k] = v
            else:
                model_state_dict[k] = v
        model.load_state_dict(model_state_dict, strict=True)

    return model