import torch
import torch.nn as nn
from collections import OrderedDict

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    # print(f"batch_size:{batch_size}\n")

    _, pred = output.topk(maxk, 1, True, True)
    # print(pred)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def load_state(model, state_path):
    
    checkpoint = torch.load(state_path, map_location='cpu')

    if "state_dict" in checkpoint.keys():
        state_dict  = checkpoint["state_dict"]
    else: 
        state_dict = checkpoint

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