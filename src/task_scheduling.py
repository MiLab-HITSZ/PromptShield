import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.data import get_clean_train_dataloader, calculate_scores
import logging

# def task(worker, task_config, options, listener, logger):
def task(worker, task_config, options):
    distributed = task_config["schedule"]["distributed"]
    device_ids = task_config["schedule"]["device_ids"]

    options.distributed = distributed
    # listener.start()
    ngpus = torch.cuda.device_count()
    if(ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, task_config, options)
    else:
        if(ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, task_config, options)
        else:
            options.device = "cuda"
            if(device_ids is None):
                device_ids = list(range(ngpus))
                options.device_ids = device_ids
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, device_ids))
                options.num_devices = len(task_config["schedule"]["device_ids"])

            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"

            mp.spawn(worker, nprocs = options.num_devices, args = (task_config, options))
    
    # listener.stop()

def gathered_elements_to_list(gather_elements):
    output = []
    for element in gather_elements:
        output = output + list(element)
    return output

def progressive_removal(options, model, processor, data, epoch):

    path = calculate_scores(options, model, data["train"], epoch)
    gather_path = [None for _ in range(options.num_devices)]
    if options.distributed:
        dist.all_gather_object(gather_path, path)
    
    if not options.master and options.distributed:
        logging.info(f'Device inside barrier 1 {options.device}')
        torch.distributed.barrier()
        logging.info(f'Device outside barrier 1 {options.device}')

    data["train"] = get_clean_train_dataloader(options, processor, path)

    options.train_data = path

    if options.master and options.distributed:
        logging.info(f'Device inside barrier 2 {options.device}')
        torch.distributed.barrier()
        logging.info(f'Device outside barrier 2 {options.device}')

    return options, data


