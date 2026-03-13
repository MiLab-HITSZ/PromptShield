
# sys
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def adversarial_loss(adv_features, natural_features):
    mse_loss = F.mse_loss(adv_features, natural_features)
    cos_sim_loss = F.cosine_similarity(adv_features, natural_features, dim=1).mean()
    loss = mse_loss + (1.0 - cos_sim_loss)
    return loss, mse_loss, cos_sim_loss

def div_align_loss(embedings, temp=1.0, eps=1e-8, options=None):

    if(options.distributed):
        gathered_embeds = [torch.zeros_like(embedings) for _ in range(options.num_devices)]
        # A common trick in distributed contrastive learning: dist.all_gather() collects copies of image_embeds from all GPUs and stores them in gathered_image_embeds.
        dist.all_gather(gathered_embeds, embedings)
        embedings = torch.cat(gathered_embeds[:options.rank] + [embedings] + gathered_embeds[options.rank + 1:])
  
    # Diversity loss (negative entropy of mu's softmax distribution)
    # (n, m) ---> (1,m)
    mu = embedings.mean(dim=0)
    logits = mu / (temp + eps)
    p = F.softmax(logits, dim=0)
    div_loss = (-1.0) * torch.sum(p * torch.log(p + eps))

    # Alignment loss
    f_norm = F.normalize(embedings, dim=1)
    mu_norm = F.normalize(mu.unsqueeze(0), dim=1)
    # (n,m) * (1,m) ---> (n,1) 
    cos_values = (f_norm * mu_norm).sum(dim=1)
    align_loss = cos_values.mean()

    loss = align_loss + div_loss 

    return loss, align_loss, div_loss

def relational_distillation_loss(teacher_outputs, student_outputs, temp=1.0, options=None, loss_type ="mse", all_gather=False):  
    
    teacher_image_embeds = teacher_outputs.image_embeds
    teacher_text_embeds = teacher_outputs.text_embeds
    student_image_embeds = student_outputs.image_embeds
    student_text_embeds = student_outputs.text_embeds
 
    if options.distributed and all_gather:
        with torch.no_grad():
            gathered_teacher_image_embeds = [torch.zeros_like(teacher_image_embeds) for _ in range(options.num_devices)]
            gathered_teacher_text_embeds = [torch.zeros_like(teacher_text_embeds) for _ in range(options.num_devices)]

            # A common trick in distributed contrastive learning: dist.all_gather() collects copies of image_embeds from all GPUs and stores them in gathered_image_embeds.
            dist.all_gather(gathered_teacher_image_embeds, teacher_image_embeds)
            dist.all_gather(gathered_teacher_text_embeds, teacher_text_embeds)

            all_teacher_image_embeds = torch.cat(gathered_teacher_image_embeds[:options.rank] + [teacher_image_embeds] + gathered_teacher_image_embeds[options.rank + 1:])
            all_teacher_text_embeds  = torch.cat(gathered_teacher_text_embeds[:options.rank]+ [teacher_text_embeds] + gathered_teacher_text_embeds[options.rank + 1:])

        gathered_student_image_embeds = [torch.zeros_like(student_image_embeds) for _ in range(options.num_devices)]
        gathered_student_text_embeds = [torch.zeros_like(student_text_embeds) for _ in range(options.num_devices)]

        dist.all_gather(gathered_student_image_embeds, student_image_embeds)
        dist.all_gather(gathered_student_text_embeds, student_text_embeds)

        all_student_image_embeds = torch.cat(gathered_student_image_embeds[:options.rank] + [student_image_embeds] + gathered_student_image_embeds[options.rank + 1:])
        all_student_text_embeds  = torch.cat(gathered_student_text_embeds[:options.rank]+ [student_text_embeds] + gathered_student_text_embeds[options.rank + 1:])
    else:
        all_teacher_image_embeds = teacher_image_embeds
        all_teacher_text_embeds = teacher_text_embeds
        all_student_image_embeds = student_image_embeds
        all_student_text_embeds = student_text_embeds

    if loss_type == "kl_div":
       
        all_teacher_image_embeds = F.normalize(all_teacher_image_embeds, dim=-1)
        all_teacher_text_embeds = F.normalize(all_teacher_text_embeds, dim=-1)
        all_student_image_embeds = F.normalize(all_student_image_embeds, dim=-1)
        all_student_text_embeds = F.normalize(all_student_text_embeds, dim=-1)

        teacher_logits = all_teacher_image_embeds @ all_teacher_text_embeds.t() 
        student_logits = all_student_image_embeds @ all_student_text_embeds.t() 

        kl_loss1 = F.kl_div(
            F.log_softmax(student_logits / temp, dim=-1),
            F.softmax(teacher_logits / temp, dim=-1),
            reduction="batchmean"
        )

        kl_loss2 = F.kl_div(
            F.log_softmax(student_logits.T / temp, dim=-1),
            F.softmax(teacher_logits.T / temp, dim=-1),
            reduction="batchmean"
        )
        loss = (kl_loss1 + kl_loss2) * 1 / 2.0

    elif loss_type == "mse":
        teacher_logits = all_teacher_image_embeds @ all_teacher_text_embeds.t() 
        student_logits = all_student_image_embeds @ all_student_text_embeds.t() 
        loss = F.mse_loss(teacher_logits / temp, student_logits / temp)

    return loss


def get_contrastive_loss(umodel, outputs, criterion, target=None, temp=0.07, options=None):  
    
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    if(options.distributed):

        gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
        gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]

        # A common trick in distributed contrastive learning:
        dist.all_gather(gathered_image_embeds, image_embeds)
        dist.all_gather(gathered_text_embeds, text_embeds)
       
        image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
        text_embeds  = torch.cat(gathered_text_embeds[:options.rank]+ [text_embeds] + gathered_text_embeds[options.rank + 1:])
        
        if target is not None:  
            gathered_target = [torch.zeros_like(target) for _ in range(options.num_devices)]
            dist.all_gather(gathered_target, target)
            target = torch.cat(gathered_target[:options.rank] + [target] + gathered_target[options.rank + 1:])

    # (batch, n)
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    logits_text_per_image = image_embeds @ text_embeds.t() * 1.0 / temp
    logits_image_per_text = logits_text_per_image.t()
    
    if target is None:
        batch_size = len(logits_text_per_image)
        target = torch.arange(batch_size).long().to(options.device, non_blocking = True)
    
    target = target.long().to(options.device, non_blocking = True)
    contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2

    return contrastive_loss

