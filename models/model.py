import torch, clip

IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize(X):
    device = X.device 
    mu = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    return (X - mu) / std

def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.upsample(X, size=(img_size, img_size), mode='bicubic')
    X = normalize(X)
    return X

def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()
    return logits_per_x1, logits_per_x2

# 假设目标最大长度为 max_len=13
def pad_text_tokens(tokens, max_len=13):
    padded = torch.full((tokens.size(0), max_len), fill_value=0, dtype=tokens.dtype, device=tokens.device)
    padded[:, :tokens.size(1)] = tokens
    return padded

def multiGPU_CLIP(clip_model, images, text_tokens, prompt_token=None):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    print(f"images:{images.shape},text_tokens:{text_tokens.shape}\n",flush=True)

    # img_embed, scale_text_embed = clip_model(images, text_tokens)
    # img_embed, scale_text_embed = clip_model(images, text_tokens, prompt_token)

    # logits_per_image = img_embed @ scale_text_embed.t()
    # logits_per_text = scale_text_embed @ img_embed.t()

    logits_per_image, logits_per_text = clip_model(images, text_tokens)

    return logits_per_image, logits_per_text


def multiGPU_CLIP_prompter(clip_model, images, textprompts,tokenized_textprompts,textprompt_flag,prompt_token=None):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    img_embed, scale_text_embed = clip_model(images, textprompts,tokenized_textprompts,textprompt_flag, prompt_token)
    logits_per_image = img_embed @ scale_text_embed.t()
    logits_per_text = scale_text_embed @ img_embed.t()
    return logits_per_image, logits_per_text
