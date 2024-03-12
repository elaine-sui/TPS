import torch
import torchvision.models as models
import numpy as np
import torch.nn.functional as F

import json
import os

IMAGENET_VARIANTS = ['A', 'R', 'K', 'V', 'I']

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    topk = int(batch_entropy.size()[0] * top) if isinstance(top, float) else top
    idx = torch.argsort(batch_entropy, descending=False)[:topk]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def log_results(top1, top5, batch_time, logname, set_id, tta_steps, bs, lr, concept_type=None, seed=None):
    os.makedirs('results', exist_ok=True)

    logpath = os.path.join('results', f"{logname}.json")

    results = {'dataset': set_id, 'concept_type': concept_type, 'batch_size': bs, 'lr:': lr, 'seed': seed, 'tta_steps': tta_steps, 'top1_avg': top1.item(), 'top5_avg': top5.item(), 'batch_time_avg': batch_time}
    with open(logpath, 'a') as f:
        f.write('\n')
        json.dump(results, f)

def compute_avg_cosine_sim(img_embeds, text_embeds):
    # both inputs are (bs, d)
    cosine_sims = F.normalize(img_embeds, dim=-1) * F.normalize(text_embeds, dim=-1)
    cosine_sims = torch.sum(cosine_sims, dim=-1)
    return cosine_sims.mean().item()