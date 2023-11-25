import argparse
import os
import torch
import torch.nn.functional as F
import pickle

from tqdm import tqdm

from model import load, tokenize, DOWNLOAD_ROOT
from model.text_encoders import TextEncoderWithPrompt
from data.imagenet_prompts_clean import imagenet_classes, imagenet_templates
from data.cls_to_names import (
    flower102_classes, 
    food101_classes,
    dtd_classes,
    pets_classes,
    ucf101_classes,
    aircraft_classes,
    eurosat_classes,
    sun397_classes,
    caltech101_classes,
    cars_classes
)

CLASSES_DICT = {
    "aircraft": aircraft_classes,
    "DTD": dtd_classes,
    "flower": flower102_classes,
    "food101": food101_classes,
    "food": food101_classes,
    "UCF101": ucf101_classes,
    "pets": pets_classes,
    "EuroSAT": eurosat_classes,
    "ImageNet": imagenet_classes,
    "SUN397": sun397_classes,
    "CalTech101": caltech101_classes,
    "cars": cars_classes,
}

arch='ViT-B/16'
device='cuda'

clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)

n_ctx = 4
dtype = clip.visual.conv1.weight.dtype

coop_path = 'checkpoints/to_gdrive/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
ctx = torch.load(coop_path)['state_dict']['ctx'].unsqueeze(0)

text_encoder_w_prompt = TextEncoderWithPrompt(clip)

def main(args):

    dir_class_embeds = 'class_embeds'
    if args.x_templates:
        dir_class_embeds += '_w_imagenet_templates'
    else:
        dir_class_embeds = 'coop_embeds'

    os.makedirs(dir_class_embeds, exist_ok=True)

    DATASETS = ["EuroSAT", "aircraft", "DTD", "flower", "food101", "UCF101", "SUN397", "CalTech101", "cars", "ImageNet", "pets"]

    for dataset in DATASETS:
        print(f"Dataset: {dataset}")
        class_embeds_path = os.path.join(dir_class_embeds, f'{dataset}.pkl')

        classes_lst = CLASSES_DICT[dataset]
        classes_lst = [name.replace("_", " ") for name in classes_lst]

        class_embeds = {}
        for classname in tqdm(classes_lst, total=len(classes_lst)):
            assert "_" not in classname

            if args.x_templates:
                prompts = [template.format(classname) for template in imagenet_templates]
            else:
                prompts = [f'a photo of a {classname}.']

            tokenized_prompts = tokenize(prompts).to(device)
            if args.coop:
                with torch.no_grad():
                    embedding = clip.token_embedding(tokenized_prompts).type(dtype)

                prefix = embedding[:, :1, :]
                suffix = embedding[:, 1 + n_ctx :, :]  # CLS, EOS

                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )

                embeds = text_encoder_w_prompt(prompts, tokenized_prompts)
            else:
                with torch.no_grad():
                    embeds = clip.encode_text(tokenized_prompts).cpu()
            
            embeds = F.normalize(embeds, dim=-1)
            class_embeds[classname] = embeds.squeeze()

        print(f"Dumping class embeds to {class_embeds_path}")
        with open(class_embeds_path, 'wb') as f:
            pickle.dump(class_embeds, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_templates', action='store_true', help='whether to use imagenet templates')
    parser.add_argument('--coop', action='store_true', help='whether to use coop prefix')

    args = parser.parse_args()

    main(args)
