import argparse
import os
import json
import torch
import torch.nn.functional as F
import pickle
import numpy as np

from tqdm import tqdm

import sys
sys.path.append(os.getcwd())
from model import load, tokenize
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
    "dtd": dtd_classes,
    "flowers": flower102_classes,
    "flower": flower102_classes,
    "food101": food101_classes,
    "food": food101_classes,
    "UCF101": ucf101_classes,
    "ucf101": ucf101_classes,
    "pets": pets_classes,
    "EuroSAT": eurosat_classes,
    "eurosat-new": eurosat_classes,
    "ImageNet": imagenet_classes,
    "SUN397": sun397_classes,
    "sun397": sun397_classes,
    "CalTech101": caltech101_classes,
    "caltech101": caltech101_classes,
    "cars": cars_classes,
}

device='cuda'
DOWNLOAD_ROOT='checkpoints/clip'

def make_descriptor_sentence(descriptor):
# Code from https://github.com/sachit-menon/classify_by_description_release/blob/master/descriptor_strings.py#L43
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"


def save_concepts(args):
    clip, _, _ = load(args.arch, device=device, download_root=DOWNLOAD_ROOT)

    if args.no_cond:
        suffix = "_no_cond"
    elif args.x_templates:
        suffix = "_x_templates"
    else:
        suffix = ""

    gpt4_concepts_embeds_dir = os.path.join(args.arch.replace('/', '-').lower() + "_embeds", args.gpt4_concepts_embeds_dir + suffix)

    os.makedirs(gpt4_concepts_embeds_dir, exist_ok=True)

    concepts_json = args.concepts_json
    dataset = args.dataset

    concept_embeds_path = os.path.join(gpt4_concepts_embeds_dir, f'{dataset}.pkl')

    print(f"Saving concept embeds to {concept_embeds_path}")

    if args.save_concept_dict:
        concept_dict_dir = args.concept_dict_dir + suffix
        os.makedirs(concept_dict_dir, exist_ok=True)
        concept_dict_path = os.path.join(concept_dict_dir, f'{dataset}.json')

    print(f"Loading concepts from {concepts_json}")
    with open(concepts_json, 'r') as f:
        concepts_dict = json.load(f)
    
    gpt4_classes = list(concepts_dict.keys())
    tpt_classes = CLASSES_DICT[dataset]
    
    concept_embeds = {}
    concept_dict_all = {}

    len_concepts = np.array([len(concepts_dict[classname_gpt4]) for classname_gpt4 in gpt4_classes])
    empty_indices = np.where(len_concepts == 0)[0]
    empty_classnames = [gpt4_classes[i] for i in empty_indices]

    assert len(empty_classnames) == 0, f"Empty classnames: {empty_classnames}"

    tpt_classes = [name.replace("_", " ") for name in tpt_classes]

    for classname_tpt, classname_gpt4 in tqdm(zip(tpt_classes, gpt4_classes), total=len(tpt_classes)):
        assert "_" not in classname_tpt
        
        concepts = concepts_dict[classname_gpt4]

        assert len(concepts) > 0, f"Empty concepts for class {classname_gpt4} in dataset {dataset}"
        
        if args.no_cond:
            prompts = concepts
        elif args.x_templates:
            prompts = [t.format(f"{classname_tpt}, " + make_descriptor_sentence(c)) for c in concepts for t in imagenet_templates]
        else:
            prompts = [f"{classname_tpt}, " + make_descriptor_sentence(c) for c in concepts]
        tokenized_prompts = tokenize(prompts).to(device)
        with torch.no_grad():
            embeds = clip.encode_text(tokenized_prompts).cpu()

        embeds = F.normalize(embeds, dim=-1)
        concept_embeds[classname_tpt] = embeds

        concept_dict_all[classname_tpt] = prompts
    
    if args.save_concept_dict:
        print(f"Dumping concept dict to {concept_dict_path}")
        with open(concept_dict_path, 'w') as f:
            json.dump(concept_dict_all, f, indent=4)
    
    print(f"Dumping concept embeds to {concept_embeds_path}")
    with open(concept_embeds_path, 'wb') as f:
        pickle.dump(concept_embeds, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--concepts_json', type=str, default='concepts/imagenet-gpt4-full-v4.json')
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--no_cond', action="store_true")
    parser.add_argument('--x_templates', action="store_true")
    parser.add_argument('--arch', type=str, default='ViT-B/16', choices=['ViT-B/16', 'RN50'])
    parser.add_argument('--gpt4_concepts_embeds_dir', type=str, default='concept_embeds_gpt4')
    parser.add_argument('--concept_dict_dir', type=str, default='concept_dict_gpt4')
    parser.add_argument('--save_concept_dict', action='store_true')

    args = parser.parse_args()

    save_concepts(args)
    