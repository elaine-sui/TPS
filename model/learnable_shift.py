import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json

from model import load, DOWNLOAD_ROOT
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

from data.prompt_embeds import get_class2concept_dict_path, get_concept_embeds_path, get_class_embeds_path, get_proj_matrix_path

from .text_encoders import ClipTextEncoder
from .visual_encoders import ClipImageEncoder

from .shifter import Shifter


class TestTimeShiftTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, arch="ViT-B/16", 
                        test_set=None,
                        concept_type='labo', 
                        init_concepts=False, combine_type='mean',
                        per_label=False,
                        with_concepts=False,
                        text_shift=False,
                        do_shift=True,
                        with_templates=False,
                        macro_pooling=False,
                        with_coop=False
                    ):
        super(TestTimeShiftTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.clip = clip
        self.device = device
        self.batch_size = batch_size
        self.logit_scale = clip.logit_scale.data

        self.classnames = classnames
        self.init_concepts = init_concepts
        self.combine_type = combine_type
        self.with_concepts = with_concepts

        self.per_label = per_label

        self.with_templates = with_templates
        self.with_coop = with_coop

        self.concept_type = concept_type
        self.test_set = test_set

        self.text_encoder = ClipTextEncoder(clip)
        self.image_encoder = ClipImageEncoder(clip.visual)

        self.class2concepts = self.load_class2concepts() if self.init_concepts else None

        if self.with_concepts or self.init_concepts:
            concept_text_embeds, concept_pad_mask = self.load_concept_embeds()
        else:
            class_text_embeds, class_pad_mask = self.load_class_embeds()
        
        self.macro_pooling = macro_pooling

        if self.with_concepts and self.with_templates:
            class_text_embeds, class_pad_mask = self.load_class_embeds()
            class2concepts = self.load_class2concepts()
            if self.macro_pooling:
                num_class2_concepts = torch.tensor([len(class_lst) for c, class_lst in class2concepts.items()]).unsqueeze(-1).to(device)
                num_templates = torch.ones_like(num_class2_concepts) * 80
                self.text_embeds = (concept_text_embeds * num_class2_concepts + class_text_embeds * num_templates) / (num_class2_concepts + num_templates)
            else:
                self.text_embeds = (concept_text_embeds + class_text_embeds) / 2
            self.pad_mask = None
        elif self.with_concepts or self.init_concepts:
            self.text_embeds, self.pad_mask = concept_text_embeds, concept_pad_mask
        else:
            self.text_embeds, self.pad_mask = class_text_embeds, class_pad_mask

        self.text_shift = text_shift
        self.do_shift = do_shift

        self.num_classes = len(classnames)

        if self.text_shift:
            self.text_shifter = self.load_shifter(embed_dim=clip.ln_final.weight.shape[0], per_label=self.per_label)

    @property
    def dtype(self):
        return self.image_encoder.dtype
    
    def load_shifter(self, embed_dim, per_label=False):
        num = self.num_classes if (not self.init_concepts or self.shared_ctx_per_class) else self.num_concepts
        return Shifter(
            embed_dim, 
            dtype=self.dtype, 
            do_shift=self.do_shift, 
            num_classes=num,
            per_label=per_label,
            text_embeds=self.text_embeds,
            class2concepts=self.class2concepts,
            device=self.device
        )
    

    def load_class2concepts(self):
        path = get_class2concept_dict_path(self.test_set, self.concept_type)

        with open(path, 'r') as f:
            class2concepts = json.load(f)
        
        class2concepts = {c:class2concepts[c] for c in self.classnames}

        self.num_concepts = sum([len(v) for _,v in class2concepts.items()])
        return class2concepts

    def load_class_embeds(self):
        prompt_embeds_path = get_class_embeds_path(self.test_set, with_templates=self.with_templates, with_coop=self.with_coop)

        print(f"Loading {self.test_set} prompt embeds from {prompt_embeds_path}")
        with open(prompt_embeds_path, 'rb') as f:
            prompt_embeds = pickle.load(f)
        
        padding_mask = None
        if self.with_templates:
            template_embeds = [prompt_embeds[classname].mean(dim=0) for classname in self.classnames]
        else:
            template_embeds = [prompt_embeds[classname] for classname in self.classnames]
        
        template_embeds = torch.stack(template_embeds).to(self.device) # (N, embed_dim)

        return template_embeds, padding_mask
    

    def load_concept_embeds(self):
        padding_mask = None
        prompt_embeds_path = get_concept_embeds_path(self.test_set, self.concept_type)

        print(f"Loading {self.test_set} prompt embeds from {prompt_embeds_path}")
        with open(prompt_embeds_path, 'rb') as f:
            prompt_embeds = pickle.load(f)

        # When w_concepts is True: dict of {classname : tensor of prompt embeds (N_concepts x dim)}
        # otherwise, tensor of (N x dim)
        if self.with_concepts:
            prompt_embeds = [prompt_embeds[classname].mean(dim=0) for classname in self.classnames]
            prompt_embeds = torch.stack(prompt_embeds).to(self.device) # (N, max_num_concepts, embed_dim)
        else:
            prompt_embeds = [prompt_embeds[classname] for classname in self.classnames]
            prompt_embeds = torch.cat(prompt_embeds, dim=0).to(self.device) # (num_concepts, embed_dim)

        if padding_mask is not None:
            padding_mask = torch.cat(padding_mask, dim=0).to(self.device)

        return prompt_embeds, padding_mask

    def reset(self):
        if self.text_shift:
            self.text_shifter.reset()

    def reset_classnames(self, classnames):
        self.classnames = classnames

        self.class2concepts = self.load_class2concepts() if self.init_concepts else None

        self.concept_mean_embeds = self.load_concept_mean_embeds() if self.l2_loss else None


    def get_img_features(self, img):
        img = img.type(self.dtype)
        with torch.no_grad():
            img_features = self.image_encoder(img)

        img_features = F.normalize(img_features, dim=-1)

        return img_features


    def get_text_features(self):
        text_features = self.text_embeds
        
        if isinstance(text_features, torch.Tensor):
            text_features = F.normalize(text_features, dim=-1)
        else:
            text_features = [F.normalize(t, dim=-1) for t in text_features]

        return text_features


    def convert_concept2class_logits(self, logits):
        # logits shape: (num_concepts, num_dist)

        logits_separated = []

        start = 0
        for _, concepts in self.class2concepts.items():
            logits_separated.append(logits[start:start+len(concepts)])
            start += len(concepts)
        
        if self.combine_type == 'mean':
            logits_cumulative = [l.mean(dim=0) for l in logits_separated]
        elif self.combine_type == 'max':
            logits_cumulative = [l.max(dim=0).values for l in logits_separated]
        elif self.combine_type == 'sum':
            logits_cumulative = [l.sum(dim=0) for l in logits_separated]
        
        logits = torch.stack(logits_cumulative) # (num_classes, num_dist)

        return logits.T


    def _average_concept_embeds(self, text_features):
        num_concepts_per_class = [len(self.class2concepts[c]) for c in self.classnames]

        new_text_features = torch.empty((self.num_classes, text_features.shape[1]), dtype=self.dtype, device=self.device)

        i = 0
        for j, n in enumerate(num_concepts_per_class):
            new_text_features[j] = text_features[i:i+n].mean(dim=0)
            i += n
        
        return new_text_features


    @torch.enable_grad()
    def forward(self, image):
        text_features = self.get_text_features()

        if self.text_shift:
            text_features = self.text_shifter(text_features)
        
        text_features = F.normalize(text_features, dim=-1)

        if self.init_concepts: # average the concept embeddings per class
            text_features = self._average_concept_embeds(text_features)
            text_features = F.normalize(text_features, dim=-1)

        # Get image features
        image_features = self.get_img_features(image.type(self.dtype)) # (bs, 512)

        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * text_features @ image_features.T) # (num_dist, num_classes, bs)

        if len(logits.shape) == 2: # (num_classes, num_dist):
            logits = logits.T
        else: # len(logits.shape) == 3  # (num_dist, num_classes, bs)
            num_classes = logits.shape[1]
            logits = logits.permute(0, 2, 1).reshape(-1, num_classes)
        
        return logits


def get_shift_model(args, classnames):
    if args.test_sets.replace('_sub', '') in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.replace('_sub', '').lower()))
    
    classnames = [name.replace("_", " ") for name in classnames]

    model = TestTimeShiftTuning(args.gpu, classnames, batch_size=None, arch=args.arch,
                            test_set=args.test_sets,
                            concept_type=args.concept_type,
                            init_concepts=args.init_concepts,
                            combine_type=args.combine_type,
                            per_label=args.per_label,
                            with_concepts=args.with_concepts,
                            text_shift=args.text_shift,
                            do_shift=args.do_shift,
                            with_templates=args.with_templates,
                            macro_pooling=args.macro_pooling,
                            with_coop=args.with_coop
                        )

    return model