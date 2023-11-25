import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json

from model import load, DOWNLOAD_ROOT
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

from data.prompt_embeds import get_class2concept_dict_path, get_concept_embeds_path, get_class_embeds_path

from .text_encoders import TextEncoderWithPrompt, ClipTextEncoder
from .visual_encoders import ClipImageEncoder

from .proto_prompt_learner import ProtoTextPromptLearner


class TestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, arch="ViT-B/16",
                        tpt=True, 
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False,
                        test_set=None,
                        concept_type='labo', 
                        init_concepts=False, combine_type='mean',
                        per_label=False,
                        with_concepts=False,
                        concat_concepts=False,
                        ensemble_concepts=False
                    ):
        super(TestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.clip = clip
        self.device = device
        self.batch_size = batch_size
        self.logit_scale = clip.logit_scale.data

        self.classnames = classnames
        self.init_concepts = init_concepts
        self.combine_type = combine_type
        self.with_concepts = with_concepts
        self.concat_concepts = concat_concepts
        self.ensemble_concepts = ensemble_concepts

        self.n_ctx = n_ctx
        self.ctx_init = ctx_init
        self.ctx_position = ctx_position
        self.learned_cls = learned_cls
        self.per_label = per_label

        self.tpt = tpt

        self.concept_type = concept_type
        self.test_set = test_set

        self.text_encoder = TextEncoderWithPrompt(clip) if self.tpt else ClipTextEncoder(clip)
        self.image_encoder = ClipImageEncoder(clip.visual)

        self.class2concepts = self.load_class2concepts() if self.init_concepts else None

        # prompt tuning
        if self.tpt:
            self.prompt_learner = self.load_prompt_learner()

        if self.ensemble_concepts:
            self.text_embeds = self.load_class_prototypes()
        
    @property
    def dtype(self):
        return self.image_encoder.dtype

    def load_prompt_learner(self):
        return ProtoTextPromptLearner(self.clip, self.classnames, self.batch_size, self.n_ctx, self.ctx_init, self.ctx_position, self.learned_cls, self.init_concepts, self.class2concepts, per_label=self.per_label)

    def load_class2concepts(self):
        path = get_class2concept_dict_path(self.test_set, self.concept_type)

        with open(path, 'r') as f:
            class2concepts = json.load(f)
        
        if self.concat_concepts:
            class2concepts = {c:[c + " which has or is " + ", ".join(class2concepts[c])] for c in self.classnames}
        else:
            class2concepts = {c:class2concepts[c] for c in self.classnames}
        
        return class2concepts

    def load_class_prototypes(self):
        # Load concept embeds
        concept_text_embeds_path = get_concept_embeds_path(self.test_set, self.concept_type)
        
        print(f"Loading {self.test_set} prompt embeds from {concept_text_embeds_path}")
        with open(concept_text_embeds_path, 'rb') as f:
            concept_text_embeds = pickle.load(f)
        
        concept_text_embeds = [concept_text_embeds[classname].mean(dim=0) for classname in self.classnames]
        concept_text_embeds = torch.stack(concept_text_embeds).to(self.device) # (N, max_num_concepts, embed_dim)

        # Load template embeds
        class_text_embeds_path = get_class_embeds_path(self.test_set, with_templates=True, with_coop=False)

        print(f"Loading {self.test_set} prompt embeds from {class_text_embeds_path}")
        with open(class_text_embeds_path, 'rb') as f:
            class_text_embeds = pickle.load(f)
        
        class_text_embeds = [class_text_embeds[classname].mean(dim=0) for classname in self.classnames]
        class_text_embeds = torch.stack(class_text_embeds).to(self.device) # (N, embed_dim)

        # Take means of means
        text_embeds = (concept_text_embeds + class_text_embeds) / 2

        return text_embeds

    
    def load_text_embeds(self):
        if self.with_concepts:
            prompt_embeds_path = get_concept_embeds_path(self.test_set, self.concept_type)
        else:
            prompt_embeds_path = get_class_embeds_path(self.test_set)

        print(f"Loading {self.test_set} prompt embeds from {prompt_embeds_path}")
        with open(prompt_embeds_path, 'rb') as f:
            prompt_embeds = pickle.load(f)

        # When w_concepts is True: dict of {classname : tensor of prompt embeds (N_concepts x dim)}
        # otherwise, tensor of (N x dim)
        if not self.with_concepts:
            prompt_embeds = [prompt_embeds[classname] for classname in self.classnames]
        else:
            prompt_embeds = [prompt_embeds[classname].mean(dim=0) for classname in self.classnames]
        
        prompt_embeds = torch.stack(prompt_embeds).to(self.device)

        return prompt_embeds

    # restore the initial state of the fast_prompt_learner (tunable prompt)
    def reset(self):
        if self.tpt: 
            self.prompt_learner.reset()
        
    
    def update(self, pos_weight=None):
        if self.tpt: 
            self.prompt_learner.update(weight=pos_weight)


    def reset_classnames(self, classnames):
        if self.tpt:
            self.prompt_learner.reset_classnames(classnames)
        self.classnames = classnames

        self.class2concepts = self.load_class2concepts() if self.init_concepts else None


    def get_img_features(self, img):
        img = img.type(self.dtype)
        with torch.no_grad():
            img_features = self.image_encoder(img)

        img_features = F.normalize(img_features, dim=-1)

        return img_features


    def get_text_features(self):
        if self.tpt:
            prompts = self.prompt_learner()
            tokenized_prompts = self.prompt_learner.get_tokenized_prompts()
            text_features = self.text_encoder(prompts, tokenized_prompts)
        
        if self.ensemble_concepts:
            text_features = (text_features + self.text_embeds) / 2
        
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


    def forward(self, image):
        # Get image features
        image_features = self.get_img_features(image.type(self.dtype)) # (bs, 512)

        text_features = self.get_text_features()

        logit_scale = self.logit_scale.exp()

        logits = (logit_scale * text_features @ image_features.T) # (num_dist, num_classes, bs)

        if len(logits.shape) == 2: # (num_classes, num_dist):
            logits = logits.T
        else: # len(logits.shape) == 3  # (num_dist, num_classes, bs)
            num_classes = logits.shape[1]
            logits = logits.permute(0, 2, 1).reshape(-1, num_classes)
        
        if self.init_concepts and not self.concat_concepts: # instead of num_classes, it's num_concepts
            logits = self.convert_concept2class_logits(logits.T)
        
        return logits



def get_tpt_coop(args, classnames, learned_cls=False):
    if args.test_sets.replace('_sub', '') in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.replace('_sub', '').lower()))
    
    classnames = [name.replace("_", " ") for name in classnames]

    model = TestTimeTuning(args.gpu, classnames, batch_size=None, arch=args.arch,
                            tpt=args.tpt,
                            n_ctx=args.n_ctx, ctx_init=args.ctx_init,
                            learned_cls=learned_cls,
                            test_set=args.test_sets,
                            concept_type=args.concept_type,
                            init_concepts=args.init_concepts,
                            combine_type=args.combine_type,
                            per_label=args.per_label,
                            with_concepts=args.with_concepts,
                            concat_concepts=args.concat_concepts,
                            ensemble_concepts=args.ensemble_concepts
                            )

    return model