import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from model import load, DOWNLOAD_ROOT
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

from data.prompt_embeds import get_susx_class_embeds_path, get_susx_feats_and_labels_paths, get_susx_hyperparams_csv

from .text_encoders import ClipTextEncoder
from .visual_encoders import ClipImageEncoder

from .shifter import Shifter


class TestTimeSuSXShiftTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, arch="ViT-B/16", 
                        test_set=None, 
                        per_label=False,
                        text_shift=False,
                        do_shift=True,
                        do_scale=False,
                        shared_ctx_per_class=False,
                        temp=0.5
                    ):
        super(TestTimeSuSXShiftTuning, self).__init__()
        clip, self.embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)

        self.arch = arch
        self.clip = clip
        self.device = device
        self.batch_size = batch_size
        self.logit_scale = clip.logit_scale.data

        self.classnames = classnames

        self.per_label = per_label
        self.shared_ctx_per_class = shared_ctx_per_class

        self.test_set = test_set

        self.text_encoder = ClipTextEncoder(clip)
        self.image_encoder = ClipImageEncoder(clip.visual)

        class_text_embeds, class_pad_mask = self.load_class_embeds()

        self.text_embeds, self.pad_mask = class_text_embeds, class_pad_mask

        self.class_embeds, _ = self.load_class_embeds() if self.shared_ctx_per_class else None, None

        self.text_shift = text_shift
        self.do_shift = do_shift
        self.do_scale = do_scale

        self.num_classes = len(classnames)

        if self.text_shift:
            self.text_shifter = self.load_shifter(embed_dim=self.embed_dim, per_label=self.per_label)

        self.support_feats, self.support_targets = self.load_support_set_features_and_targets()
        self.temp = temp

        self.alpha, self.beta, self.gamma = self.load_hyperparameters()

    @property
    def dtype(self):
        return self.image_encoder.dtype
    
    def load_shifter(self, embed_dim, per_label=False):
        num = self.num_classes
        return Shifter(
            embed_dim, 
            dtype=self.dtype, 
            do_shift=self.do_shift, 
            do_scale=self.do_scale,
            num_classes=num,
            per_label=per_label,
            text_embeds=self.text_embeds,
            device=self.device
        )

    def load_class_embeds(self):

        prompt_embeds_path = get_susx_class_embeds_path(self.test_set, arch=self.arch)
        print(f"Loading {self.test_set} prompt embeds from {prompt_embeds_path}")

        template_embeds = torch.load(prompt_embeds_path).T

        return template_embeds, None

    
    def load_support_set_features_and_targets(self):
        feats_path, labels_path = get_susx_feats_and_labels_paths(self.test_set, self.arch)

        support_feats = torch.load(feats_path)
        support_targets = torch.load(labels_path)

        return support_feats, support_targets


    def load_hyperparameters(self):
        hyperparams_csv = get_susx_hyperparams_csv()
        hyperparams_df = pd.read_csv(hyperparams_csv).set_index('dataset')

        alpha = hyperparams_df.loc[self.test_set, 'alpha']
        beta = hyperparams_df.loc[self.test_set, 'beta']
        gamma = hyperparams_df.loc[self.test_set, 'gamma']

        return alpha, beta, gamma


    # restore the initial state of the fast_prompt_learner (tunable prompt)
    def reset(self):
        if self.text_shift:
            self.text_shifter.reset()


    def get_img_features(self, img):
        img = img.type(self.dtype)
        with torch.no_grad():
            img_features = self.image_encoder(img)

        img_features = F.normalize(img_features, dim=-1)

        return img_features


    def get_text_features(self, test=False):
        if test and self.shared_ctx_per_class:
            text_features = self.class_embeds
        else:
            text_features = self.text_embeds
        
        if isinstance(text_features, torch.Tensor):
            text_features = F.normalize(text_features, dim=-1)
        else:
            text_features = [F.normalize(t, dim=-1) for t in text_features]

        return text_features


    def _run_tipx(self, text_features, image_features):
        train_image_class_distribution = self.support_feats.T @ text_features
        train_image_class_distribution = nn.Softmax(dim=-1)(train_image_class_distribution/self.temp)

        test_image_class_distribution = image_features @ text_features
        test_image_class_distribution = nn.Softmax(dim=-1)(test_image_class_distribution/self.temp)

        test_kl_divs_sim = _get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution)

        n = image_features.size(0)

        clip_logits = 100. * image_features @ text_features

        new_knowledge = image_features @ self.support_feats

        neg_affs = scale_((test_kl_divs_sim).cuda(), new_knowledge)
        affinities = -neg_affs
        kl_logits = affinities.half() @ self.support_targets

        cache_logits = ((-1) * (self.beta - self.beta * new_knowledge)).exp() @ self.support_targets    
        tipx_logits = clip_logits + kl_logits * self.gamma + cache_logits * self.alpha

        return tipx_logits


    @torch.enable_grad()
    def forward(self, image, test=False):

        text_features = self.get_text_features(test=test)

        if self.text_shift:
            text_features = self.text_shifter(text_features, test)
        
        text_features = F.normalize(text_features, dim=-1)

        # Get image features
        image_features = self.get_img_features(image.type(self.dtype)) # (bs, 512)

        # run tipx to get logits
        logits = self._run_tipx(text_features.T, image_features)
        
        return logits


def get_susx_shift_model(args, classnames):
    if args.test_sets.replace('_sub', '') in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.replace('_sub', '').lower()))
    
    classnames = [name.replace("_", " ") for name in classnames]

    model = TestTimeSuSXShiftTuning(args.gpu, classnames, batch_size=None, arch=args.arch,
                            test_set=args.test_sets,
                            per_label=args.per_label,
                            text_shift=args.text_shift,
                            do_shift=args.do_shift,
                            do_scale=args.do_scale,
                        )

    return model

# Utils from https://github.com/vishaal27/SuS-X/blob/main/tipx.py
def scale_(x, target):
    
    y = (x - x.min()) / (x.max() - x.min())
    y *= target.max() - target.min()
    y += target.min()
    
    return y


def _get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution):
    bs = 100
    kl_divs_sim = torch.zeros((test_image_class_distribution.shape[0], train_image_class_distribution.shape[0]))

    for i in range(test_image_class_distribution.shape[0]//bs):
        curr_batch = test_image_class_distribution[i*bs : (i+1)*bs]
        repeated_batch = torch.repeat_interleave(curr_batch, train_image_class_distribution.shape[0], dim=0)    
        q = train_image_class_distribution
        q_repeated = torch.cat([q]*bs)
        kl = repeated_batch * (repeated_batch.log() - q_repeated.log())
        kl = kl.sum(dim=-1)
        kl = kl.view(bs, -1)
        kl_divs_sim[ i*bs : (i+1)*bs , : ] = kl  

    return kl_divs_sim