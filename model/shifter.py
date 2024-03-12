import torch
import torch.nn as nn

class Shifter(nn.Module):
    def __init__(self, 
            embed_dim=512, 
            dtype=None, 
            do_shift=True, 
            do_scale=False,
            num_classes=1000,
            per_label=False,
            text_embeds=None,
            num_components=-1,
            class2concepts=None,
            device=None
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.per_label = per_label
        self.num_classes = num_classes

        self.dtype = dtype
        self.device = device
        self.do_shift = do_shift
        self.do_scale = do_scale

        self.num_classes_per_concept = [len(v) for _,v in class2concepts.items()] if class2concepts is not None else None

        self.text_embeds = text_embeds

        if num_components == -1:
            num_components = num_classes
        self.num_components = num_components

        if per_label:
            shift_init = torch.zeros((num_classes, embed_dim), dtype=self.dtype)
            scale_init = torch.ones((num_classes, embed_dim), dtype=self.dtype)
        else:
            shift_init = torch.zeros(embed_dim, dtype=self.dtype)
            scale_init = torch.ones(embed_dim, dtype=self.dtype)

        if self.do_shift:
            self.shift_init_state_original = shift_init.detach().clone()
            self.shift_init_state = shift_init.detach().clone()
            self.shift = nn.Parameter(shift_init)

        if self.do_scale:
            self.scale_init_state_original = scale_init.detach().clone()
            self.scale_init_state = scale_init.detach().clone()
            self.scale = nn.Parameter(scale_init)


    def reset(self):
        if self.do_shift:
            self.shift.copy_(self.shift_init_state)
        
        if self.do_scale:
            self.scale.copy_(self.scale_init_state)

    def forward(self, img_embed, test=False):
        x = img_embed

        if self.do_scale:
            x = self.scale * x

        if self.do_shift:
            if test or self.num_classes_per_concept is None:
                x += self.shift
            else: # need to expand per concept
                shift = [self.shift[i].repeat(n, 1) for i,n in enumerate(self.num_classes_per_concept)]
                shift = torch.cat(shift, dim=0)

                x += shift
                
        return x
        