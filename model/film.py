import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, 
            embed_dim=512, 
            dtype=None, 
            num_classes=1000,
            per_label=False,
            text_embeds=None,
            device=None
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.per_label = per_label
        self.num_classes = num_classes

        self.dtype = dtype
        self.device = device

        self.text_embeds = text_embeds

        if self.per_label:
            self.fc_scale = nn.Linear(self.num_classes * self.embed_dim, self.num_classes * self.embed_dim).to(self.device)
            self.fc_shift = nn.Linear(self.num_classes * self.embed_dim, self.num_classes * self.embed_dim).to(self.device)
        else:
            self.fc_scale = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)
            self.fc_shift = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)

        self.fc_scale_weight_init = self.fc_scale.weight

    def reset(self):
        self.fc_scale.reset_parameters()
        self.fc_shift.reset_parameters()
        
    def forward(self, img_embed, test=False):
        x = img_embed

        if self.per_label:
            x = x.flatten()
        
        scale = self.fc_scale(x)
        shift = self.fc_shift(x)

        x = x * scale + shift

        if self.per_label:
            x = x.reshape(self.num_classes, -1)
                
        return x
        