import torch
import torch.nn as nn

class Shifter(nn.Module):
    def __init__(self, 
            embed_dim=512, 
            dtype=None, 
            do_shift=True, 
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

        self.num_classes_per_concept = [len(v) for _,v in class2concepts.items()] if class2concepts is not None else None

        self.text_embeds = text_embeds

        if num_components == -1:
            num_components = num_classes
        self.num_components = num_components

        if per_label:
            shift_init = torch.zeros((num_classes, embed_dim), dtype=self.dtype)
        else:
            shift_init = torch.zeros(embed_dim, dtype=self.dtype)

        if self.do_shift:
            self.shift_init_state_original = shift_init.detach().clone()
            self.shift_init_state = shift_init.detach().clone()
            self.shift = nn.Parameter(shift_init)

    
    def reset(self):
        if self.do_shift:
            self.shift.copy_(self.shift_init_state)

    def update(self, weight, entropy):
        if self.do_shift:
            grad_norm = self.shift._grad.norm()
            if (self.register_counter < self.update_batch_size) and (entropy < self.entropy_thresh) and (grad_norm < self.grad_norm_thresh):
                self.add_to_register()
                self.avg_entropy_batch += entropy
            
            if self.register_counter == self.update_batch_size:
                print("Update shift EMA")
                self.update_entropy_ema(self.avg_entropy_batch / self.update_batch_size)
                print(f"Entropy EMA: {self.entropy_ema}")
                self.update_shift_init_ema(weight)
                self.reset_shift_register()
                self.avg_entropy_batch = 0.
    
    def add_to_register(self):
        shift_vector = self.shift.detach().clone()
        self.shift_register[self.register_counter] = shift_vector
        self.register_counter += 1

    def update_entropy_ema(self, entropy, weight=0.9):
        if self.entropy_ema is None:
            self.entropy_ema = entropy
        else:
            self.entropy_ema = weight * self.entropy_ema + (1 - weight) * entropy

    def reset_shift_register(self):
        self.shift_register = torch.empty(self.update_batch_size, self.num_classes, self.embed_dim)
        self.register_counter = 0
    
    def update_shift_init_ema(self, weight):
        print("Update shift init state")
        shift_batch_mean = self.shift_register.mean(dim=0)
        self.shift_init_state = weight * self.shift_init_state + (1 - weight) * shift_batch_mean
        self.shift.copy_(self.shift_init_state)
        # print(f"Shift norm (per class): {self.shift.norm(dim=-1)}")
        
    def save_current_state(self):
        if self.do_shift:
            self.prev_shift = self.shift.detach().clone()

    def forward(self, img_embed):
        x = img_embed

        if self.do_shift:
            if self.num_classes_per_concept is None:
                x += self.shift
            else: # need to expand per concept
                shift = [self.shift[i].repeat(n, 1) for i,n in enumerate(self.num_classes_per_concept)]
                shift = torch.cat(shift, dim=0)

                x += shift
                
        return x
        