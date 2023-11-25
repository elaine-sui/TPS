import torch
import torch.nn as nn

class ClipImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super(ClipImageEncoder, self).__init__()
        self.encoder = clip_model

        self.ln_post_weight_init = clip_model.ln_post.weight.detach().clone()
        self.ln_post_bias_init = clip_model.ln_post.bias.detach().clone()

    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype
    
    def reset(self):
        self.encoder.ln_post.weight.copy_(self.ln_post_weight_init)
        self.encoder.ln_post.bias.copy_(self.ln_post_bias_init)


    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        return x


class ImageEncoderWithPrompt(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.input_resolution = clip_model.input_resolution
        self.output_dim = clip_model.output_dim
        self.conv1 = clip_model.conv1
        self.class_embedding = clip_model.class_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_pre = clip_model.ln_pre
        self.transformer = clip_model.transformer
        self.ln_post = clip_model.ln_post
        self.proj = clip_model.proj
    
    @property
    def dtype(self):
        return self.conv1.weight.dtype
    
    def get_tokens(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        return x

    def forward(self, x: torch.Tensor, prompts: torch.Tensor):
        # Code adapted from https://github.com/openai/CLIP/blob/main/clip/model.py

        x = self.get_tokens(x)

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        prompts = prompts.expand(x.shape[0], -1, -1)

        # Prepend the tuneable prompt ((batch_size, cls_token + n_prompt + n_patches, hidden_dim))
        x = torch.cat((
            x[:, :1, :],
            prompts,
            x[:, 1:, :]
        ), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x