import torch
import torch.nn as nn

from model import tokenize

class ProtoTextPromptLearner(nn.Module):
    def __init__(self, 
        clip_model, 
        classnames, 
        batch_size=None, 
        n_ctx=16, 
        ctx_init=None, 
        ctx_position='end', 
        learned_cls=False, 
        init_concepts=False, 
        class2concepts=None, 
        per_label=False, 
        optimize=True
    ):
        super().__init__()
        
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype 
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        self.class2concepts = class2concepts # {classname: [list of concepts]}
        self.n_cls = n_cls
        self.init_concepts = init_concepts
        self.per_label = per_label
        self.optimize = optimize
        
        self.clip_model = clip_model

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the context with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.n_ctx = n_ctx

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()

        self.classnames = None
        self.reset_classnames(classnames)

        self.num = self.n_concepts if (self.init_concepts) else self.n_cls

        if self.per_label:
            ctx_vectors = ctx_vectors.repeat(self.num, 1, 1) # (num classes, L, D)

        if self.optimize:
            self.ctx = nn.Parameter(ctx_vectors) # to be optimized
        else:
            self.ctx = ctx_vectors

        self.ctx_init = ctx_init

        self.class_token_position = ctx_position
        self.n_ctx = n_ctx
        
    def reset(self):
        ctx_vectors = self.ctx_init_state

        if self.per_label:
            ctx_vectors = ctx_vectors.repeat(self.num, 1, 1) # (num classes, L, D)

        if self.optimize:
            self.ctx.copy_(ctx_vectors) # to be optimized
        else:
            self.ctx = ctx_vectors

    def update(self, weight, vector, ema=True):
        ctx_vectors = self.ctx

        if self.per_label:
            if ema:
                ctx_vectors[self.indices_kept] = (1 - weight) * vector + weight * ctx_vectors[self.indices_kept]
            else:
                ctx_vectors[self.indices_kept] = ctx_vectors[self.indices_kept] * vector
        else:
            ctx_vectors = (1 - weight) * vector + weight * ctx_vectors

        if self.optimize:
            self.ctx.copy_(ctx_vectors) # to be optimized
        else:
            self.ctx = ctx_vectors

    def reset_classnames(self, classnames):
        self._reset_classnames(classnames)

        if self.init_concepts:
            self._reset_concepts(classnames)
        
        self.num = self.n_concepts if (self.init_concepts) else self.n_cls


    def _reset_concepts(self, classnames):
        all_concepts = []
        for name in classnames:
            concepts = self.class2concepts[name]
            all_concepts.extend(concepts)

        self.num_concepts_per_class = [len(self.class2concepts[name]) for name in classnames]

        concept_prompts = [self.prompt_prefix + " " + concept + "." for concept in all_concepts]
        self.n_concepts = len(concept_prompts)

        tokenized_concept_prompts = torch.cat([tokenize(p, truncate=True) for p in concept_prompts]).to(self.device)

        with torch.no_grad():
            concept_embedding = self.clip_model.token_embedding(tokenized_concept_prompts).type(self.dtype)
        
        self.concept_token_prefix = concept_embedding[:, :1, :]
        self.concept_token_suffix = concept_embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.tokenized_concept_prompts = tokenized_concept_prompts

    def _reset_classnames(self, classnames):
        self.n_cls = len(classnames)
        classnames = [name.replace("_", " ") for name in classnames]

        if self.classnames is not None and len(set(classnames) - set(self.classnames)) == 0:
            self.indices_kept = [self.classnames.index(c) for c in classnames]

            self.class_token_prefix = self.class_token_prefix[self.indices_kept]
            self.class_token_suffix = self.class_token_suffix[self.indices_kept]
            self.tokenized_class_prompts = self.tokenized_class_prompts[self.indices_kept]
        else:
            self.indices_kept = torch.arange(len(classnames))
            class_prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

            tokenized_class_prompts = torch.cat([tokenize(p) for p in class_prompts]).to(self.device)

            with torch.no_grad():
                class_embedding = self.clip_model.token_embedding(tokenized_class_prompts).type(self.dtype)

            self.class_token_prefix = class_embedding[:, :1, :]
            self.class_token_suffix = class_embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

            self.tokenized_class_prompts = tokenized_class_prompts
        
        self.classnames = classnames
    
    def get_tokenized_prompts(self, override=None):
        # if not stage1 and self.init_concepts:
        if override == 'class' or not self.init_concepts:
            return self.tokenized_class_prompts
        elif override == 'concept' or self.init_concepts:
            return self.tokenized_concept_prompts
        
        return None # shouldn't get here


    def forward(self, init=None, override=None):
        # the init will be used when computing CLIP directional loss
        # num = self.n_concepts if (not stage1 and self.init_concepts) else self.n_cls
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.num, -1, -1)

        if override == 'class' or not self.init_concepts:
            prefix = self.class_token_prefix
            suffix = self.class_token_suffix
        elif override == 'concept' or self.init_concepts:
            prefix = self.concept_token_prefix
            suffix = self.concept_token_suffix

        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=-2,
            )
        else:
            raise ValueError

        return prompts