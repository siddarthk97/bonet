"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, input_dim, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.input_dim = input_dim
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head
        self.ctr = 0

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTDiscrete(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.heads = []
        self.test = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        # for _ in range(config.input_dim):
        #     self.heads.append(nn.Linear(config.n_embd, config.vocab_size, bias=True).cuda())
        # self.head = nn.Linear(config.n_embd, config.input_dim, bias=True)
        self.head = nn.Linear(config.n_embd, config.input_dim * config.vocab_size, bias=True)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd, bias=True), nn.Tanh())
        self.point_embeddings = nn.Sequential(nn.Linear(config.input_dim, config.n_embd, bias=True), nn.Tanh())

        nn.init.normal_(self.point_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, points, targets=None, rtgs=None, timesteps=None, values=None):
        # points: (batch, block_size, dim)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)
        b, t, _ = rtgs.size()

        if points is not None:
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32)).to(points.device)
            point_embeddings = self.point_embeddings(points.type(torch.float32)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((b, t*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=point_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings
            token_embeddings[:,1::2,:] = point_embeddings[:,-t + int(targets is None):,:]
        else: # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            token_embeddings = rtg_embeddings

        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, b, dim=0) # batch_size, traj_length, n_embd

        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        # position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        # x = self.drop(token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        # preds = self.head(x)
        logit_preds = self.head(x)
        # logits = []
        # for k in range(len(self.heads)):
        #     logits.append(self.heads[k](x))

        if points is not None:
            # logits = [lgts[:,::2,:] for lgts in logits]
            logit_preds = logit_preds[:, ::2, :]
            # preds = preds[:, ::2, :] # only keep predictions from regret_embeddings
        else:
            logit_preds = logit_preds
            # preds = preds

        # if we are given some desired targets also calculate the loss

        # print("logits shape", logit_preds.shape)
        loss = None
        logits_shape = logit_preds.shape
        logits = []
        # for i in range(self.config.input_dim):
        #     logits.append(logit_preds[:, :, i * self.config.vocab_size : (i + 1) * self.config.vocab_size])
        logits = logit_preds.reshape(logits_shape[0], logits_shape[1], self.config.input_dim, self.config.vocab_size)
        # print("logits shapeee", logit_preds.min(), logit_preds.max())

        # logits = logits.reshape(logits_shape[0], logits_shape[1], self.config.input_dim, self.config.vocab_size)
        # print("shapee: ", logits[0].shape)
        if targets is not None:
            loss = 0
            # print(logits[0].shape)
            
            for k in range(logits.shape[-2]):
                # print(logits[k].reshape(-1, logits[k].size(-1)).shape)
                # print("=============")
                # print(targets[:,:,k].reshape(-1).shape)
                # loss += F.cross_entropy(logits[k].reshape(-1, logits[k].size(-1)), targets[:,:,k].reshape(-1).long())
                # print(logits[:, :, k, :].reshape(-1, logits[:, :, k, :].size(-1)).shape, targets[:,:,k].reshape(-1).long().shape)
                loss += F.cross_entropy(logits[:, :, k, :].reshape(-1, logits[:, :, k, :].size(-1)), targets[:,:,k].reshape(-1).long())
                # print(logits.min(), logits.max())
                # print(loss)


        # print(logits.shape, loss)
        return logits, loss

    @torch.no_grad()
    def evaluate(self, rtg, unroll_length, function, device='cpu', update_regret=True, initial_points=None, initial_rtgs=None, mean=False):
        """
        Return tuple of lists of ([points], [regrets])
        """

        # set to evaluation mode
        self.eval()


        rtgs, points, timesteps, ret = [rtg], [], [0], []
        K = 0

        if torch.is_tensor(initial_points):
            initial_points = initial_points.to(device)
            initial_rtgs = initial_rtgs.to(device)
            K = len(initial_points)
            # initial_rtg = initial_rtgs[0] + rtg
            # rtgs = [initial_rtg] + list(initial_rtgs[:].cpu())  #TODO batching
            initial_rtgs = list(initial_rtgs[:].cpu() + rtg) + [rtg]
            initial_points = list(initial_points[:, :])
            rtgs[0] = initial_rtgs[0]

        # print("inishaep:",  initial_points[0].shape)
        for i in range(unroll_length):
            pts = None
            if points:
                pts = torch.stack(points).view(1, -1, self.config.input_dim).to(device)

            ret_to_go = torch.tensor(rtgs).view(1, -1, 1).type(torch.float32).to(device)
            ts = torch.Tensor([timesteps[-1]]).view(1, -1, 1).type(torch.int64).to(device)

            out_preds, _ = self.forward(points=pts if points else None, rtgs=ret_to_go, timesteps=ts)

            # convert preds to a array
            preds = []
            for k in range(self.config.input_dim):
                preds.append(out_preds[:, :, k, :])
                
            current_preds = [p[:, -1, :] for p in preds]
            q = []
            for pred in current_preds:
                _, u = torch.topk(pred, k=1, dim=-1)
                qq = int(u)
                q.append(qq)

            q = np.asarray(q)
            current_regret = float(function.regret(q.reshape(1, -1)))

            # print(q.shape)

            if i >= K:
                ret.append((q, current_regret))
                points.append(torch.tensor(q, dtype=torch.float32).to(device))

                if update_regret and not mean:
                    if rtgs[-1] - current_regret < 0:
                        rtgs.append(rtgs[-1])
                    else:
                        rtgs.append(rtgs[-1] - current_regret)
                elif update_regret:
                    # print("gayaa")
                    saved = rtgs[-1]
                    if i < 127:
                        new_rtg = (saved * (128 - timesteps[-1]) - current_regret) / (127 - timesteps[-1])
                        if new_rtg < 0:
                            new_rtg = saved
                        rtgs.append(new_rtg)
                    else:
                        # print("gayuyiuayiuyiuy")
                        rtgs.append(saved)
                else:
                    rtgs.append(rtgs[-1])
            else:
                # ret.append((initial_points[i], initial_rtgs[i+1]))
                points.append(initial_points[i])
                rtgs.append(initial_rtgs[i+1])

            # ret.append((q, current_regret))

            # points.append(torch.tensor(q, dtype=torch.float32))
            # print("ppp", torch.tensor(q, dtype=torch.float32).shape)
            # if update_regret:
            #     rtgs.append(rtgs[-1] - current_regret)
            # else:
            #     rtgs.append(rtgs[-1])

            timesteps.append(timesteps[-1] + 1)

            context_length = self.config.block_size // 2
            points = points[-context_length:]
            rtgs = rtgs[-context_length:]
            timesteps = timesteps[-context_length:]

            if rtgs[-1] < 0:
                print("WARN: RTG less than zero in unroll number", i)

        return [point for point, _ in ret], [regret for _, regret in ret]
