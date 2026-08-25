import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from functools import partial
import contextlib
from dataclasses import dataclass, asdict, fields
import wandb


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
# Use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None, **kwargs):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7, **kwargs):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

def ns_conv(G, steps = 20, eps=1e-7, **kwargs):
    assert len(G.shape) == 2
    a, b, c = (3, -3.2,  1.2)
    X = G.float()
    X = X / (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A 
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

def targeted_conv(G, steps = 20, tau: float = 1e-3, top=True):
    assert G.ndim >= 2
    X = G.float()
    if G.size(-1) > G.size(-2):
        X = X.mT
    n = X.size(-1)
    I = torch.eye(n, dtype=X.dtype, device=X.device) 
    M = X.mT @ X - tau**2 * I
    signedM = ns_conv(M, steps)
    projBot = 0.5 * (I - signedM)
    projTop = 0.5 * (I + signedM)
    nsX = ns_conv(X, steps)
    if top:
        tgted = X @ projBot + nsX @ projTop
    else:
        tgted = nsX @ projBot + X @ projTop
    if G.size(-1) > G.size(-2):
        tgted = tgted.mT
    return tgted


def identity(G, steps:int = 3, **kwargs):
    return G



def compute_effective_rank(svds):
    nuc = svds.sum()
    if(nuc == 0):
        return torch.tensor(0.0, device=svds.device)
    p = svds/nuc
    p = p[p > 0]
    return torch.exp((-p*torch.log(p)).sum())


@torch.no_grad()
def make_record(step_num, i, W, g_raw, g_mom, u, prev_topU, tol=1e-8, k=32):
    # shared spectral record for one matrix: used by Muon's in-step hook and the
    # main-loop tracker for torch optimizers, so all runs log identical quantities
    rec = {'step': step_num, 'i': i, 'shape': tuple(W.shape)}
    W = W.float()
    Us, S, Vh = torch.linalg.svd(W, full_matrices=False)
    rec['p_spec'] = S.cpu()
    rec['g_spec'] = torch.linalg.svdvals(g_raw.float()).cpu()
    rec['m_spec'] = torch.linalg.svdvals(g_mom.float()).cpu()
    u = u.float()
    rec['u_spec'] = torch.linalg.svdvals(u).cpu()
    newU = prev_topU
    if S[0] > tol:
        d = torch.einsum('ir,ir->r', Us, u @ Vh.mH)     # d_r = u_r^T Z v_r
        rec['drift_diag'] = d.cpu()
        rec['tangent_frac'] = (1 - d.pow(2).sum() / u.pow(2).sum()).item()
        rec['radial_coef'] = ((u * W).sum() / W.pow(2).sum()).item()
        kk = min(k, Us.size(1))
        if prev_topU is not None:
            rec['rot_overlap'] = (prev_topU.mH @ Us[:, :kk]).pow(2).sum().div(kk).item()
        newU = Us[:, :kk].clone()
    else:
        rec['skipped'] = True
    return rec, newU


zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5, targeted=targeted_conv, identity=identity, conv=ns_conv)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='conv', backend_steps=20, tau=0.0, arm='top',
                 rms_match=False, track_every=25, track_dir=None, track_k=32, track_tol=1e-8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend,
                        backend_steps=backend_steps, tau=tau, arm=arm, rms_match=rms_match)
        self._step = 0
        self.track_every, self.track_dir = track_every, track_dir
        self.track_k, self.track_tol = track_k, track_tol
        self.records = []
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
        for group in self.param_groups:
            lr, momentum = group['lr'], group['momentum']
            zb = zeropower_backends[group['backend']]
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            track = self.track_every > 0 and self._step % self.track_every == 0
            curr_idx = 0
            for i, p in enumerate(group['params']):
                if i % world == rank:
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    mg = g.add(buf, alpha=momentum) if group['nesterov'] else buf.clone()

                    u = zb(mg, steps=group['backend_steps'],
                           tau=group['tau'], top=(group['arm'] == 'top')).float()
                    pre_rms = u.pow(2).mean().sqrt()
                    if group['rms_match']:
                        u = u * ((1.0 / max(u.shape) ** 0.5) / (pre_rms + 1e-12))
                    else:
                        u = u * max(1, u.size(0) / u.size(1)) ** 0.5   # legacy scale

                    if track:
                        rec, state['topU'] = make_record(self._step, i, p, g, mg, u,
                                                         state.get('topU'), self.track_tol, self.track_k)
                        rec['pre_rms'] = pre_rms.item()
                        self.records.append(rec)

                    updates_flat[curr_idx:curr_idx + p.numel()] = u.flatten()
                curr_idx += p.numel()

            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr_idx = 0
            for p in group['params']:
                s = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(s, alpha=-lr)
                curr_idx += p.numel()

        self._step += 1
        if self.track_dir and self.records and self._step % (10 * self.track_every) == 0:
            self.flush()

    def flush(self):
        if self.track_dir and self.records:
            r = int(os.environ.get('RANK', 0))
            torch.save(self.records, os.path.join(self.track_dir, f'spectra_rank{r}.pt'))

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, dim, n_head):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        # value residual lambda
        self.lamb = nn.Parameter(torch.tensor(0.5)) # @Grad62304977
        # rotary embeddings
        self.rotary = Rotary(dim // n_head) # dim // n_head = head_dim
        # output projection
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x, vi, block_mask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q = self.c_q(x).view(B, T, self.n_head, -1)
        k = self.c_k(x).view(B, T, self.n_head, -1)
        v = self.c_v(x).view(B, T, self.n_head, -1)
        v = (1 - self.lamb) * v + self.lamb * vi.view_as(v) # @Grad62304977
        q, k = norm(q), norm(k) # QK norm suggested by @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config.n_embd, config.n_head)
        self.mlp = MLP(config.n_embd)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, vi, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), vi, block_mask)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        # U-net design by @brendanh0gan
        self.num_encoder_layers = config.n_layer // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
            vte = nn.Embedding(config.vocab_size, config.n_embd*12),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.lm_head.weight.data.zero_() # @Grad62304977

    def forward(self, idx, target, attn_blocksize):

        docs = (idx == 50256).cumsum(0)
        def document_causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          document_mask = docs[q_idx] == docs[kv_idx]
          window_mask = q_idx - kv_idx < attn_blocksize
          return causal_mask & document_mask & window_mask

        S = len(idx)
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

        # forward the GPT model itself
        x = self.transformer.wte(idx[None]) # token embeddings of shape (b, t, n_embd)
        x = norm(x) # @Grad62304977
        x0 = x
        vi = self.transformer.vte(idx[None]).chunk(12, dim=-1)

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x = self.transformer.h[i](x, vi[i], x0, block_mask)
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.transformer.h[self.num_encoder_layers + i](x, vi[self.num_encoder_layers+i], x0, block_mask)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        batch_size = self.T * self.num_processes
        buf = self.tokens[self.current_position:self.current_position+self.T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1] # inputs
        y = buf[1:] # targets
        # advance current position and load next shard if necessary
        self.current_position += batch_size
        if self.current_position + batch_size >= len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    optim : str = "Muon"
    batch_size : int = 8 # batch size, in sequences, across all devices
    sequence_length : int = 64*1024 # sequence length, in tokens
    num_iterations : int = 1530 # number of iterations to run
    warmup_iters : int = 250
    cooldown_iters : int = 600 # number of iterations of linear warmup/cooldown for triangular or trapezoidal schedule
    weight_decay : float = 0.01 # matrix-optimizer decoupled wd; 0.01 = torch AdamW default the baseline has always run
    muon_lr : float = .05
    backend : str = "newtonschulz5"
    backend_steps : int = 20
    tau : float = 0.0
    arm : str = "top"
    rms_match : bool = True
    seed : int = 0
    track_every : int = 25 # spectral tracking cadence in steps; 0 disables
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()

# minimal --key=value CLI overrides, cast to the dataclass field types
_ftypes = {f.name: f.type for f in fields(Hyperparameters)}
for _a in sys.argv[1:]:
    assert _a.startswith('--') and '=' in _a, f"bad arg {_a}, expected --key=value"
    _k, _v = _a[2:].split('=', 1)
    assert _k in _ftypes, f"unknown hyperparameter {_k}"
    _t = _ftypes[_k]
    setattr(args, _k, _v.lower() in ('1', 'true') if _t == bool else _t(_v))

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write(code)
        f.write('='*100 + '\n')
def print0(s, logonly=False):
    if master_process:
        with open(logfile, "a") as f:
            if not logonly:
                print(s)
            f.write(s+'\n')
# log information about the hardware/software environment this is running on
# and print the full `nvidia-smi` to file
print0(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:")
import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}', logonly=True)
print0('='*100, logonly=True)

# convenience variables
T = args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (T * ddp_world_size) == 0
val_steps = args.val_tokens // (T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (ddp_world_size) == 0
train_accumulation_steps = args.batch_size // ddp_world_size

# load tokens
train_loader = DistributedDataLoader(args.input_bin, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, T, ddp_rank, ddp_world_size)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
print0('='*100, logonly=True)
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model

run = None
if master_process:
    _group = f"{args.backend}-{args.arm}-tau{args.tau:g}" if args.backend == 'targeted' else args.backend
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "spectral-muon-rerun"),
        group=_group,
        name=f"{_group}-s{args.seed}",
        config=asdict(args) | {"world_size": ddp_world_size, "run_id": run_id},
    )


# init the optimizer(s)
optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight, raw_model.transformer.vte.weight], lr=0.6, betas=(0.8, 0.95), fused=True)
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=0.008, betas=(0.8, 0.95), fused=True)
params = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
# names for per-layer wandb series: "0.attn.c_q.weight" -> layer "0", type "attn.c_q"
matrix_names = [n for n, p in raw_model.transformer.h.named_parameters() if p.ndim == 2]
matrix_layers = [n.split('.')[0] for n in matrix_names]
matrix_types = ['.'.join(n.split('.')[1:-1]) for n in matrix_names]
if (args.optim == "Muon"):
    print("Using Muon")
    optimizer3 = Muon(matrix_params, lr=args.muon_lr, backend=args.backend, momentum=0.95,
                      backend_steps=args.backend_steps, tau=args.tau, arm=args.arm,
                      rms_match=args.rms_match, track_every=args.track_every,
                      track_dir=(logdir if master_process else None))
else:
    print("Using AdamW")
    optimizer3 = torch.optim.AdamW(matrix_params, lr=0.0018, betas=(0.9, 0.95), weight_decay=args.weight_decay)
optimizer4 = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.8, 0.95), fused=True) # note that this learning rate is neither sensitive nor tuned
optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
# learning rate decay scheduler (linear warmup and cooldown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    # 3) linear cooldown
    else:
        decay_ratio = (args.num_iterations - it) / args.cooldown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# tracking state for non-Muon matrix optimizers (e.g. AdamW): Muon tracks inside its own step
nonmuon_records, nonmuon_topU = [], {}

# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # Set the attention blocksize for the current step, in chunks of 64. By @fernbear.bsky.social
    attn_blocksize = torch.tensor(64*((step/args.num_iterations * (1792 - 64) + 64)//64), dtype=torch.int, device='cuda')

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_loss += model(x_val, y_val, attn_blocksize=attn_blocksize)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if run is not None:
            run.log({"val_loss": val_loss}, step=step)
        print0(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        ctx = model.no_sync() if i < train_accumulation_steps else contextlib.nullcontext()
        with ctx: # there's no need to sync gradients every accumulation step
            # forward pass
            loss = model(x, y, attn_blocksize=attn_blocksize)
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            loss.backward()
        train_loss = loss.detach()
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    """if step % 100 == 0:
        p_origs = []
        for i, p in enumerate(matrix_params):
            p_svds = torch.linalg.svdvals(p.detach().float())
            run.log({f"p_erank{i}": compute_effective_rank(p_svds).item()}, step=step)
            g_svds = torch.linalg.svdvals(p.grad.detach().float())
            run.log({f"g_erank{i}": compute_effective_rank(g_svds).item()}, step=step)
            p_origs.append(p.detach().clone())"""

        
    # momentum warmup for Muon
    frac = min(step/300, 1)
    optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    # snapshot for non-Muon tracking: lr must be read before sched.step() advances it
    track_now = (not isinstance(optimizer3, Muon)) and master_process \
                and args.track_every > 0 and step % args.track_every == 0
    if track_now:
        lr3 = optimizer3.param_groups[0]['lr']
        wd3 = optimizer3.param_groups[0].get('weight_decay', 0.0)
        p_prev = [p.detach().float().clone() for p in matrix_params]
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    if track_now:
        with torch.no_grad():
            for i, p in enumerate(matrix_params):
                # decoupled update: p_new = p_prev - lr*wd*p_prev - lr*dir  =>  recover dir exactly
                u = (p_prev[i] - p.detach().float()) / lr3 - wd3 * p_prev[i]
                m = optimizer3.state[p].get('exp_avg', torch.zeros_like(p))
                rec, nonmuon_topU[i] = make_record(step, i, p_prev[i], p.grad, m, u, nonmuon_topU.get(i))
                nonmuon_records.append(rec)
        if step % (10 * args.track_every) == 0:
            torch.save(nonmuon_records, os.path.join(logdir, 'spectra_rank0.pt'))
    # unified wandb logging: per-matrix scalars (per layer) plus per-type averages, one log call
    if run is not None and args.track_every > 0 and step % args.track_every == 0:
        source = optimizer3.records if isinstance(optimizer3, Muon) else nonmuon_records
        recs = [r for r in source if r['step'] == step]
        metrics, avgs = {}, {}
        for r in recs:
            t, L = matrix_types[r['i']], matrix_layers[r['i']]
            vals = {f'erank_{k}': compute_effective_rank(r[f'{k}_spec']).item() for k in ('p', 'g', 'm', 'u')}
            vals['frac_above_tau'] = (r['m_spec'] > args.tau).float().mean().item()
            for key in ('tangent_frac', 'radial_coef', 'rot_overlap', 'pre_rms'):
                if key in r:
                    vals[key] = r[key]
            for k, v in vals.items():
                metrics[f'{k}/{t}/L{L}'] = v
                avgs.setdefault(f'{k}/{t}/avg', []).append(v)
        if metrics:
            metrics |= {k: sum(v) / len(v) for k, v in avgs.items()}
            run.log(metrics, step=step)
    # null the gradients
    model.zero_grad(set_to_none=True)
    """if step % 100 == 0:
        for i, p in enumerate(matrix_params):
            update = p.detach() - p_origs[i]
            u_svds = torch.linalg.svdvals(update.detach().float())
            run.log({f"u_erank{i}": compute_effective_rank(u_svds).item()}, step=step)"""
            

    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    approx_time = training_time_ms + 1000 * (time.time() - t0)
    if run is not None:
        run.log({"train_loss": train_loss.item()}, step=step)
    print0(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

if master_process:
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
if isinstance(optimizer3, Muon):
    optimizer3.flush()
elif master_process and nonmuon_records:
    torch.save(nonmuon_records, os.path.join(logdir, 'spectra_rank0.pt'))
if run is not None:
    _sp = os.path.join(logdir, 'spectra_rank0.pt')
    if os.path.exists(_sp):
        run.save(_sp, base_path='logs')

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
