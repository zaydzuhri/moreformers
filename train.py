import os
import time
import torch
import pickle
import wandb
import argparse
import json
import math
import numpy as np
from models.gpt import GPTConfig, GPT
from models.gpt_modes import GPTModes
from models.fadeformer_linear import FadeFormerLinear
from models.fadeformer_rank import FadeFormerRank
from models.fadeformer_static import FadeFormerStatic
from models.fadeformer_stagger import FadeFormerStagger
from models.fadeformer_half import FadeFormerHalf
from models.fadeformer_pool import FadeFormerPool
from models.fadeformer_trans import FadeFormerTrans
from models.fadeformer_cut import FadeFormerCut
from models.fadeformer_even import FadeFormerEven
from contextlib import nullcontext
from tqdm import tqdm

# I/O
init_from = 'scratch'
out_dir = 'out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# Eval parameters
eval_interval = 1000
log_interval = 1
eval_iters = 200
eval_only = False # set to true to only run evaluation
always_save = True # set to true to always save the model after evaluation
# wandb logging
wandb_log = True
wandb_project = 'fadeformer'
wandb_run_name = 'gpt2-testing'
# data
dataset = 'shakespeare'
batch_size = 16
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
# model
model_type = 'gpt' # 'gpt' or 'fadeformer'
model_name = 'gpt2'
new_model_name = 'fadeformer'
pretrain = False
ctx_size = 1024
target_size = ctx_size # for fadeformer
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# optimizer
lr = 6e-4 # max lr
max_iters = 600000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value
# scheduler
warmup_iters = 5000
decay_lr = True
lr_decay_iters = 600000
min_lr = 6e-5
# system
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
compile = False # change when in linux for pytorch 2.0
# torch
torch.manual_seed(69)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ctx = nullcontext() if device.type == 'cpu' else torch.cuda.amp.autocast(dtype=dtype)
# override globals with json file
argparser = argparse.ArgumentParser()
argparser.add_argument('--config', type=str, default=None, help='json file name in configs folder')
args = argparser.parse_args()
if args.config is not None:
    with open(os.path.join('configs', args.config+'.json'), 'r') as f:
        config = json.load(f)
    for k,v in config.items():
        globals()[k] = v
#--------------------------------------------------------------------------------

# Poor man's data loader
data_dir = os.path.join('data', dataset)
# Load data from prepared .bin files (see prepare.py in data folder)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
# get_batch samples batches of blocks of tokens randomly from the data and returns x input and y target sequences
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # get random indices for each batch
    ix = torch.randint(len(data) - ctx_size, (batch_size,))
    # x is the input sequence, y is the target sequence which is x shifted by 1
    x = torch.stack([torch.from_numpy((data[i:i+ctx_size]).astype(np.int64)) for i in ix])
    if model_type == 'gpt' or model_type == 'gpt-modes' or model_type == 'fadeformer-even':
        y = torch.stack([torch.from_numpy((data[i+1:i+1+ctx_size]).astype(np.int64)) for i in ix])
    elif model_type == 'fadeformer-linear':
        y = torch.stack([torch.from_numpy((data[i+1:i+1+target_size]).astype(np.int64)) for i in ix])
    elif model_type == 'fadeformer-rank':
        target_size = int(ctx_size // (2**n_layer))
        y = torch.stack([torch.from_numpy((data[i+1+(ctx_size-target_size):i+1+ctx_size]).astype(np.int64)) for i in ix])
    else:
        target_size = int(ctx_size // (2**(n_layer-2)))
        y = torch.stack([torch.from_numpy((data[i+1+(ctx_size-target_size):i+1+ctx_size]).astype(np.int64)) for i in ix])
    if device.type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# model init
model_args = dict(
    n_layer=n_layer, 
    n_head=n_head, 
    n_embd=n_embd, 
    ctx_size=ctx_size,
    bias=bias, 
    vocab_size=None, 
    dropout=dropout
) # start with model_args from globals

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# override later when loading from a checkpoint
iter_num = 0
best_val_loss = 9e9

if init_from == 'scratch':
    print('Initializing model from scratch')
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    if model_type == 'gpt':
        model = GPT(gptconf)
    elif model_type == 'gpt-modes':
        model = GPTModes(gptconf)
    elif model_type == 'fadeformer-linear':
        model = FadeFormerLinear(gptconf)
    elif model_type == 'fadeformer-rank':
        model = FadeFormerRank(gptconf)
    elif model_type == 'fadeformer-static':
        model = FadeFormerStatic(gptconf)
    elif model_type == 'fadeformer-stagger':
        model = FadeFormerStagger(gptconf)
    elif model_type == 'fadeformer-half':
        model = FadeFormerHalf(gptconf)
    elif model_type == 'fadeformer-pool':
        model = FadeFormerPool(gptconf)
    elif model_type == 'fadeformer-trans':
        model = FadeFormerTrans(gptconf)
    elif model_type == 'fadeformer-cut':
        if pretrain:
            model = FadeFormerCut(gptconf, pretrain=True)
        else:
            model = FadeFormerCut(gptconf)
    elif model_type == 'fadeformer-even':
        model = FadeFormerEven(gptconf)
elif init_from == 'continue':
    print('Continuing from checkpoint')
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, model_name+'.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    if model_type == 'gpt':
        model = GPT(gptconf)
    elif model_type == 'gpt-modes':
        model = GPTModes(gptconf)
    elif model_type == 'fadeformer-linear':
        model = FadeFormerLinear(gptconf)
    elif model_type == 'fadeformer-rank':
        model = FadeFormerRank(gptconf)
    elif model_type == 'fadeformer-static':
        model = FadeFormerStatic(gptconf)
    elif model_type == 'fadeformer-stagger':
        model = FadeFormerStagger(gptconf)
    elif model_type == 'fadeformer-half':
        model = FadeFormerHalf(gptconf)
    elif model_type == 'fadeformer-pool':
        model = FadeFormerPool(gptconf)
    elif model_type == 'fadeformer-trans':
        model = FadeFormerTrans(gptconf)
    elif model_type == 'fadeformer-cut':
        if pretrain:
            model = FadeFormerCut(gptconf, pretrain=True)
        else:
            model = FadeFormerCut(gptconf)
    elif model_type == 'fadeformer-even':
        model = FadeFormerEven(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.' # remove weird prefix (according to nanoGPT)
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    # iter_num = checkpoint['iter_num']
    # best_val_loss = checkpoint['best_val_loss']

# print parameter count of model
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

model.to(device)

# GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)
# TODO: add support for loading from a checkpoint

# compile model for better performance
if compile:
    print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# learning rate scheduler
def get_lr(it):
    if it < warmup_iters:
        return lr * it / warmup_iters
    elif decay_lr:
        return min_lr + (lr - min_lr) * (1 + math.cos(math.pi * (it - warmup_iters) / lr_decay_iters)) / 2
    else:
        return lr

# estimator for averaging loss over several batches and both splits
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# function for measuring perplexity of transformer language models
@torch.no_grad()
def measure_perplexity():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters//10)
        for k in range(eval_iters//10):
            X, Y = get_batch(split)
            with ctx:
                # use a sliding window to compute log-likelihoods
                window_size = ctx_size # maximum context size of the model
                log_probs = torch.zeros(Y.shape) # store log probabilities of each token
                for i in range(0, Y.shape[1], window_size):
                    # get the input and target segments
                    input_ids = X[:, i:i+window_size]
                    labels = Y[:, i:i+window_size]
                    # get the model outputs
                    logits, loss = model(input_ids, labels)
                    # get the log probabilities of the target tokens
                    log_probs[:, i:i+window_size] = logits.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                # compute the average negative log-likelihood
                loss = -log_probs.mean()
            losses[k] = loss.item()
        # compute the perplexity as the exponentiation of the loss
        out[split] = torch.exp(losses).mean()
    model.train()
    return out


# logging
if wandb_log:
    wandb.init(project=wandb_project, name=wandb_run_name)

# training loop
X, Y = get_batch('train') # first batch
t0 = time.time()
for it in (pbar := tqdm(range(iter_num, max_iters), desc="Training")):
    # set learning rate
    learning_rate = get_lr(it)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    # eval
    if it % eval_interval == 0:
        losses = estimate_loss()
        # perplexities = measure_perplexity()
        pbar.set_description(f"Training | Loss: {losses['train']:.4f}")
        if wandb_log:
            wandb.log({
                'train/loss': losses['train'], 
                'val/loss': losses['val'], 
                # 'train/perplexity': perplexities['train'],
                # 'val/perplexity': perplexities['val'],
                'iter': it,
                'lr': learning_rate
            })
        # save best
        if losses['val'] < best_val_loss + (best_val_loss/1000) or always_save:
            best_val_loss = losses['val']
            if it > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                pbar.set_description(f"Training | Loss: {losses['train']:.4f} | Saving to {out_dir}/{(model_name if init_from == 'scratch' else new_model_name)}.pt")
                torch.save(checkpoint, os.path.join(out_dir, (model_name if init_from == 'scratch' else new_model_name)+'.pt'))
    if it == 0 and eval_only:
        print("train/loss: ", losses['train'])
        print("val/loss: ", losses['val'])
        # print("train/perplexity: ", perplexities['train'])
        # print("val/perplexity: ", perplexities['val'])
        break

    # forward and backward pass with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            # scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        # each microstep adds scaled gradients to the optimizer's gradient buffers
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    # only after here are the gradients applied to the model
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if it % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        pbar.set_description(f"Training | Loss: {lossf:.5f}")
    
if not eval_only:
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    print(f"Saving latest iteration with loss: {losses['train']:.4f} | Saving to {out_dir}/{(model_name if init_from == 'scratch' else new_model_name)}-latest.pt")
    torch.save(checkpoint, os.path.join(out_dir, (model_name if init_from == 'scratch' else new_model_name)+'-latest.pt'))
