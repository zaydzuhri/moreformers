"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import argparse
import time
import numpy as np
from torch.nn.functional import cross_entropy
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
from models.fadeformer_residual import FadeFormerResidual
from models.lessformer_qkk import LessFormerQKK
from models.lessformer_mqa import LessFormerMQA
from models.lessformer_mqx import LessFormerMQX
from models.lessformer_mqxk import LessFormerMQXK
from models.llama import LLaMA
from models.llama_mqa import LLaMAMQA
from models.lessllama import LessLLaMA
from models.nonellama import NoneLLaMA
from models.weightllama import WeightLLaMA
from models.buffllama import BuffLLaMA
from models.sumllama import SumLLaMA
from models.doublellama import DoubleLLaMA
from models.localllama import LocalLLaMA
from models.fadellama import FadeLLaMA
from models.fadellama_sum import FadeLLaMASum
from models.fadellama_invert import FadeLLaMAInvert
from models.fadellama_post import FadeLLaMAPost
from models.fadellama_v import FadeLLaMAV
from models.fadellama_ff import FadeLLaMAFF
from models.fadellama_attff import FadeLLaMAAttFF
from models.fadellama_fflin import FadeLLaMAFFLin

# -----------------------------------------------------------------------------
out_dir = 'out' # model output directory
model_type = 'gpt'
model_name = 'mini-gpt'
seed = 420
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_type', type=str, default=model_type)
argparser.add_argument('--model_name', type=str, default=model_name)
argparser.add_argument('--dataset', type=str, default='poetry')
argparser.add_argument('--eval_iters', type=int, default=100)
argparser.add_argument('--last_k', type=int, default=None)
argparser.add_argument('--ctx_size', type=int, default=None)
args = argparser.parse_args()
model_name = args.model_name
model_type = args.model_type
eval_iters = args.eval_iters
last_k = args.last_k
dataset = args.dataset
# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device.type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

# Poor man's data loader
data_dir = os.path.join('data', dataset)
# Load data from prepared .bin files (see prepare.py in data folder)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
# get_batch samples batches of blocks of tokens randomly from the data and returns x input and y target sequences
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # get random indices for each batch
    ix = torch.randint(len(data) - config['ctx_size'], (config['batch_size'],))
    # x is the input sequence, y is the target sequence which is x shifted by 1
    x = torch.stack([torch.from_numpy((data[i:i+config['ctx_size']]).astype(np.int64)) for i in ix])
    if 'gpt' in model_type or 'llama' in model_type or 'lessformer' in model_type or 'moreformer' in model_type or model_type == 'fadeformer-residual':
        y = torch.stack([torch.from_numpy((data[i+1:i+1+config['ctx_size']]).astype(np.int64)) for i in ix])
    elif model_type == 'fadeformer-linear':
        y = torch.stack([torch.from_numpy((data[i+1:i+1+target_size]).astype(np.int64)) for i in ix])
    elif model_type == 'fadeformer-rank':
        target_size = int(config['ctx_size'] // (2**n_layer))
        y = torch.stack([torch.from_numpy((data[i+1+(config['ctx_size']-target_size):i+1+config['ctx_size']]).astype(np.int64)) for i in ix])
    else:
        target_size = int(config['ctx_size'] // (2**(n_layer-2)))
        y = torch.stack([torch.from_numpy((data[i+1+(config['ctx_size']-target_size):i+1+config['ctx_size']]).astype(np.int64)) for i in ix])
    if device.type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, model_name+'.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
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
    model = FadeFormerCut(gptconf)
elif model_type == 'fadeformer-even':
    model = FadeFormerEven(gptconf)
elif model_type == 'fadeformer-residual':
    model = FadeFormerResidual(gptconf)
elif model_type == 'lessformer-qkk':
    model = LessFormerQKK(gptconf)
elif model_type == 'lessformer-mqa':
    model = LessFormerMQA(gptconf)
elif model_type == 'lessformer-mqx':
    model = LessFormerMQX(gptconf)
elif model_type == 'lessformer-mqxk':
    model = LessFormerMQXK(gptconf)
elif model_type == 'llama':
    model = LLaMA(gptconf)
elif model_type == 'llama-mqa':
    model = LLaMAMQA(gptconf)
elif model_type == 'lessllama':
    model = LessLLaMA(gptconf)
elif model_type == 'nonellama':
    model = NoneLLaMA(gptconf)
elif model_type == 'weightllama':
    model = WeightLLaMA(gptconf)
elif model_type == 'buffllama':
    model = BuffLLaMA(gptconf)
elif model_type == 'sumllama':
    model = SumLLaMA(gptconf)
elif model_type == 'doublellama':
    model = DoubleLLaMA(gptconf)
elif model_type == 'localllama':
    model = LocalLLaMA(gptconf)
elif model_type == 'fadellama':
    model = FadeLLaMA(gptconf)
elif model_type == 'fadellama-sum':
    model = FadeLLaMASum(gptconf)
elif model_type == 'fadellama-invert':
    model = FadeLLaMAInvert(gptconf)
elif model_type == 'fadellama-post':
    model = FadeLLaMAPost(gptconf)
elif model_type == 'fadellama-v':
    model = FadeLLaMAV(gptconf)
elif model_type == 'fadellama-ff':
    model = FadeLLaMAFF(gptconf)
elif model_type == 'fadellama-attff':
    model = FadeLLaMAAttFF(gptconf)
elif model_type == 'fadellama-fflin':
    model = FadeLLaMAFFLin(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.' # remove weird prefix (according to nanoGPT)
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

# load hyperparams to a dict
config = checkpoint['model_args']
if args.ctx_size is not None:
    config['ctx_size'] = args.ctx_size
print(f"model: {model_name}, dataset: {dataset}")
print(f"model_type: {model_type}")
print(f"last_k: {last_k}, eval_iters: {eval_iters}, ctx_size: {config['ctx_size']}")
print(f"num params: {model.get_num_params():,}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# Create CUDA events
start_event_generate = torch.cuda.Event(enable_timing=True)
end_event_generate = torch.cuda.Event(enable_timing=True)
start_event_forward = torch.cuda.Event(enable_timing=True)
end_event_forward = torch.cuda.Event(enable_timing=True)
start_event_backward = torch.cuda.Event(enable_timing=True)
end_event_backward = torch.cuda.Event(enable_timing=True)
generate_times = []
forward_times = []
backward_times = []

# estimator for averaging loss and perplexity over several batches and both splits
def estimate_loss_perplexity():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                # time generation with pytorch cuda events
                torch.cuda.synchronize()
                start_event_generate.record()

                pred = model.generate(X, max_new_tokens=1, temperature=1.0, top_k=None)

                end_event_generate.record()
                torch.cuda.synchronize()
                # compute the time in milliseconds
                generate_times.append(start_event_generate.elapsed_time(end_event_generate))

                # time the forward pass with pytorch cuda events
                torch.cuda.synchronize()
                start_event_forward.record()

                logits, loss_pt = model(X, Y)

                end_event_forward.record()
                torch.cuda.synchronize()
                # compute the time in milliseconds
                forward_times.append(start_event_forward.elapsed_time(end_event_forward))
                
                # time the backward pass with pytorch cuda events
                model.zero_grad()
                torch.cuda.synchronize()
                start_event_backward.record()

                loss_pt.backward()

                end_event_backward.record()
                torch.cuda.synchronize()
                # compute the time in milliseconds
                backward_times.append(start_event_backward.elapsed_time(end_event_backward))

            # calculate loss for only the last_k tokens
            if last_k is not None:
                logits = logits[:, -last_k:, :]
                Y = Y[:, -last_k:]
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)), Y.reshape(-1), ignore_index=-1)
            losses[k] = loss.item()
        # compute the average loss and perplexity
        out[split] = {'loss': losses.mean(), 'perplexity': torch.exp(losses).mean()}
    return out

print('Evaluating...')
# evaluate the model on train and val splits and get the loss and perplexity
loss_perplexity = estimate_loss_perplexity()
print(f"train loss: {loss_perplexity['train']['loss']:.02f}, perplexity: {loss_perplexity['train']['perplexity']:.02f}")
print(f"val loss: {loss_perplexity['val']['loss']:.02f}, perplexity: {loss_perplexity['val']['perplexity']:.02f}")
print(f"Average generation time: {sum(generate_times)/len(generate_times):.02f}ms/token")
print(f"Average forward pass time: {sum(forward_times)/len(forward_times):.02f}ms")
print(f"Average backward pass time: {sum(backward_times)/len(backward_times):.02f}ms")
