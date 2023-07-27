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

# -----------------------------------------------------------------------------
out_dir = 'out' # model output directory
model_type = 'gpt'
model_name = 'mini-gpt'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5 # number of samples to draw
max_new_tokens = 1000 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 69
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

argparser = argparse.ArgumentParser()
argparser.add_argument('--start', type=str, default=start)
argparser.add_argument('--model_type', type=str, default=model_type)
argparser.add_argument('--model_name', type=str, default=model_name)
args = argparser.parse_args()
start = args.start
model_name = args.model_name
model_type = args.model_type
# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device.type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

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

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.' # remove weird prefix (according to nanoGPT)
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

# print('Model architecture:')
# print(model)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

times = []
# run generation with timer
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            torch.cuda.synchronize()
            t0 = time.time()
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('--------------------------------------')
            torch.cuda.synchronize()
            print(f'Generation took {time.time()-t0:.02f}s for {max_new_tokens} tokens.')
            print('======================================')
            times.append(time.time()-t0)

print(f'Average generation time: {sum(times)/len(times):.02f}s for {max_new_tokens} tokens.')
