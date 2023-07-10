import os
import tiktoken
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("wikipedia", "20220301.simple", split="train")

data = ''

for i in tqdm(range(len(dataset))):
    data += dataset[i]['title'] + '\n'
    data += dataset[i]['text'] + '\n'

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile('train.bin')
val_ids.tofile('val.bin')
