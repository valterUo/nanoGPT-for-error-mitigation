import os
from datasets import load_dataset
import numpy as np
import tiktoken

dataset = load_dataset("tatoeba", lang1="en", lang2="fi", trust_remote_code=True)

# Divide the dataset into train_src, train_tgt and val_src and val_tgt bins
train_src = [doc["en"] for doc in dataset['train']['translation']]
train_tgt = [doc["fi"] for doc in dataset['train']['translation']]
# From training data, take random 20% for validation
n = len(train_src)
val_indices = np.random.choice(n, int(n*0.2), replace=False)
val_src = [train_src[i] for i in val_indices]
val_tgt = [train_tgt[i] for i in val_indices]
train_src = [train_src[i] for i in range(n) if i not in val_indices]
train_tgt = [train_tgt[i] for i in range(n) if i not in val_indices]

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
def process(list_of_examples):
    all_ids = []
    for example in list_of_examples:
        ids = enc.encode_ordinary(example) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        all_ids.extend(ids)
    return all_ids

# Save the data to disk
# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_src_ids = process(train_src)
train_tgt_ids = process(train_tgt)
val_src_ids = process(val_src)
val_tgt_ids = process(val_tgt)

print(f"train_src has {len(train_src_ids):,} tokens")
print(f"train_tgt has {len(train_tgt_ids):,} tokens")
print(f"val_src has {len(val_src_ids):,} tokens")
print(f"val_tgt has {len(val_tgt_ids):,} tokens")

# export to bin files
train_src_ids = np.array(train_src_ids, dtype=np.uint16)
train_tgt_ids = np.array(train_tgt_ids, dtype=np.uint16)
val_src_ids = np.array(val_src_ids, dtype=np.uint16)
val_tgt_ids = np.array(val_tgt_ids, dtype=np.uint16)

train_src_ids.tofile(os.path.join(os.path.dirname(__file__), 'train_src.bin'))
train_tgt_ids.tofile(os.path.join(os.path.dirname(__file__), 'train_tgt.bin'))  
val_src_ids.tofile(os.path.join(os.path.dirname(__file__), 'val_src.bin'))
val_tgt_ids.tofile(os.path.join(os.path.dirname(__file__), 'val_tgt.bin'))