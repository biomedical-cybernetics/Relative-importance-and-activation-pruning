# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
import os
# os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset



# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    # trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []

    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        
    return trainloader, testenc

def get_pile(nsamples, seed, seqlen, tokenizer):
    dataset = load_dataset("../mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=seed)
    samples = []
    n_run = 0
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(dataset) - 1)
            trainenc = tokenizer(dataset[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        samples.append((inp, tar))

    return samples, samples
    # now concatenate all samples and split according to block size
    # cat_samples = torch.cat(samples, dim=1)
    # print(cat_samples[:10], cat_samples.shape)
    # exit()
    # n_split = cat_samples.shape[1] // block_size
    # print(f" * Split into {n_split} blocks")
    # return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('../../c4', data_files={'train': 'c4-train.00000-of-01024.json'}, split='train')
    valdata = load_dataset('../../c4', data_files={'validation': 'c4-validation.00000-of-00008.json'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, _

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    
    while True:
        try:
            if 'wikitext2' in name:
                return get_wikitext2(nsamples, seed, seqlen, tokenizer)
            if "c4" in name:
                return get_c4(nsamples, seed, seqlen, tokenizer)
            if "ptb" in name:
                return get_ptb(nsamples, seed, seqlen, tokenizer)
            if "pile" in name:
                return get_pile(nsamples, seed, seqlen, tokenizer)
            print("Connection successful!")
            break
        except ConnectionError or ProxyError:
            print("Connection error, try again!")
    # if 'wikitext2' in name:
    #     return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    # if "c4" in name:
    #     return get_c4(nsamples, seed, seqlen, tokenizer)
    # if "ptb" in name:
    #     return get_ptb(nsamples, seed, seqlen, tokenizer)
        