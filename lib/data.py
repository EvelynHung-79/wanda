# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
import json
import gzip
import pandas as pd

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
    # Load train and test datasets from local parquet files using pandas
    traindata = pd.read_parquet('./wikitext2_dataset/train-00000-of-00001.parquet')
    testdata = pd.read_parquet('./wikitext2_dataset/test-00000-of-00001.parquet')

    # Convert to list of texts
    train_texts = traindata['text'].tolist()
    test_texts = testdata['text'].tolist()

    # Encode datasets
    trainenc = tokenizer(" ".join(train_texts), return_tensors='pt')
    testenc = tokenizer("\n\n".join(test_texts), return_tensors='pt')

    # Generate samples from training set
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

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets directly from gzipped JSON files
    def load_gzipped_json(file_path):
        data = []
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    traindata = load_gzipped_json('./en/c4-train.00000-of-01024.json.gz')
    valdata = load_gzipped_json('./en/c4-validation.00000-of-00008.json.gz')

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
    # valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = tokenizer(' '.join([x['text'] for x in valdata[:1100]]), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)