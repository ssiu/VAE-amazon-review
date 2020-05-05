import json
import gzip
import numpy as np
import math
from collections import Counter


def get_review_scores(file_path):
    """
    reads a JSON file, get distribution of review scores

    Args
        file_path (str):

    Returns

    """
    review_scores = [0, 0, 0, 0, 0]
    with gzip.open(file_path) as f:
        for l in f:
            x = json.loads(l.strip())
            if 'reviewText' not in x or 'overall' not in x:
                continue
            review_scores[int(x['overall'])-1] += 1
    return review_scores


def get_sents_len(dataset):
    # get sentence length distribution
    len_freq = Counter([len(sent) for sent in dataset])
    sorted_len_freq = sorted(len_freq.keys())
    #json.dump([(l, len_freq[l]) for l in sorted_len_freq], open('./len_distribution.json', 'w'), indent=2)
    return sorted_len_freq


def load_dataset(file_path):
    """
    param file_path:
    return
    """
    dataset = []
    with open(file_path, 'r') as f:
        for l in f:
            x = json.loads(l)
            dataset.append(x)
    return dataset


def save_dataset(dataset, file_path):
    with open(file_path, 'w') as f:
        for l in dataset:
            json.dump(l, f)
            f.write('\n')


def save_corpus(dataset, file_path):
    json.dump(dataset, open(file_path, 'w'), indent=2)


def remove_unk(dataset, vocab):
    clean_dataset = []
    for sent in dataset:
        word_ids = vocab.vocab_entry.words2indices(sent)
        if vocab.vocab_entry['<unk>'] not in word_ids:
            clean_dataset.append(sent)
    return clean_dataset


def input_transpose(sents, pad_token):
    """
    tranposes a dataset from List[List[str]] to array of size (seq_len, num_sents)

    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)
    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])
    return sents_t


def batch_loader(dataset, batch_size, shuffle=False):
    """
    splits a dataset into batches for training
    """
    batch_num = math.ceil(len(dataset)/batch_size)
    index_array = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(index_array)
    for i in range(batch_num):
        example_indices = index_array[i*batch_size: (i+1)*batch_size]
        examples = [dataset[example_index] for example_index in example_indices]
        yield i, examples


def annealing(x, a=0.9, b=5):
    '''
    Function for KL annealing, a,b are fine-tuned for 10 epochs
    '''
    z = 1/(1+math.exp(-a*x+b))
    return z


def word_dropout(sents, keep_rate, unk):
    """
    word dropout
    """
    new_sents = []
    for sent in sents:
        size = len(sent)
        keep = np.random.binomial(n=1, p=keep_rate, size=size)
        new_sent = [sent[i] if keep[i] == 1 else unk for i in range(size)]
        new_sents.append(new_sent)
    return new_sents


def search_substring(dict, substring, file_path):
    """
    This is for cleaning datasets. It searches the vocab dictionary and grabs any vocabs that contains a particular substring.
    """
    list = []
    for key in dict.keys():
        if substring in key:
            list.append(key)
    json.dump(list, open(file_path, 'w'), indent=2)
