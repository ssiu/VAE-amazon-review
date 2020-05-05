#!/usr/bin/env python
"""
Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE
Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from typing import List
from collections import Counter
from itertools import chain
from utils import read_corpus, input_transpose
import json
import torch


class VocabEntry(object):
    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<unk>'] = 3

        self.unk_id = self.word2id['<unk>']

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        word_ids = self.words2indices(sents)
        sents_t = input_transpose(word_ids, self['<pad>'])
        sents_var = torch.tensor(data=sents_t, device=device)
        return sents_var

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        # print distribution graph for vocabulary
        word_distribution = sorted([word_freq[w] for w in valid_words], reverse=True)
        json.dump(word_distribution, open('./word_distribution.json', 'w'), indent=2)
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry


class Vocab(object):
    def __init__(self, vocab_entry: VocabEntry):
        self.vocab_entry = vocab_entry

    @staticmethod
    def build(sents, vocab_size, freq_cutoff) -> 'Vocab':
        print('initialize source vocabulary ..')
        dct = VocabEntry.from_corpus(sents, vocab_size, freq_cutoff)
        return Vocab(dct)

    def save(self, file_path):
        json.dump(dict(self.vocab_entry.word2id), open(file_path, 'w'), indent=2)

    @staticmethod
    def load(file_path):
        word2id = json.load(open(file_path, 'r'))
        return Vocab(VocabEntry(word2id))

    def __repr__(self):
        return 'Vocab(%d words)' % (len(self.vocab_entry))





#if __name__ == '__main__':
#    args = docopt(__doc__)
#    print('read in sentences: %s' % args['--sentences'])
#    sents = read_corpus(args['--data'])
#    vocab = Vocab.build(sents, int(args['--size']), int(args['--freq-cutoff']))
#    print('generated vocabulary, source %d words' % (len(vocab.vocab_entry)))

#    vocab.save(args['VOCAB_FILE'])
#    print('vocabulary saved to %s' % args['VOCAB_FILE'])


