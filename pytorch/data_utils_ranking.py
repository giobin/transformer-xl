import os, sys
import glob

from collections import Counter, OrderedDict
import numpy as np
import torch

from utils.vocabulary import Vocab

class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i+1:i+1+seq_len]

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()

class LMSimpleOrderedIteratorTwoFiles(object):
    def __init__(self, data_body, data_cand, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.data_body = data_body
        self.data_cand = data_cand
        self.device = device

    def get_batch(self, i, bptt=None):
        # get candidate len without <eos>
        # get lengths
        body_len = self.data_body[i][:-1].size(0)
        cand_len = [self.data_cand[i][:-1].size(0) for i in range(i * self.bsz, i * self.bsz + self.bsz)]

        # get data. body -> [len, 1] cand -> [len, bsz]
        body_data = self.data_body[i][:-1].unsqueeze(1).contiguous().to(self.device)
        body_target = self.data_body[i][1:].unsqueeze(1).contiguous().to(self.device)

        cand_batch = torch.nn.utils.rnn.pad_sequence(self.data_cand[i * self.bsz: i * self.bsz + self.bsz], batch_first=True)
        cand_data = cand_batch[:, :-1].transpose_(0, 1).contiguous().to(self.device)
        cand_target = cand_batch[:, 1:].transpose_(0, 1).contiguous().to(self.device)

        return body_data, cand_data, body_target, cand_target, body_len, cand_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, len(self.data_body)):
            yield self.get_batch(i)

    def __iter__(self):
        return self.get_fixlen_iter()

class LMSimpleOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.data = data
        self.device = device

    def get_batch(self, i, bptt=None):
        # get candidate len without <eos>
        seq_len = [self.data[i][:-1].size(0) for i in range(i, i + self.bsz)]
        batch = torch.nn.utils.rnn.pad_sequence(self.data[i : i + self.bsz], batch_first=True)

        data = batch[:, :-1].transpose_(0,1).contiguous().to(self.device)
        target = batch[:, 1:].transpose_(0,1).contiguous().to(self.device)

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, len(self.data), self.bsz):
            yield self.get_batch(i)

    def __iter__(self):
        return self.get_fixlen_iter()


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)

        self.vocab.count_file(os.path.join(path, 'train.txt'))
        self.vocab.count_file(os.path.join(path, 'valid.txt'))
        self.vocab.count_file(os.path.join(path, 'test.txt'))

        self.vocab.count_file(os.path.join(path, 'test_body.txt'))
        self.vocab.count_file(os.path.join(path, 'test_cand.txt'))

        self.vocab.build_vocab()

        # valid and test are generated swapping token with idx and keeping the line structure
        # no concatenation of lines
        self.train = self.vocab.encode_file(
            os.path.join(path, 'train.txt'), ordered=True)
        self.valid = self.vocab.encode_file(
            os.path.join(path, 'valid.txt'), ordered=False)
        self.test = self.vocab.encode_file(
            os.path.join(path, 'test.txt'), ordered=False)

        # test body doesn't need <eos>
        self.test_body = self.vocab.encode_file(
            os.path.join(path, 'test_body.txt'), ordered=False, add_eos=False)
        self.test_cand = self.vocab.encode_file(
            os.path.join(path, 'test_cand.txt'), ordered=False)

    def get_iterator(self, split, *args, **kwargs):
        if split in ['valid', 'test', 'train']:
            data = self.valid if split == 'valid' else self.test
            if split == 'train':
                data = self.train
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif split == 'valid':
                data = self.valid
                #data_iter = LMSimpleOrderedIterator(data, *args, **kwargs)
                data_iter = LMSimpleOrderedIteratorTwoFiles(self.test_body, self.test_cand, *args, **kwargs)
            elif split == 'test':
                data = self.test
                data_iter = LMSimpleOrderedIteratorTwoFiles(self.test_body, self.test_cand, *args, **kwargs)
                #data_iter = LMSimpleOrderedIterator(data, *args, **kwargs)

        return data_iter


def get_lm_corpus(datadir, dataset):
    fn = os.path.join(datadir, 'cache_ranking.pt')
    if os.path.exists(fn):
        print('Loading cached ranking dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset for ranking {}...'.format(dataset))
        assert dataset == 'dummy'
        kwargs = {}
        kwargs['special'] = ['<eos>']
        kwargs['lower_case'] = False

        corpus = Corpus(datadir, dataset, **kwargs)
        torch.save(corpus, fn)

    return corpus

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/text8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='text8',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8', 'dummy'],
                        help='dataset name')
    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir, args.dataset)
    print('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
