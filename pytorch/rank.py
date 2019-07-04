# coding: utf-8
import argparse
import time
import math
import os, sys

import torch
import numpy as np

from data_utils_ranking import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import get_logger

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8', 'dummy'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['all', 'valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=5,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str, required=True,
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda" if args.cuda else "cpu")

# Get logger
logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
                     log_=not args.no_log)

# Load dataset
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)

va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)


# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
model.backward_compatible()
model = model.to(device)

logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
       args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))

model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True

###############################################################################
# Evaluation code
###############################################################################
def get_mask(lens):
  max_len = max(lens)
  lens = torch.LongTensor(lens)
  mask = torch.arange(max_len).expand(lens.size(0), max_len) < lens.unsqueeze(1)
  return mask.float()

def format_rank(loss_ranking, idx):
  top_k = torch.topk(loss_ranking, 5, largest=False)
  log = 'topk for batch {}: {}'.format(idx, top_k)
  return log

def get_rank(loss_ranking):
  sort, idx = torch.sort(loss_ranking, descending=False)
  real_target_position = (idx == 0).nonzero()
  return real_target_position

def get_rankings(ranking_positions):
  rank_len = len(ranking_positions)
  ranking_positions = np.array(ranking_positions)
  r_0 = sum(ranking_positions < 1) / rank_len
  r_5 = sum(ranking_positions < 5) / rank_len
  r_10 = sum(ranking_positions < 10) / rank_len
  return r_0, r_5, r_10

def mean_with_lengths(data, lengths):
  return torch.FloatTensor([sum(d) / l for d, l in zip(data, lengths)])

def evaluate_2file_setup(eval_iter):
  # Turn on evaluation mode which disables dropout.
  model.eval()
  total_len, total_loss = 0, 0.
  start_time = time.time()
  memories = model.init_mems()
  ranking_positions = []
  with torch.no_grad():
    #mems = tuple()
    for idx, (body_data, cand_data, body_target, cand_target, body_len, cand_len) in enumerate(eval_iter):
      overall_len = [l + body_len for l in cand_len]
      ret = model(body_data, body_target, *memories)
      loss, mems = ret[0], ret[1:]
      loss_body = loss.squeeze()
      quasi_mean_body = mean_with_lengths(loss_body.unsqueeze(0).expand(100, -1), overall_len)
      # expand mems 100 times on the batch dim
      mems = [m.expand(-1, 100, -1) for m in mems]
      ret = model(cand_data, cand_target, *mems)
      loss, _ = ret[0], ret[1:]
      # get batch first
      loss = loss.t()
      # mask padding
      mask = get_mask(cand_len)
      loss = loss * mask
      # get loss for every sentence

      loss_cand = mean_with_lengths(loss, overall_len)
      loss_ranking = quasi_mean_body + loss_cand
      # get ranking
      print(format_rank(loss_ranking, idx))
      real_target_position = get_rank(loss_ranking)
      ranking_positions.append(real_target_position.item())

      # if iteration % 20 == 0 print recalls till now
      if idx % 1 == 0:
        r_0, r_5, r_10 = get_rankings(ranking_positions)
        print('R@1 = {}%, R@5 = {}%, R@10 = {}%'.format(r_0 * 100, r_5 * 100, r_10 * 100))

      # get overall batch loss
      loss = loss_ranking.mean()
      print('losses: {:.2f}'.format(loss))
      total_loss += sum(cand_len) * loss.item()
      total_len += sum(cand_len)
    print(ranking_positions)
    total_time = time.time() - start_time
  logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
    total_time, 1000 * total_time / (idx + 1)))
  return total_loss / total_len

def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_len, total_loss = 0, 0.
    start_time = time.time()
    memories = model.init_mems()
    with torch.no_grad():
        mems = tuple()
        for idx, (data, target, seq_len) in enumerate(eval_iter):
            ret = model(data, target, *memories)
            loss, mems = ret[0], ret[1:]
            # get batch first
            loss = loss.t()
            # mask padding
            mask = get_mask(seq_len)
            loss = loss * mask
            # get loss for every sentence
            loss_ranking = mean_with_lengths(loss, seq_len)
            # get ranking
            print(format_rank(loss_ranking, idx))
            # get overall batch loss
            loss = loss_ranking.mean()
            print('losses: {:.2f}'.format(loss))
            total_loss += sum(seq_len) * loss.item()
            total_len += sum(seq_len)
        total_time = time.time() - start_time
    logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
            total_time, 1000 * total_time / (idx+1)))
    return total_loss / total_len

# Run on test data.
if args.split == 'all':
    test_loss = evaluate(te_iter)
    valid_loss = evaluate(va_iter)
elif args.split == 'valid':
    valid_loss = evaluate(va_iter)
    test_loss = None
elif args.split == 'test':
    test_loss = evaluate(te_iter)
    valid_loss = None

def format_log(loss, split):
    if args.dataset in ['enwik8', 'text8']:
        log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
            split, loss, loss / math.log(2))
    else:
        log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
            split, loss, math.exp(loss))
    return log_str

log_str = ''
if valid_loss is not None:
    log_str += format_log(valid_loss, 'valid')
if test_loss is not None:
    log_str += format_log(test_loss, 'test')

logging('=' * 100)
logging(log_str)
logging('=' * 100)
