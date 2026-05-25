#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pretty-print the test results that TTransE.py appends to
./result/{DATASET}.txt — one row per run.

Each row in that file has the format:
    {filename}  testSet  Hits@1  Hits@3  Hits@10  MeanRank  MRR

Usage:
    python3 show_results.py                       # default: result/FinReflectKG.txt
    python3 show_results.py -d ICEWS14
    python3 show_results.py -f /full/path/result.txt
    python3 show_results.py --best                # show only the best run by MRR
"""

import argparse
import os
import re
import sys


def parse_filename_params(fname):
    """Pull (key,value) pairs out of the encoded TTransE checkpoint name."""
    # e.g. dropout_0_l_0.001_es_10_L_1_em_100_bs_128_m_1.0_f_1_mo_0.9_s_0_op_1_lo_0_TTransE.ckpt
    fname = re.sub(r'_TTransE.*$', '', fname)
    parts = fname.split('_')
    pairs = {}
    i = 0
    while i < len(parts) - 1:
        k, v = parts[i], parts[i + 1]
        # heuristic: a numeric v means k is a key, otherwise it's a continuation
        if re.fullmatch(r'-?\d+(\.\d+)?', v):
            pairs[k] = v
            i += 2
        else:
            i += 1
    return pairs


def fmt_row(row, show_hp=False):
    """row = [filename, 'testSet', hit1, hit3, hit10, meanrank, mrr]"""
    if len(row) < 7:
        return None
    fname, _, h1, h3, h10, mr, mrr = row[:7]
    out = (
        f"  Hits@1   = {float(h1):.4f}  ({100*float(h1):5.2f}%)\n"
        f"  Hits@3   = {float(h3):.4f}  ({100*float(h3):5.2f}%)\n"
        f"  Hits@10  = {float(h10):.4f}  ({100*float(h10):5.2f}%)\n"
        f"  MeanRank = {float(mr):8.2f}\n"
        f"  MRR      = {float(mrr):.4f}"
    )
    if show_hp:
        hp = parse_filename_params(fname)
        # show the most informative ones
        keys = ['l', 'em', 'bs', 'm', 'L', 'f', 'op', 'lo', 's']
        labels = {'l': 'lr', 'em': 'dim', 'bs': 'bs', 'm': 'margin',
                  'L': 'L1', 'f': 'filter', 'op': 'opt', 'lo': 'loss', 's': 'seed'}
        hp_str = '  '.join(f'{labels.get(k,k)}={hp[k]}' for k in keys if k in hp)
        out = f"  [hyperparams] {hp_str}\n" + out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='FinReflectKG',
                    help='dataset name (looks for ./result/{name}.txt)')
    ap.add_argument('-f', '--file', default=None,
                    help='direct path to a result file (overrides --dataset)')
    ap.add_argument('--best', action='store_true',
                    help='show only the run with the highest MRR')
    args = ap.parse_args()

    path = args.file or os.path.join('./result', f'{args.dataset}.txt')
    if not os.path.exists(path):
        sys.exit(f'no result file at {path}\n'
                 f'(have you run `bash run_finreflectkg.sh` yet?)')

    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 7 and parts[1] == 'testSet':
                rows.append(parts)

    if not rows:
        sys.exit(f'no testSet rows found in {path}')

    print(f'\n[result file] {path}')
    print(f'[runs] {len(rows)}')

    if args.best:
        best = max(rows, key=lambda r: float(r[6]))
        print('\n=== BEST RUN (by MRR) ===')
        print(fmt_row(best, show_hp=True))
        return

    for i, row in enumerate(rows, 1):
        print(f'\n--- run #{i} ---')
        print(fmt_row(row, show_hp=True))


if __name__ == '__main__':
    main()
