#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify the FinReflectKG_TTransE data folder is correctly formatted for
RE-Net's TTransE baseline. Does NOT need PyTorch — pure stdlib.

Run:
    python3 verify_finreflectkg_data.py --data ./data/FinReflectKG_TTransE
"""

import argparse
import os
import sys


def load_quads(path):
    quads = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_no}: expected 4 fields, got {len(parts)}")
            try:
                h, r, t, ts = (int(x) for x in parts)
            except ValueError as e:
                raise ValueError(f"{path}:{line_no}: non-integer field: {e}")
            quads.append((h, r, t, ts))
    return quads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    # 1. stat.txt
    stat_path = os.path.join(args.data, "stat.txt")
    if not os.path.exists(stat_path):
        sys.exit(f"missing {stat_path}")
    with open(stat_path) as f:
        parts = f.read().split()
    if len(parts) != 3:
        sys.exit(f"stat.txt should contain 3 numbers (E R T), got {parts}")
    n_e, n_r, n_t = map(int, parts)
    print(f"[stat] entities={n_e}  relations={n_r}  time_steps={n_t}")

    # 2. quadruple files
    splits = ["train2id.txt", "valid2id.txt", "test2id.txt"]
    all_ents = set()
    all_rels = set()
    all_times = set()
    total = 0
    for name in splits:
        path = os.path.join(args.data, name)
        if not os.path.exists(path):
            sys.exit(f"missing {path}")
        quads = load_quads(path)
        total += len(quads)
        bad = []
        for h, r, t, ts in quads:
            if not (0 <= h < n_e):    bad.append(("h", h))
            if not (0 <= t < n_e):    bad.append(("t", t))
            if not (0 <= r < n_r):    bad.append(("r", r))
            if not (0 <= ts < n_t):   bad.append(("ts", ts))
            all_ents.add(h); all_ents.add(t)
            all_rels.add(r); all_times.add(ts)
        if bad:
            sys.exit(f"{path}: out-of-range ids: {bad[:5]} (and more)")
        print(f"[ok] {name}: {len(quads)} quadruples")

    print(f"[summary] total quadruples = {total}")
    print(f"[summary] distinct entity ids seen = {len(all_ents)}")
    print(f"[summary] distinct relation ids seen = {len(all_rels)}")
    print(f"[summary] distinct time ids seen     = {len(all_times)}")

    # 3. vocab files (optional but recommended)
    for vocab in ["entity2id.txt", "relation2id.txt", "time2id.txt"]:
        p = os.path.join(args.data, vocab)
        if os.path.exists(p):
            n = sum(1 for _ in open(p))
            print(f"[vocab] {vocab}: {n} entries")

    print("\nALL CHECKS PASSED — data is ready for TTransE.py")


if __name__ == "__main__":
    main()
