#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess FinReflectKG `three_relations_sample2000.tsv` into the format
expected by RE-Net's TTransE baseline.

Source TSV columns (12):
    triplet_id, entity, entity_type, relationship, target, target_type,
    start_date, end_date, ticker, year, source_file, chunk_id

Time is taken from `start_date` at month granularity ('YYYY-MM'); rows whose
start_date is a placeholder fall back to the `year` column ('YYYY-00').

Output (under ./data/FinReflectKG_TTransE/):
    train2id.txt   - head_id <tab> rel_id <tab> tail_id <tab> time_id
    valid2id.txt
    test2id.txt
    stat.txt       - "<num_entities> <num_relations> <num_times>"
    entity2id.txt  - entity_string <tab> id   (for inspection)
    relation2id.txt
    time2id.txt

Usage:
    cd /path/to/RE-Net/baselines
    python3 preprocess_finreflectkg.py \
        --src /Users/yc/Documents/AI_SupplyChain_RA/FinReflect/finreflectkg_subset/three_relations_sample2000.tsv \
        --out ./data/FinReflectKG_TTransE \
        --seed 42
"""

import argparse
import csv
import os
import random
import re
from collections import OrderedDict


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_month_year(s):
    """Parse a 'Month YYYY' string (e.g. 'January 2020') into 'YYYY-MM'.

    Returns None when no month name + 4-digit year can be found (e.g. the
    'default_start_timestamp' placeholder rows). The caller falls back to the
    `year` column in that case.
    """
    if not s:
        return None
    low = s.lower()
    month = None
    for name, idx in _MONTHS.items():
        if name in low:
            month = idx
            break
    if month is None:
        return None
    y = re.search(r"(\d{4})", s)
    if not y:
        return None
    return f"{int(y.group(1))}-{month:02d}"


def time_sort_key(token):
    """Order time tokens chronologically: 'YYYY-MM' and fallback 'YYYY-00'.

    'YYYY-00' (year-only fallback) sorts just before that year's January.
    """
    year, month = token.split("-")
    return (int(year), int(month))


def load_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def build_vocab(values):
    """Stable, deterministic id assignment: first-seen order."""
    vocab = OrderedDict()
    for v in values:
        if v not in vocab:
            vocab[v] = len(vocab)
    return vocab


def write_id_file(path, vocab):
    with open(path, "w", encoding="utf-8") as f:
        for token, idx in vocab.items():
            f.write(f"{token}\t{idx}\n")


def write_quadruples(path, quads):
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t, ts in quads:
            # RE-Net TTransE expects 4 whitespace-separated ints per line:
            #     head  rel  tail  time
            # (note: order is head, rel, tail, time -- not the column order)
            f.write(f"{h}\t{r}\t{t}\t{ts}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to three_relations_sample2000.tsv")
    ap.add_argument("--out", default="./data/FinReflectKG_TTransE", help="output folder")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    args = ap.parse_args()

    random.seed(args.seed)

    rows = load_tsv(args.src)
    print(f"[load] read {len(rows)} rows from {args.src}")

    # --- field selection ---
    # subject: lower-cased ticker / org code (column `entity`)
    # relation: column `relationship`
    # object:   column `target`
    # time:     `start_date` discretised to month granularity -> 'YYYY-MM'.
    #           When start_date is a placeholder (e.g. 'default_start_timestamp')
    #           we fall back to the `year` column as a year-only 'YYYY-00' slot.
    quads_str = []
    n_fallback = 0
    for r in rows:
        h = r["entity"].strip().lower()
        rel = r["relationship"].strip().lower()
        t = r["target"].strip().lower()
        ts = parse_month_year(r["start_date"])
        if ts is None:                       # fallback to the `year` column
            y = r["year"].strip()
            ts = f"{y}-00" if y else None
            if ts is not None:
                n_fallback += 1
        if not (h and rel and t and ts):
            continue
        quads_str.append((h, rel, t, ts))

    print(f"[clean] kept {len(quads_str)} non-empty quadruples "
          f"({n_fallback} used year-column fallback)")

    # --- build unified entity vocabulary (subject ∪ object) ---
    ent_iter = []
    for h, rel, t, ts in quads_str:
        ent_iter.append(h)
        ent_iter.append(t)
    entity2id = build_vocab(ent_iter)

    rel_iter = [rel for _, rel, _, _ in quads_str]
    relation2id = build_vocab(rel_iter)

    # Time: sort chronologically (year, month) so consecutive months get
    # consecutive ids. Tokens are 'YYYY-MM' (or 'YYYY-00' year-only fallback).
    times_sorted = sorted({ts for _, _, _, ts in quads_str}, key=time_sort_key)
    time2id = OrderedDict((y, i) for i, y in enumerate(times_sorted))

    print(f"[vocab] |E|={len(entity2id)}  |R|={len(relation2id)}  |T|={len(time2id)}")

    # --- map to ids ---
    quads_id = [
        (entity2id[h], relation2id[rel], entity2id[t], time2id[ts])
        for (h, rel, t, ts) in quads_str
    ]

    # --- de-duplicate (some (s,p,o,t) may repeat across triplets) ---
    quads_id = list(OrderedDict.fromkeys(quads_id))
    print(f"[dedup] {len(quads_id)} unique quadruples")

    # --- random 80/10/10 split ---
    random.shuffle(quads_id)
    n = len(quads_id)
    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)
    train = quads_id[:n_train]
    valid = quads_id[n_train:n_train + n_valid]
    test = quads_id[n_train + n_valid:]
    print(f"[split] train={len(train)}  valid={len(valid)}  test={len(test)}")

    # --- write outputs ---
    os.makedirs(args.out, exist_ok=True)
    write_quadruples(os.path.join(args.out, "train2id.txt"), train)
    write_quadruples(os.path.join(args.out, "valid2id.txt"), valid)
    write_quadruples(os.path.join(args.out, "test2id.txt"), test)

    with open(os.path.join(args.out, "stat.txt"), "w") as f:
        f.write(f"{len(entity2id)}\t{len(relation2id)}\t{len(time2id)}\n")

    write_id_file(os.path.join(args.out, "entity2id.txt"), entity2id)
    write_id_file(os.path.join(args.out, "relation2id.txt"), relation2id)
    write_id_file(os.path.join(args.out, "time2id.txt"), time2id)

    print(f"[done] wrote files under {args.out}")


if __name__ == "__main__":
    main()
