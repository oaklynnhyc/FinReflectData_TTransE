#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess FinReflectKG `three_relations_sample2000.tsv` into the format
expected by RE-Net's TTransE baseline.

Source TSV columns (12):
    triplet_id, entity, entity_type, relationship, target, target_type,
    start_date, end_date, ticker, year, source_file, chunk_id

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
from collections import OrderedDict


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
    # time:     column `year` (we discretise to one slot per year)
    quads_str = []
    for r in rows:
        h = r["entity"].strip().lower()
        rel = r["relationship"].strip().lower()
        t = r["target"].strip().lower()
        ts = r["year"].strip()
        if not (h and rel and t and ts):
            continue
        quads_str.append((h, rel, t, ts))

    print(f"[clean] kept {len(quads_str)} non-empty quadruples")

    # --- build unified entity vocabulary (subject ∪ object) ---
    ent_iter = []
    for h, rel, t, ts in quads_str:
        ent_iter.append(h)
        ent_iter.append(t)
    entity2id = build_vocab(ent_iter)

    rel_iter = [rel for _, rel, _, _ in quads_str]
    relation2id = build_vocab(rel_iter)

    # Time: sort numerically so that consecutive years get consecutive ids
    years_sorted = sorted({ts for _, _, _, ts in quads_str}, key=lambda x: int(x))
    time2id = OrderedDict((y, i) for i, y in enumerate(years_sorted))

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
