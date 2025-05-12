#!/usr/bin/env python
"""
Prepare ACTER IOB files for GLiNER fine‑tuning (flat labels).
Usage:
  python preprocess_acter_gliner.py \
         --input_dirs data/acter/en/*/iob_annotations/without_named_entities \
         --train_output train_full.tsv \
         --test_output  test_full.tsv
"""
import argparse
import random
from pathlib import Path
from typing import List, Tuple


class Sentence:
    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.labels: List[str] = []

    def add(self, token: str, label: str) -> None:
        self.tokens.append(token)
        self.labels.append(label)

    def to_tsv(self) -> str:
        """Return one sentence in TAB‑separated format with flat labels."""
        flat = [lab if lab == "O" else lab.split("-")[-1] for lab in self.labels]
        return "\n".join(f"{tok}\t{lab}" for tok, lab in zip(self.tokens, flat))

    def is_empty(self) -> bool:       # renamed for clarity
        return not self.tokens


def read_dir(folder: Path) -> List[Sentence]:
    """Load every *.tsv in `folder`, skipping malformed lines."""
    sentences, current = [], Sentence()
    for fp in sorted(folder.glob("*.tsv")):
        with fp.open(encoding="utf‑8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line:                         # sentence boundary
                    if not current.is_empty():
                        sentences.append(current)
                        current = Sentence()
                    continue
                if line.count("\t") != 1:            # skip bad rows silently
                    continue
                token, label = line.split("\t")
                current.add(token, label)
    if not current.is_empty():
        sentences.append(current)
    return sentences


def split(
    sents: List[Sentence], test_size: float, val_size: float, seed: int
) -> Tuple[List[Sentence], List[Sentence], List[Sentence]]:
    random.seed(seed)
    random.shuffle(sents)
    test_cut = int(len(sents) * (1 - test_size))
    val_cut = int(test_cut * (1 - val_size))
    return sents[:val_cut], sents[val_cut:test_cut], sents[test_cut:]


def write(path: Path, sents: List[Sentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf‑8") as f:
        for i, sent in enumerate(sents):
            if i:
                f.write("\n\n")
            f.write(sent.to_tsv())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dirs", nargs="+", type=Path, required=True)
    p.add_argument("--train_output", type=Path, required=True)
    p.add_argument("--test_output", type=Path, required=True)
    p.add_argument("--val_output", type=Path)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sentences: List[Sentence] = []
    for d in args.input_dirs:
        sentences.extend(read_dir(d))

    train, val, test = split(sentences, args.test_size, args.val_size, args.seed)
    write(args.train_output, train)
    write(args.test_output, test)
    if args.val_output and val:
        write(args.val_output, val)


if __name__ == "__main__":
    main()
