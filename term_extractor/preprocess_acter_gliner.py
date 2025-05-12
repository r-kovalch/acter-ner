#!/usr/bin/env python
"""
Create train / test (/ val) TSVs from ACTER IOB files
for later conversion with `spacy convert --converter iob`.

Example:
  python preprocess_acter_gliner.py \
         --input_dirs data/acter/en/*/iob_annotations/without_named_entities \
         --train_output train_full.tsv \
         --test_output  test_full.tsv \
         --val_size 0.1
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
        """Return the sentence in token–label TSV format (keeps IOB tags)."""
        return "\n".join(f"{tok}\t{lab}"
                         for tok, lab in zip(self.tokens, self.labels))

    def is_empty(self) -> bool:
        return not self.tokens


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def read_dir(folder: Path) -> List[Sentence]:
    sentences, cur = [], Sentence()

    for fp in sorted(folder.glob("*.tsv")):
        with fp.open(encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line:                  # blank line → new sentence
                    if not cur.is_empty():
                        sentences.append(cur)
                        cur = Sentence()
                    continue
                if line.count("\t") != 1:     # skip malformed row
                    continue

                token, label = line.split("\t")

                # ---- FIX orphan tags --------------------------------------
                if label == "B":
                    label = "B-TERM"
                elif label == "I":
                    label = "I-TERM"
                # -----------------------------------------------------------

                cur.add(token, label)

    if not cur.is_empty():
        sentences.append(cur)
    return sentences



def split(
    sents: List[Sentence],
    test_size: float,
    val_size: float,
    seed: int
) -> Tuple[List[Sentence], List[Sentence], List[Sentence]]:
    """Shuffle and perform train / val / test split."""
    random.seed(seed)
    random.shuffle(sents)

    test_cut = int(len(sents) * (1 - test_size))
    val_cut = int(test_cut * (1 - val_size))

    train = sents[:val_cut]
    val   = sents[val_cut:test_cut]
    test  = sents[test_cut:]
    return train, val, test


def write(path: Path, sentences: List[Sentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, sent in enumerate(sentences):
            if i:
                f.write("\n\n")
        f.write(sent.to_tsv())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dirs", nargs="+", type=Path, required=True,
                   help="Folders that contain *.tsv ACTER files")
    p.add_argument("--train_output", type=Path, required=True)
    p.add_argument("--test_output",  type=Path, required=True)
    p.add_argument("--val_output",   type=Path, help="If omitted, no dev split")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size",  type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # gather data
    sentences: List[Sentence] = []
    for directory in args.input_dirs:
        sentences.extend(read_dir(directory))

    # shuffle & split
    train, val, test = split(sentences, args.test_size, args.val_size, args.seed)

    # write out
    write(args.train_output, train)
    write(args.test_output,  test)
    if args.val_output and val:
        write(args.val_output, val)


if __name__ == "__main__":
    main()
