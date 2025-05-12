#!/usr/bin/env python
"""
Pre‑process ACTER TSV files for GLiNER fine‑tuning.

Changes vs. original
--------------------
* Skip lines that lack a real <TAB> (root cause of the unpack error).
* Use '\n', not '\\n', when writing out files.
* Split with maxsplit=1 so extra tabs inside a token don’t break parsing.
* Provide --verbose to print progress every N sentences.
"""
import argparse
import random
from pathlib import Path
from typing import List, Tuple

###############################################################################
# Helper class
###############################################################################

class Sentence:
    def __init__(self):
        self.tokens: List[str] = []
        self.labels: List[str] = []

    def add(self, token: str, label: str) -> None:
        self.tokens.append(token)
        self.labels.append(label)

    def to_tsv(self) -> str:
        """Return the sentence as IOB‑free TSV."""
        flat = [lab if lab == "O" else lab.split("-")[-1] for lab in self.labels]
        return "\n".join(f"{tok}\t{lab}" for tok, lab in zip(self.tokens, flat))

    def is_empty(self) -> bool:
        return not self.tokens

###############################################################################
# Low‑level file helpers
###############################################################################

def read_dir(folder: Path) -> List[Sentence]:
    """Read every *.tsv in *folder* and return a list of Sentence objects."""
    sentences: List[Sentence] = []
    current = Sentence()

    for fp in sorted(folder.glob("*.tsv")):
        with fp.open(encoding="utf8") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if not line:                              # blank line = sentence boundary
                    if not current.is_empty():
                        sentences.append(current)
                        current = Sentence()
                    continue
                if "\t" not in line:                      # malformed → skip
                    continue
                token, label = line.split("\t", 1)        # split once, ignore extra tabs
                current.add(token, label)

    if not current.is_empty():                            # flush last sentence
        sentences.append(current)
    return sentences


def split(
    sentences: List[Sentence],
    test_size: float,
    val_size: float,
    seed: int
) -> Tuple[List[Sentence], List[Sentence], List[Sentence]]:
    """Train/val/test split with reproducible shuffling."""
    random.seed(seed)
    random.shuffle(sentences)
    test_cut = int(len(sentences) * (1 - test_size))
    val_cut = int(test_cut * (1 - val_size))
    return sentences[:val_cut], sentences[val_cut:test_cut], sentences[test_cut:]


def write(path: Path, sents: List[Sentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as fh:
        for i, sent in enumerate(sents):
            if i:
                fh.write("\n\n")          # real blank line between sentences
            fh.write(sent.to_tsv())

###############################################################################
# CLI
###############################################################################

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dirs", nargs="+", type=Path, required=True)
    p.add_argument("--train_output", type=Path, required=True)
    p.add_argument("--test_output", type=Path, required=True)
    p.add_argument("--val_output", type=Path)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true", help="Print progress every 1 000 sents")
    args = p.parse_args()

    sentences: List[Sentence] = []
    for i, directory in enumerate(args.input_dirs, 1):
        dir_sents = read_dir(directory)
        sentences.extend(dir_sents)
        if args.verbose:
            print(f"Loaded {len(dir_sents):,} sentences from #{i} {directory}")

    train, val, test = split(sentences, args.test_size, args.val_size, args.seed)

    if args.verbose:
        print(f"⤷ Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    write(args.train_output, train)
    write(args.test_output, test)
    if args.val_output and val:
        write(args.val_output, val)


if __name__ == "__main__":
    main()
