import argparse
from pathlib import Path
import random
from typing import List, Tuple


class Sentence:
    def __init__(self):
        self.tokens: List[str] = []
        self.labels: List[str] = []

    def add(self, token: str, label: str):
        self.tokens.append(token)
        self.labels.append(label)

    def to_tsv(self) -> str:
        # Strip BIO prefix and optional '-TERM' suffix
        flat_labels = [
            (lab if lab == "O" else lab.split("-")[-1].replace("TERM", "TERM"))
            for lab in self.labels
        ]
        return "\\n".join(f"{tok}\\t{lab}" for tok, lab in zip(self.tokens, flat_labels))

    def empty(self):
        return len(self.tokens) == 0


def read_dir(folder: Path) -> List[Sentence]:
    sents: List[Sentence] = []
    cur = Sentence()
    for fp in folder.glob("*.tsv"):
        with fp.open() as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    if not cur.empty():
                        sents.append(cur)
                        cur = Sentence()
                    continue
                tok, lab = line.split("\\t")
                cur.add(tok, lab)
    if not cur.empty():
        sents.append(cur)
    return sents


def split(sentences: List[Sentence], test_size: float, val_size: float, seed: int) -> Tuple[
    List[Sentence], List[Sentence], List[Sentence]]:
    random.seed(seed)
    random.shuffle(sentences)
    test_cut = int(len(sentences) * (1 - test_size))
    val_cut = int(test_cut * (1 - val_size))
    train, val, test = sentences[:val_cut], sentences[val_cut:test_cut], sentences[test_cut:]
    return train, val, test


def write(out: Path, sents: List[Sentence]):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, s in enumerate(sents):
            if i:
                f.write("\\n\\n")
            f.write(s.to_tsv())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dirs", nargs="+", type=Path, required=True)
    p.add_argument("--train_output", type=Path, required=True)
    p.add_argument("--test_output", type=Path, required=True)
    p.add_argument("--val_output", type=Path)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    all_sents: List[Sentence] = []
    for d in args.input_dirs:
        all_sents.extend(read_dir(d))

    train, val, test = split(all_sents, args.test_size, args.val_size, args.seed)

    write(args.train_output, train)
    write(args.test_output, test)
    if args.val_output and val:
        write(args.val_output, val)


if __name__ == "__main__":
    main()
