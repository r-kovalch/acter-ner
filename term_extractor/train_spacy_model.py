"""Training script for NER using Spacy."""

import random
from pathlib import Path
from typing import List, Tuple, Optional
import spacy
from spacy.tokens import Doc, DocBin
from spacy.training import Example
import typer
from wasabi import msg


class IOBDataConverter:
    """Converter for IOB data to Spacy format."""

    def __init__(self, lang: str = "en"):
        """Initialize converter.

        Args:
            lang: Language code for Spacy
        """
        self.nlp = spacy.blank(lang)
        # Create NER pipe and add to pipeline
        if "ner" not in self.nlp.pipe_names:
            self.nlp.add_pipe("ner", last=True)

    def read_iob_file(self, file_path: Path) -> List[Tuple[List[str], List[str]]]:
        """Read IOB file and return sentences with labels.

        Args:
            file_path: Path to IOB file

        Returns:
            List of (tokens, labels) tuples
        """
        sentences: List[Tuple[List[str], List[str]]] = []
        current_tokens: List[str] = []
        current_labels: List[str] = []

        with Path(file_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split("\t")
                    current_tokens.append(token)
                    current_labels.append(label)
                elif current_tokens:  # End of sentence
                    sentences.append((current_tokens, current_labels))
                    current_tokens = []
                    current_labels = []

            if current_tokens:  # Handle last sentence
                sentences.append((current_tokens, current_labels))

        return sentences

    def iob_to_spans(self, tokens: List[str], labels: List[str]) -> List[Tuple[int, int, str]]:
        """Convert IOB labels to entity spans.

        Args:
            tokens: List of tokens
            labels: List of IOB labels

        Returns:
            List of (start, end, label) tuples
        """
        spans = []
        start = None
        current_label = None

        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith("B-"):
                if start is not None:
                    spans.append((start, i, current_label))
                start = i
                current_label = label[2:]  # Remove B- prefix
            elif label.startswith("I-"):
                if start is None:
                    start = i
                    current_label = label[2:]  # Remove I- prefix
            else:  # O label
                if start is not None:
                    spans.append((start, i, current_label))
                    start = None
                    current_label = None

        if start is not None:  # Handle entity at end of sentence
            spans.append((start, len(tokens), current_label))

        return spans

    def convert_to_spacy(
        self, file_path: Path, output_path: Optional[Path] = None
    ) -> List[Example]:
        """Convert IOB file to Spacy training examples.

        Args:
            file_path: Path to IOB file
            output_path: Optional path to save DocBin

        Returns:
            List of Spacy training examples
        """
        sentences = self.read_iob_file(file_path)
        examples = []
        doc_bin = DocBin()

        for tokens, labels in sentences:
            doc = Doc(self.nlp.vocab, words=tokens)
            ents = self.iob_to_spans(tokens, labels)
            doc.ents = [
                doc.char_span(
                    doc[start].idx, doc[end - 1].idx + len(doc[end - 1].text), label=label
                )
                for start, end, label in ents
            ]

            if output_path:
                doc_bin.add(doc)
            else:
                examples.append(Example.from_dict(doc, {"entities": ents}))

        if output_path:
            doc_bin.to_disk(output_path)

        return examples


def train_ner(
    train_path: Path,
    output_dir: Path,
    eval_path: Optional[Path] = None,
    n_iter: int = 30,
    batch_size: int = 1000,
    dropout: float = 0.2,
    lang: str = "en",
) -> None:
    """Train Spacy NER model.

    Args:
        train_path: Path to training data
        output_dir: Path to save model
        eval_path: Optional path to evaluation data
        n_iter: Number of training iterations
        batch_size: Batch size for training
        dropout: Dropout rate
        lang: Language code
    """
    output_dir = Path(output_dir)

    if output_dir.exists():
        msg.warn(f"Output directory {output_dir} already exists")

    msg.info("Converting data to Spacy format...")
    converter = IOBDataConverter(lang)
    train_examples = converter.convert_to_spacy(train_path)

    if eval_path:
        eval_examples = converter.convert_to_spacy(eval_path)
    else:
        eval_examples = None

    msg.info("Creating NER pipeline...")
    nlp = spacy.blank(lang)

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add labels from training data
    for example in train_examples:
        for ent in example.reference.ents:
            ner.add_label(ent.label_)

    # Get names of other pipes to disable during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    msg.info("Starting training...")
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        for itn in range(n_iter):
            random.shuffle(train_examples)
            losses = {}

            # Batch up the examples
            for batch in spacy.util.minibatch(train_examples, size=batch_size):
                nlp.update(
                    batch,
                    drop=dropout,
                    losses=losses,
                )

            msg.info(f"Iteration {itn+1}: Losses = {losses}")

            # Evaluate on dev set
            if eval_examples:
                scorer = nlp.evaluate(eval_examples)
                msg.info(f"Evaluation scores: {scorer.scores}")

    # Save model
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    nlp.to_disk(output_dir)
    msg.good(f"Saved model to {output_dir}")


def main(
    train_path: Path = typer.Argument(..., help="Path to training data"),
    output_dir: Path = typer.Argument(..., help="Path to save model"),
    eval_path: Optional[Path] = typer.Option(None, help="Path to evaluation data"),
    n_iter: int = typer.Option(30, help="Number of training iterations"),
    batch_size: int = typer.Option(1000, help="Batch size for training"),
    dropout: float = typer.Option(0.2, help="Dropout rate"),
    lang: str = typer.Option("en", help="Language code"),
) -> None:
    """Train a Spacy NER model on IOB data."""
    train_ner(
        train_path=train_path,
        output_dir=output_dir,
        eval_path=eval_path,
        n_iter=n_iter,
        batch_size=batch_size,
        dropout=dropout,
        lang=lang,
    )


if __name__ == "__main__":
    typer.run(main)
