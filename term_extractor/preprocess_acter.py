"""Script for preparing IOB datasets from multiple TSV files."""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random


class SentenceTokens:
    """Class to store tokens and labels for a sentence."""
    
    def __init__(self) -> None:
        """Initialize empty lists for tokens and labels."""
        self.tokens: List[str] = []
        self.labels: List[str] = []
        
    def add_token(self, token: str, label: str) -> None:
        """Add token and label to sentence.
        
        Args:
            token: Token text
            label: IOB label
        """
        self.tokens.append(token)
        self.labels.append(label)
        
    def is_empty(self) -> bool:
        """Check if sentence is empty.
        
        Returns:
            True if sentence has no tokens
        """
        return len(self.tokens) == 0
    
    def to_tsv_lines(self) -> List[str]:
        """Convert sentence to TSV lines.
        
        Returns:
            List of TSV formatted lines
        """
        return [f"{t}\t{l + '-TERM' if l != 'O' else l}" for t, l in zip(self.tokens, self.labels)]


class IOBDatasetProcessor:
    """Processor for IOB formatted datasets."""
    
    def __init__(
        self,
        input_dirs: List[Path],
        train_output: Path,
        test_output: Path,
        val_output: Optional[Path] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42
    ):
        """Initialize dataset processor.
        
        Args:
            input_dirs: List of directories containing IOB files
            train_output: Path to output train file
            test_output: Path to output test file
            val_output: Optional path to output validation file
            test_size: Fraction of data to use for testing
            val_size: Fraction of data to use for validation
            seed: Random seed for splitting
        """
        self.input_dirs = input_dirs
        self.train_output = train_output
        self.test_output = test_output
        self.val_output = val_output
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        random.seed(seed)
        
        # Create output directories if they don't exist
        self.train_output.parent.mkdir(parents=True, exist_ok=True)
        self.test_output.parent.mkdir(parents=True, exist_ok=True)
        if self.val_output:
            self.val_output.parent.mkdir(parents=True, exist_ok=True)

    def read_iob_file(self, file_path: Path) -> List[SentenceTokens]:
        """Read IOB formatted file and split into sentences.
        
        Args:
            file_path: Path to IOB file
            
        Returns:
            List of sentences with tokens and labels
        """
        sentences: List[SentenceTokens] = []
        current_sentence = SentenceTokens()
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                
                if not line:  # Empty line indicates sentence boundary
                    if not current_sentence.is_empty():
                        sentences.append(current_sentence)
                        current_sentence = SentenceTokens()
                    continue
                    
                try:
                    token, label = line.split("\t")
                    current_sentence.add_token(token, label)
                except ValueError as e:
                    print(f"Warning: Skipping malformed line in {file_path}: {line}")
                    continue
        
        # Add last sentence if not empty
        if not current_sentence.is_empty():
            sentences.append(current_sentence)
            
        return sentences

    def process_directory(self, directory: Path) -> List[SentenceTokens]:
        """Process all IOB files in a directory.
        
        Args:
            directory: Directory containing IOB files
            
        Returns:
            List of sentences from all files
        """
        sentences: List[SentenceTokens] = []
        
        for file_path in directory.glob("*.tsv"):
            try:
                file_sentences = self.read_iob_file(file_path)
                sentences.extend(file_sentences)
                print(f"Processed {file_path}: {len(file_sentences)} sentences")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
                
        return sentences

    def split_sentences(
        self, 
        sentences: List[SentenceTokens]
    ) -> Tuple[List[SentenceTokens], List[SentenceTokens], List[SentenceTokens]]:
        """Split sentences into train/val/test sets.
        
        Args:
            sentences: List of all sentences
            
        Returns:
            Tuple of (train_sentences, val_sentences, test_sentences)
        """
        random.shuffle(sentences)
        
        test_idx = int(len(sentences) * (1 - self.test_size))
        val_idx = int(test_idx * (1 - self.val_size))
        
        if self.val_output:
            train = sentences[:val_idx]
            val = sentences[val_idx:test_idx]
            test = sentences[test_idx:]
        else:
            train = sentences[:test_idx]
            val = []
            test = sentences[test_idx:]
            
        return train, val, test

    def write_sentences(self, sentences: List[SentenceTokens], output_path: Path) -> None:
        """Write sentences to IOB formatted file.
        
        Args:
            sentences: List of sentences to write
            output_path: Path to output file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for i, sentence in enumerate(sentences):
                if i > 0:
                    f.write("\n\n")  # Empty line between sentences
                lines = sentence.to_tsv_lines()
                f.write("\n".join(lines))

    def process(self) -> None:
        """Process all input directories and create train/val/test splits."""
        # Collect all sentences
        all_sentences: List[SentenceTokens] = []
        
        for directory in self.input_dirs:
            if not directory.exists():
                print(f"Warning: Directory {directory} does not exist")
                continue
                
            print(f"\nProcessing directory: {directory}")
            sentences = self.process_directory(directory)
            all_sentences.extend(sentences)
            
        print(f"\nTotal sentences collected: {len(all_sentences)}")
        
        # Split sentences
        train_sentences, val_sentences, test_sentences = self.split_sentences(all_sentences)
        
        print(f"\nSplit sizes:")
        print(f"Train: {len(train_sentences)} sentences")
        if self.val_output:
            print(f"Validation: {len(val_sentences)} sentences")
        print(f"Test: {len(test_sentences)} sentences")
        
        # Write output files
        print("\nWriting output files...")
        self.write_sentences(train_sentences, self.train_output)
        self.write_sentences(test_sentences, self.test_output)
        if self.val_output and val_sentences:
            self.write_sentences(val_sentences, self.val_output)
            
        print("\nDataset preparation completed!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Prepare IOB datasets from multiple TSV files"
    )
    
    parser.add_argument(
        "--input_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Directories containing IOB files",
    )
    parser.add_argument(
        "--train_output",
        type=Path,
        required=True,
        help="Path to output train file",
    )
    parser.add_argument(
        "--test_output",
        type=Path,
        required=True,
        help="Path to output test file",
    )
    parser.add_argument(
        "--val_output",
        type=Path,
        help="Optional path to output validation file",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    
    return parser.parse_args()


def main() -> None:
    """Run dataset preparation pipeline."""
    args = parse_args()
    
    processor = IOBDatasetProcessor(
        input_dirs=args.input_dirs,
        train_output=args.train_output,
        test_output=args.test_output,
        val_output=args.val_output,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    
    processor.process()


if __name__ == "__main__":
    main()