import argparse
from pathlib import Path
from typing import List, Optional, Tuple


from preprocess_acter import IOBDatasetProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory with the input files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--normalize_spaces",
        default=False,
        help="Whether to normalize spaces in the text before saving",
    )

    args = parser.parse_args()
    # Create the output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Creating the processor
    # We are bullshitting the processor to not write anything to disk
    # because we are not interested in the train/test split
    processor = IOBDatasetProcessor(
        input_dirs=[args.input_dir],
        train_output=Path("/dev/null"),
        test_output=Path("/dev/null"),
    )

    sentences = processor.process_directory(args.input_dir)
    print(f"Number of sentences: {len(sentences)}")
    print(
        f"Number of sentences with labels: {len([s for s in sentences if s.get_labels()])}"
    )
    print(f"Number of labels: {sum([len(s.get_labels()) for s in sentences])}")

    processor.save_text_and_labels(
        sentences=sentences,
        output_path=args.output_dir,
        normalize_spaces=args.normalize_spaces,
    )
