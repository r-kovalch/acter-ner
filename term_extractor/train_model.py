"""Training script for term extraction model."""

import argparse
from pathlib import Path

from dataset_processor import (
    ACTERProcessor,
    DatasetConfig,
    HyperparameterOptimizer,
    ModelConfig,
    TermExtractionTrainer,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train term extraction model")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name/path of pretrained model",
    )
    parser.add_argument(
        "--train_path",
        type=Path,
        required=True,
        help="Path to training data",
    )
    parser.add_argument(
        "--val_path",
        type=Path,
        help="Path to validation data",
    )
    parser.add_argument(
        "--test_path",
        type=Path,
        help="Path to test data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--optimize_hyperparams",
        action="store_true",
        help="Run hyperparameter optimization",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of optimization trials",
    )

    return parser.parse_args()


def main() -> None:
    """Run training pipeline."""
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize configs
    dataset_config = DatasetConfig(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    model_config = ModelConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
    )

    # Initialize processor
    processor = ACTERProcessor(dataset_config)

    if args.optimize_hyperparams:
        # Run hyperparameter optimization
        print("Starting hyperparameter optimization...")
        optimizer = HyperparameterOptimizer(
            model_config,
            dataset_config,
            processor,
            n_trials=args.n_trials,
        )
        best_params = optimizer.optimize()

        # Update model config with best params
        for param, value in best_params.items():
            setattr(model_config, param, value)

        print(f"Best hyperparameters found: {best_params}")

    # Train model with best/default params
    print("Starting model training...")
    trainer = TermExtractionTrainer(
        model_config,
        dataset_config,
        processor,
    )
    trainer = trainer.train()

    # Save final model
    trainer.save_model(str(args.output_dir / "final_model"))
    trainer.save_state()
    print(f"Model saved to {args.output_dir}")

    # Evaluate on test set if available
    if args.test_path:
        print("Evaluating on test set...")
        metrics = trainer.evaluate(trainer.prepare_datasets()[2])
        print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()
