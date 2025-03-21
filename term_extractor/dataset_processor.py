"""Term extraction training pipeline using HuggingFace transformers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


import optuna
from datasets import Dataset

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""

    train_path: Path
    val_path: Optional[Path]
    test_path: Optional[Path]
    max_length: int = 512
    batch_size: int = 16
    num_workers: int = 4


class ACTERProcessor:
    """Processor for ACTER dataset in IOB format."""

    def __init__(self, config: DatasetConfig):
        """Initialize ACTER processor.

        Args:
            config: Dataset configuration
        """
        self.config = config
        self.label2id = {"O": 0, "B": 1, "I": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def read_iob_file(self, file_path: Path) -> List[Dict[str, List]]:
        """Read IOB formatted file and return list of examples.

        Args:
            file_path: Path to IOB file

        Returns:
            List of examples with tokens and labels
        """
        examples = []
        current_tokens = []
        current_labels = []

        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split("\t")
                    current_tokens.append(token)
                    current_labels.append(label)
                elif current_tokens:
                    examples.append({"tokens": current_tokens, "labels": current_labels})
                    current_tokens = []
                    current_labels = []

        if current_tokens:
            examples.append({"tokens": current_tokens, "labels": current_labels})

        return examples

    def create_dataset(self, examples: List[Dict[str, List]]) -> Dataset:
        """Create HuggingFace dataset from examples.

        Args:
            examples: List of examples with tokens and labels

        Returns:
            HuggingFace Dataset
        """
        return Dataset.from_dict(
            {
                "tokens": [ex["tokens"] for ex in examples],
                "labels": [ex["labels"] for ex in examples],
            }
        )


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def tokenize(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize examples.

        Args:
            examples: Dictionary with tokens and labels

        Returns:
            Dictionary with input_ids, attention_mask and labels
        """
        pass


class TransformerTokenizer(BaseTokenizer):
    """Tokenizer using HuggingFace transformers."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, label2id: Dict[str, int]):
        """Initialize transformer tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            label2id: Label to ID mapping
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def tokenize(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize examples.

        Args:
            examples: Dictionary with tokens and labels

        Returns:
            Dictionary with input_ids, attention_mask and labels
        """
        # Process one example at a time to ensure proper alignment
        encoded_inputs = []
        for i in range(len(examples["tokens"])):
            tokenized = self.tokenizer(
                examples["tokens"][i],
                is_split_into_words=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )

            # Align labels with tokens
            word_ids = tokenized.word_ids()
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id[examples["labels"][i][word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            tokenized["labels"] = label_ids
            encoded_inputs.append(tokenized)

        # Combine all examples into a batch
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for encoding in encoded_inputs:
            for key in batch.keys():
                batch[key].append(encoding[key])

        return batch


@dataclass
class ModelConfig:
    """Configuration for model training."""

    model_name: str
    output_dir: Path
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    fp16: bool = False


class TermExtractionTrainer:
    """Trainer for term extraction model."""

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        processor: ACTERProcessor,
    ):
        """Initialize trainer.

        Args:
            model_config: Model configuration
            dataset_config: Dataset configuration
            processor: Dataset processor
        """
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.processor = processor

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name, add_prefix_space=True
        )
        self.transformer_tokenizer = TransformerTokenizer(
            self.tokenizer, dataset_config.max_length, processor.label2id
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_config.model_name,
            num_labels=len(processor.label2id),
            id2label=processor.id2label,
            label2id=processor.label2id,
        )

    def prepare_datasets(self) -> tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        """Prepare train, validation and test datasets.

        Returns:
            Tuple of datasets (train, val, test)
        """
        train_examples = self.processor.read_iob_file(self.dataset_config.train_path)
        train_dataset = self.processor.create_dataset(train_examples)
        train_dataset = train_dataset.map(
            self.transformer_tokenizer.tokenize,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        val_dataset = None
        if self.dataset_config.val_path:
            val_examples = self.processor.read_iob_file(self.dataset_config.val_path)
            val_dataset = self.processor.create_dataset(val_examples)
            val_dataset = val_dataset.map(
                self.transformer_tokenizer.tokenize,
                batched=True,
                remove_columns=val_dataset.column_names,
            )

        test_dataset = None
        if self.dataset_config.test_path:
            test_examples = self.processor.read_iob_file(self.dataset_config.test_path)
            test_dataset = self.processor.create_dataset(test_examples)
            test_dataset = test_dataset.map(
                self.transformer_tokenizer.tokenize,
                batched=True,
                remove_columns=test_dataset.column_names,
            )

        return train_dataset, val_dataset, test_dataset

    def train(self, trial: Optional[optuna.Trial] = None) -> Union[float, Trainer]:
        """Train the model.

        Args:
            trial: Optuna trial for hyperparameter optimization

        Returns:
            Validation loss if trial is provided, otherwise trained Trainer
        """
        if trial:
            # Hyperparameter search space
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
            weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
            warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
            num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)
        else:
            learning_rate = self.model_config.learning_rate
            weight_decay = self.model_config.weight_decay
            warmup_ratio = self.model_config.warmup_ratio
            num_train_epochs = self.model_config.num_train_epochs

        training_args = TrainingArguments(
            output_dir=str(self.model_config.output_dir),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=self.dataset_config.batch_size,
            per_device_eval_batch_size=self.dataset_config.batch_size,
            evaluation_strategy="epoch" if self.dataset_config.val_path else "no",
            save_strategy="epoch" if self.dataset_config.val_path else "no",
            fp16=self.model_config.fp16,
            report_to="none",
        )

        train_dataset, val_dataset, _ = self.prepare_datasets()

        # Initialize data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, padding=True, return_tensors="pt"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        if trial:
            # Return validation loss for optimization
            metrics = trainer.evaluate()
            return metrics["eval_loss"]

        return trainer


class HyperparameterOptimizer:
    """Optimizer for hyperparameter search."""

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        processor: ACTERProcessor,
        n_trials: int = 20,
    ):
        """Initialize optimizer.

        Args:
            model_config: Model configuration
            dataset_config: Dataset configuration
            processor: Dataset processor
            n_trials: Number of optimization trials
        """
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.processor = processor
        self.n_trials = n_trials

    def optimize(self) -> Dict[str, float]:
        """Run hyperparameter optimization.

        Returns:
            Dictionary with best hyperparameters
        """
        study = optuna.create_study(direction="minimize")

        def objective(trial: optuna.Trial) -> float:
            trainer = TermExtractionTrainer(self.model_config, self.dataset_config, self.processor)
            return trainer.train(trial)

        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params
