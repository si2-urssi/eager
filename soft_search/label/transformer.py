#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..constants import NSFFields

try:
    from datasets import ClassLabel, Dataset, Features, Value, load_metric
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EvalPrediction,
        Trainer,
        TrainingArguments,
        pipeline,
    )
except ImportError:
    raise ImportError(
        "Extra dependencies are needed for the `label.transformer` "
        "submodule of `soft-search`. "
        "Install with `pip install soft-search[transformer]`."
    )

if TYPE_CHECKING:
    from datasets.arrow_dataset import Batch
    from transformers.pipelines.base import Pipeline
    from transformers.tokenization_utils_base import BatchEncoding

###############################################################################

DEFAULT_SOFT_SEARCH_TRANSFORMER_PATH = Path("soft-search-transformer/").resolve()
TRANSFORMER_LABEL_COL = "transformer_label"

###############################################################################


def train(
    df: pd.DataFrame,
    label_col: str,
    text_col: str = NSFFields.abstractText,
    model_storage_dir: Union[str, Path] = DEFAULT_SOFT_SEARCH_TRANSFORMER_PATH,
    base_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
) -> Path:
    # Handle storage dir
    model_storage_dir = Path(model_storage_dir).resolve()

    # Rename cols
    df = df.copy(deep=True)
    df = df[[label_col, text_col]]
    df = df.rename(columns={label_col: "label", text_col: "text"})

    # Cast to dataset
    dataset = Dataset.from_pandas(
        df,
        features=Features(
            label=ClassLabel(names=df["label"].unique().tolist()),
            text=Value(dtype="string"),
        ),
    )

    # Get splits
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column="label")

    # Preprocess
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def preprocess_function(examples: "BatchEncoding") -> "Batch":
        return tokenizer(examples["text"], truncation=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_dataset_dict = dataset_dict.map(preprocess_function, batched=True)

    # AutoModel
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    # Training Args
    training_args = TrainingArguments(
        output_dir=model_storage_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Compute accuracy metrics
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred: EvalPrediction) -> Optional[Dict]:
        predictions = np.argmax(eval_pred.predictions, axis=-1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_dict["train"],
        eval_dataset=tokenized_dataset_dict["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Store model
    trainer.save_model()
    return model_storage_dir


def _apply_transformer(text: str, classifier: "Pipeline") -> str:
    return classifier(text, top_k=1)[0]


def label(
    df: pd.DataFrame,
    apply_column: str = NSFFields.abstractText,
    label_column: str = TRANSFORMER_LABEL_COL,
    model: Union[str, Path] = DEFAULT_SOFT_SEARCH_TRANSFORMER_PATH,
) -> pd.DataFrame:
    # Load label pipeline
    classifier = pipeline("text-classification", model=str(model))

    # Partial func
    apply_classifier = partial(_apply_transformer, classifier=classifier)
    df[label_column] = df[apply_column].apply(apply_classifier)
    return df
