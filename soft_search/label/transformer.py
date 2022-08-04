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
DEFAULT_BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
TRANSFORMER_LABEL_COL = "transformer_label"

###############################################################################


def train(
    df: Union[str, Path, pd.DataFrame],
    label_col: str,
    text_col: str = NSFFields.abstractText,
    model_storage_dir: Union[str, Path] = DEFAULT_SOFT_SEARCH_TRANSFORMER_PATH,
    base_model: str = DEFAULT_BASE_MODEL,
) -> Path:
    """
    Fine-tune a transformer model to classify the provided labels.

    Parameters
    ----------
    df: Union[str, Path, pd.DataFrame]
        The data to use for training.
        Only CSV file format is supported when providing a file path.
    label_col: str
        The column name which contains the labels.
    text_col: str
        The column name which contains the raw text.
        Default: "abstractText"
    model_storage_dir: Union[str, Path]
        The path to store the model to.
        Default: "soft-search-transformer/"
    base_model: str
        The base model to fine-tune.
        Default: "distilbert-base-uncased-finetuned-sst-2-english"

    Returns
    -------
    Path
        The path to the stored model.

    Examples
    --------
    Example training from supplied manually labelled data.

    >>> from soft_search.data import load_soft_search_2022
    >>> from soft_search.label import transformer
    >>> df = load_soft_search_2022()
    >>> model = transformer.train(
    ...     df,
    ...     label_col="stated_software_will_be_created",
    ...     text_col="abstractText"
    ... )

    See Also
    --------
    label
        A function to apply a model across a pandas DataFrame.
    """
    # Handle storage dir
    model_storage_dir = Path(model_storage_dir).resolve()

    # Read DataFrame
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df)

    # Rename cols
    df = df.copy(deep=True)
    df = df[[label_col, text_col]]
    df = df.rename(columns={label_col: "label", text_col: "text"})
    label_names = df["label"].unique().tolist()

    # Construct label to id and vice-versa LUTs
    label2id, id2label = dict(), dict()
    for i, label in enumerate(label_names):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Cast to dataset
    dataset = Dataset.from_pandas(
        df,
        features=Features(
            label=ClassLabel(num_classes=len(id2label), names=label_names),
            text=Value(dtype="string"),
        ),
    )

    # Get splits
    dataset_dict = dataset.train_test_split(test_size=0.25, stratify_by_column="label")

    # Preprocess
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def preprocess_function(examples: "BatchEncoding") -> "Batch":
        return tokenizer(examples["text"], truncation=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_dataset_dict = dataset_dict.map(preprocess_function, batched=True)

    # AutoModel
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

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
        num_train_epochs=5,
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
    return classifier(text, truncation=True, top_k=1)[0]["label"]


def label(
    df: pd.DataFrame,
    apply_column: str = NSFFields.abstractText,
    label_column: str = TRANSFORMER_LABEL_COL,
    model: Union[str, Path] = DEFAULT_SOFT_SEARCH_TRANSFORMER_PATH,
) -> pd.DataFrame:
    """
    In-place add a new column to the provided pandas DataFrame with a label
    of software predicted or not using a trained transformer model.

    Parameters
    ----------
    df: pd.DataFrame
        The pandas DataFrame to in-place add a column with the
        software predicted outcome labels.
    apply_column: str
        The column to use for "prediction".
        Default: "abstractText"
    label_column: str
        The name of the column to add with outcome "prediction".
        Default: "transformer_label"
    model: Union[str, Path]
        The path to the stored model.
        Default: "soft-search-transformer/"

    Returns
    -------
    pd.DataFrame
        The same pandas DataFrame but with a new column added in-place containing
        the software outcome prediction.

    See Also
    --------
    soft_search.nsf.get_nsf_dataset
        Function to get an NSF dataset for prediction.
    """
    # Load label pipeline
    classifier = pipeline("text-classification", model=str(model), tokenizer=str(model))

    # Partial func
    apply_classifier = partial(_apply_transformer, classifier=classifier)
    df[label_column] = df[apply_column].apply(apply_classifier)
    return df
