#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    pipeline,
)

if TYPE_CHECKING:
    from datasets.arrow_dataset import Batch
    from transformers.pipelines.base import Pipeline
    from transformers.tokenization_utils_base import BatchEncoding

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

DEFAULT_SOFT_SEARCH_TRANSFORMER_PATH = Path("soft-search-transformer/").resolve()
DEFAULT_BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
TRANSFORMER_LABEL_COL = "transformer_label"
HUGGINGFACE_HUB_SOFT_SEARCH_MODEL = "evamaxfield/soft-search"

###############################################################################


def _train(
    df: Union[str, Path, pd.DataFrame],
    label_col: str = "label",
    text_col: str = "text",
    model_storage_dir: Union[str, Path] = DEFAULT_SOFT_SEARCH_TRANSFORMER_PATH,
    base_model: str = DEFAULT_BASE_MODEL,
    extra_training_args: Dict[str, Any] = {},
) -> Tuple[Path, Trainer]:
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
    dataset = Dataset.from_pandas(df)
    dataset = dataset.class_encode_column("label")

    # Get splits
    dataset_dict = dataset.train_test_split(test_size=0.25, stratify_by_column="label")

    # Log splits
    log.info(
        f"Training dataset splits:\n"
        f"\t'train': {len(dataset_dict['train'])}\n"
        f"\t'test': {len(dataset_dict['test'])}"
    )

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
        **extra_training_args,
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
    return model_storage_dir, trainer


def train(
    df: Union[str, Path, pd.DataFrame],
    label_col: str = "label",
    text_col: str = "text",
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
        Default: "label"
    text_col: str
        The column name which contains the raw text.
        Default: "text"
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

    >>> from soft_search.data import load_joined_soft_search_2022
    >>> from soft_search.label import transformer
    >>> from sklearn.model_selection import train_test_split
    >>> df = load_joined_soft_search_2022()
    >>> train, test = train_test_split(
    ...     df,
    ...     test_size=0.3,
    ...     stratify=df["label"]
    ... )
    >>> model = transformer.train(train)

    See Also
    --------
    label
        A function to apply a model across a pandas DataFrame.
    """
    model_storage_dir, _ = _train(
        df=df,
        label_col=label_col,
        text_col=text_col,
        model_storage_dir=model_storage_dir,
        base_model=base_model,
    )
    return model_storage_dir


def _train_and_upload_transformer(seed: int = 0) -> Path:
    import os
    import random

    import numpy as np
    import torch

    from soft_search.data import load_joined_soft_search_2022

    # Set a bunch of seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Load data, train
    df = load_joined_soft_search_2022()
    model, _ = _train(
        df,
        model_storage_dir=DEFAULT_SOFT_SEARCH_TRANSFORMER_PATH,
        extra_training_args=dict(
            push_to_hub=True,
            hub_model_id=HUGGINGFACE_HUB_SOFT_SEARCH_MODEL,
            hub_strategy="end",
            hub_token=os.environ["HUGGINGFACE_TOKEN"],
        ),
    )

    return model


def _apply_transformer(text: str, classifier: "Pipeline") -> str:
    return classifier(text, truncation=True, top_k=1)[0]["label"]


def label(
    df: pd.DataFrame,
    apply_column: str = "text",
    label_column: str = TRANSFORMER_LABEL_COL,
    model: Union[str, Path] = HUGGINGFACE_HUB_SOFT_SEARCH_MODEL,
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
        Default: "text"
    label_column: str
        The name of the column to add with outcome "prediction".
        Default: "transformer_label"
    model: Union[str, Path]
        The path to the stored model.
        Default: https://huggingface.co/evamaxfield/soft-search (latest CI model)

    Returns
    -------
    pd.DataFrame
        The same pandas DataFrame but with a new column added in-place containing
        the software outcome prediction.

    See Also
    --------
    soft_search.nsf.get_nsf_dataset
        Function to get an NSF dataset for prediction.

    Examples
    --------
    Example application to a new NSF dataset.

    >>> from soft_search import constants, nsf
    >>> from soft_search.label import transformer
    >>> df = nsf.get_nsf_dataset(
    ...     "2016-01-01",
    ...     "2017-01-01",
    ...     dataset_fields=[constants.NSFFields.abstractText],
    ... )
    >>> predicted = transformer.label(
    ...     df,
    ...     apply_column=constants.NSFFields.abstractText,
    ... )
    """
    # Load label pipeline
    classifier = pipeline("text-classification", model=str(model), tokenizer=str(model))

    # Partial func
    apply_classifier = partial(_apply_transformer, classifier=classifier)
    df[label_column] = df[apply_column].apply(apply_classifier)
    return df
