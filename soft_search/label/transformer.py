#!/usr/bin/env python

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    pipeline,
)

from ..constants import DEFAULT_SEMANTIC_EMBEDDING_MODEL
from ..data.soft_search_2022 import SoftSearch2022DatasetFields
from ..metrics import EvaluationMetrics

if TYPE_CHECKING:
    from datasets.arrow_dataset import Batch
    from transformers.pipelines.base import Pipeline
    from transformers.tokenization_utils_base import BatchEncoding
    from transformers.trainer_utils import TrainOutput

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

DEFAULT_SOFT_SEARCH_TRANSFORMER_PATH = Path("soft-search-transformer/").resolve()
TRANSFORMER_LABEL_COL = "transformer_label"
HUGGINGFACE_HUB_SOFT_SEARCH_MODEL = "evamaxfield/soft-search"

###############################################################################


def train(
    train_df: Union[str, Path, pd.DataFrame],
    test_df: Union[str, Path, pd.DataFrame],
    text_col: str = SoftSearch2022DatasetFields.abstract_text,
    label_col: str = SoftSearch2022DatasetFields.label,
    model_storage_path: Union[str, Path] = DEFAULT_SOFT_SEARCH_TRANSFORMER_PATH,
    base_model: str = DEFAULT_SEMANTIC_EMBEDDING_MODEL,
    extra_training_args: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Trainer, "TrainOutput", EvaluationMetrics]:
    """
    Fine-tune a transformer model to classify the provided labels.

    This function will both train and evaluate the performance of the
    fine-tuned transformer.

    Parameters
    ----------
    train_df: Union[str, Path, pd.DataFrame]
        The data to use for training.
        Only CSV file format is supported when providing a file path.
    test_df: Union[str, Path, pd.DataFrame]
        The data to use for training.
        Only CSV file format is supported when providing a file path.
    text_col: str
        The column name which contains the raw text.
        Default: "abstract_text"
    label_col: str
        The column name which contains the labels.
        Default: "label"
    model_storage_path: Union[str, Path]
        The path to store the model to.
        Default: "soft-search-transformer/"
    base_model: str
        The base model to fine-tune.
        Default: "distilbert-base-uncased-finetuned-sst-2-english"
    extra_training_args: Optional[Dict[str, Any]]
        Any extra arguments to pass to the Trainer object.

    Returns
    -------
    Path
        The path to the stored model.
    Trainer
        The Trainer object.
    TrainOutput
        The final output of the trainer.train() call.
    EvaluationMetrics
        The evaluation metrics.

    Examples
    --------
    Example training from supplied manually labelled data.

    >>> from soft_search.data import load_joined_soft_search_2022
    >>> from soft_search.label import transformer
    >>> from sklearn.model_selection import train_test_split
    >>> df = load_joined_soft_search_2022()
    >>> train, test = train_test_split(
    ...     df,
    ...     test_size=0.2,
    ...     stratify=df["label"]
    ... )
    >>> model = transformer.train(train)

    See Also
    --------
    label
        A function to apply a model across a pandas DataFrame.
    """
    # Handle storage dir
    model_storage_path = Path(model_storage_path).resolve()

    # Read DataFrame
    if isinstance(train_df, (str, Path)):
        train_df = pd.read_csv(train_df)
    # Read DataFrame
    if isinstance(test_df, (str, Path)):
        test_df = pd.read_csv(test_df)

    # Rename cols
    train_df = train_df.copy(deep=True)
    train_df = train_df[[label_col, text_col]]
    train_df = train_df.rename(columns={label_col: "label", text_col: "text"})
    test_df = test_df.copy(deep=True)
    test_df = test_df[[label_col, text_col]]
    test_df = test_df.rename(columns={label_col: "label", text_col: "text"})

    # Train and test should have the same label names
    # only grab from train
    label_names = train_df["label"].unique().tolist()

    # Construct label to id and vice-versa LUTs
    label2id, id2label = {}, {}
    for i, label in enumerate(label_names):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Cast to dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    train_dataset = train_dataset.class_encode_column("label")
    test_dataset = test_dataset.class_encode_column("label")

    # Preprocess
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def preprocess_function(examples: "BatchEncoding") -> "Batch":
        return tokenizer(examples["text"], truncation=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

    # AutoModel
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Catch None extra args
    if extra_training_args is None:
        extra_training_args = {}

    # Training Args
    training_args = TrainingArguments(
        output_dir=model_storage_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        num_train_epochs=5,
        weight_decay=0.01,
        **extra_training_args,
    )

    # Compute accuracy metrics
    acc_metric = load_metric("accuracy")
    pre_metric = load_metric("precision")
    rec_metric = load_metric("recall")
    f1_metric = load_metric("f1")

    def compute_metrics(eval_pred: EvalPrediction) -> Optional[Dict]:
        predictions = np.argmax(eval_pred.predictions, axis=-1)
        f1_score = f1_metric.compute(
            predictions=predictions,
            references=eval_pred.label_ids,
        )
        acc_score = acc_metric.compute(
            predictions=predictions,
            references=eval_pred.label_ids,
        )
        pre_score = pre_metric.compute(
            predictions=predictions,
            references=eval_pred.label_ids,
        )
        rec_score = rec_metric.compute(
            predictions=predictions,
            references=eval_pred.label_ids,
        )
        return {
            **f1_score,
            **acc_score,
            **pre_score,
            **rec_score,
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    epoch_metrics = trainer.train()
    raw_eval_metrics = trainer.evaluate(tokenized_test_dataset)
    eval_metrics = EvaluationMetrics(
        model="transformer",
        accuracy=raw_eval_metrics["eval_accuracy"],
        precision=raw_eval_metrics["eval_precision"],
        recall=raw_eval_metrics["eval_recall"],
        f1=raw_eval_metrics["eval_f1"],
    )

    # Store model
    trainer.save_model()
    return model_storage_path, trainer, epoch_metrics, eval_metrics


def _train_and_upload_transformer(seed: int = 0) -> Path:
    import os

    from ..data import load_soft_search_2022_training
    from ..seed import set_seed

    # Set global seed
    set_seed(seed)

    # Load data, train
    df = load_soft_search_2022_training()
    train_df, test_df = train_test_split(df, test_size=0.2)
    model, _, _, _ = train(
        train_df,
        test_df,
        extra_training_args={
            "push_to_hub": True,
            "hub_model_id": HUGGINGFACE_HUB_SOFT_SEARCH_MODEL,
            "hub_strategy": "end",
            "hub_token": os.environ["HUGGINGFACE_TOKEN"],
        },
    )

    return model


def _apply_transformer(text: str, classifier: "Pipeline") -> str:
    return classifier(text, truncation=True, top_k=1)[0]["label"]


def label(
    df: pd.DataFrame,
    apply_column: str = SoftSearch2022DatasetFields.abstract_text,
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
