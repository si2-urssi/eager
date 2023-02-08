#!/usr/bin/env python

import logging
import pickle
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from embetter.text import SentenceEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, make_pipeline

from ..constants import DEFAULT_SEMANTIC_EMBEDDING_MODEL
from ..data import _DATA_DIR
from ..data.soft_search_2022 import SoftSearch2022DatasetFields
from ..metrics import EvaluationMetrics

###############################################################################

DEFAULT_SOFT_SEARCH_SEMANTIC_LOGIT_PATH = Path(
    "soft-search-semantic-logit.pkl"
).resolve()
ARCHIVED_SOFT_SEARCH_SEMANTIC_LOGIT_PATH = (
    _DATA_DIR / DEFAULT_SOFT_SEARCH_SEMANTIC_LOGIT_PATH.name
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def train(
    train_df: Union[str, Path, pd.DataFrame],
    test_df: Union[str, Path, pd.DataFrame],
    text_col: str = SoftSearch2022DatasetFields.abstract_text,
    label_col: str = SoftSearch2022DatasetFields.label,
    model_storage_path: Union[str, Path] = DEFAULT_SOFT_SEARCH_SEMANTIC_LOGIT_PATH,
) -> Tuple[Path, Pipeline, EvaluationMetrics]:
    # Handle storage dir
    model_storage_path = Path(model_storage_path).resolve()

    # Read DataFrame
    if isinstance(train_df, (str, Path)):
        train_df = pd.read_csv(train_df)
    # Read DataFrame
    if isinstance(test_df, (str, Path)):
        test_df = pd.read_csv(test_df)

    # Build the pipeline
    pipeline = make_pipeline(
        SentenceEncoder(DEFAULT_SEMANTIC_EMBEDDING_MODEL),
        LogisticRegressionCV(max_iter=10000),
    )

    # Fit the pipeline
    pipeline.fit(train_df[text_col], train_df[label_col])

    # Save the pipeline
    with open(model_storage_path, "wb") as open_f:
        pickle.dump(pipeline, open_f)

    # Eval
    preds = pipeline.predict(test_df[text_col])
    pre, rec, f1, _ = precision_recall_fscore_support(
        test_df[label_col],
        preds,
        average="weighted",
    )
    acc = accuracy_score(test_df[label_col], preds)
    return (
        model_storage_path,
        pipeline,
        EvaluationMetrics(
            model="semantic-logit",
            precision=pre,
            recall=rec,
            f1=f1,
            accuracy=acc,
        ),
    )


def label() -> None:
    pass
