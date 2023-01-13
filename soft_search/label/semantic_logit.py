#!/usr/bin/env python

import logging
import pickle
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..constants import DEFAULT_SEMANTIC_EMBEDDING_MODEL
from ..data.soft_search_2022 import SoftSearch2022DatasetFields
from ..metrics import EvaluationMetrics

###############################################################################

DEFAULT_SOFT_SEARCH_TFIDF_LOGIT_PATH = Path("soft-search-semantic-logit.pkl").resolve()
TFIDF_LOGIT_LABEL = "semantic_logit_label"

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def get_transformer() -> SentenceTransformer:
    return SentenceTransformer(DEFAULT_SEMANTIC_EMBEDDING_MODEL)


def train(
    train_df: Union[str, Path, pd.DataFrame],
    test_df: Union[str, Path, pd.DataFrame],
    text_col: str = SoftSearch2022DatasetFields.abstract_text,
    label_col: str = SoftSearch2022DatasetFields.label,
    model_storage_path: Union[str, Path] = DEFAULT_SOFT_SEARCH_TFIDF_LOGIT_PATH,
) -> Tuple[Path, LogisticRegressionCV, SentenceTransformer, EvaluationMetrics]:
    # Handle storage dir
    model_storage_path = Path(model_storage_path).resolve()

    # Read DataFrame
    if isinstance(train_df, (str, Path)):
        train_df = pd.read_csv(train_df)
    # Read DataFrame
    if isinstance(test_df, (str, Path)):
        test_df = pd.read_csv(test_df)

    # Get semantic transformer
    text_transformer = get_transformer()

    # Get encodings
    x_train = text_transformer.encode(train_df[text_col].to_numpy())
    x_test = text_transformer.encode(test_df[text_col].to_numpy())

    # Logit Model
    logit = LogisticRegressionCV(max_iter=10000)
    clf = logit.fit(x_train, train_df[label_col])

    # Store model
    with open(model_storage_path, "wb") as open_f:
        pickle.dump(clf, open_f)

    # Eval
    preds = logit.predict(x_test)
    pre, rec, f1, _ = precision_recall_fscore_support(
        test_df[label_col],
        preds,
        average="weighted",
    )
    acc = accuracy_score(test_df[label_col], preds)
    return (
        model_storage_path,
        clf,
        text_transformer,
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
