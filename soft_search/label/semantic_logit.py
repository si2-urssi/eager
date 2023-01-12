#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Tuple, Union
import pickle

import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from ..data.soft_search_2022 import SoftSearch2022DatasetFields
from ..metrics import EvaluationMetrics
from ..constants import DEFAULT_SEMANTIC_EMBEDDING_MODEL

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
    # Get semantic transformer
    text_transformer = get_transformer()

    # Get encodings
    X_train = text_transformer.encode(train_df[text_col].to_numpy())
    X_test = text_transformer.encode(test_df[text_col].to_numpy())

    # Logit Model
    logit = LogisticRegressionCV(max_iter=10000)
    clf = logit.fit(X_train, train_df[label_col])

    # Store model
    with open(model_storage_path, "wb") as open_f:
        pickle.dump(clf, open_f)

    # Eval
    preds = logit.predict(X_test)
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
            precision=pre,
            recall=rec,
            f1=f1,
            accuracy=acc,
        ),
    )


def label():
    pass