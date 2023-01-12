#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pickle
import re
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..data.soft_search_2022 import SoftSearch2022DatasetFields
from ..metrics import EvaluationMetrics

###############################################################################

DEFAULT_SOFT_SEARCH_TFIDF_LOGIT_PATH = Path("soft-search-tfidf-logit.pkl").resolve()
TFIDF_LOGIT_LABEL = "tfidf_logit_label"

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def setup_tfidf_vectorizer() -> TfidfVectorizer:
    # Stem and clean
    stemmer = PorterStemmer()
    pattern = re.compile(r"(?u)\b\w\w+\b")

    def stem(text: str):
        tokens = pattern.findall(text)
        stems = [stemmer.stem(item) for item in tokens]
        return stems

    # Vectorize
    text_transformer = TfidfVectorizer(
        tokenizer=stem,
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
    )
    return text_transformer


def train(
    train_df: Union[str, Path, pd.DataFrame],
    test_df: Union[str, Path, pd.DataFrame],
    text_col: str = SoftSearch2022DatasetFields.abstract_text,
    label_col: str = SoftSearch2022DatasetFields.label,
    model_storage_path: Union[str, Path] = DEFAULT_SOFT_SEARCH_TFIDF_LOGIT_PATH,
) -> Tuple[Path, LogisticRegressionCV, TfidfVectorizer, EvaluationMetrics]:
    # Get vectorizer
    text_transformer = setup_tfidf_vectorizer()

    # Get encodings
    X_train = text_transformer.fit_transform(train_df[text_col])
    X_test = text_transformer.transform(test_df[text_col])

    # Logit Model
    logit = LogisticRegressionCV(max_iter=10000)
    logit.fit(X_train, train_df[label_col])

    # Store model
    with open(model_storage_path, "wb") as open_f:
        pickle.dump(logit, open_f)

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
        logit,
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
