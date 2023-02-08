"""Different software outcome labellers."""

import pickle
from typing import TYPE_CHECKING

from .tfidf_logit import (
    ARCHIVED_SOFT_SEARCH_ABSTRACT_SOURCE_TFIDF_LOGIT_PATH,
    ARCHIVED_SOFT_SEARCH_OUTCOMES_SOURCE_TFIDF_LOGIT_PATH,
)

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


def load_tfidf_logit_for_prediction_from_abstract() -> "Pipeline":
    with open(ARCHIVED_SOFT_SEARCH_ABSTRACT_SOURCE_TFIDF_LOGIT_PATH, "rb") as open_f:
        return pickle.load(open_f)


def load_tfidf_logit_for_prediction_from_outcomes() -> "Pipeline":
    with open(ARCHIVED_SOFT_SEARCH_OUTCOMES_SOURCE_TFIDF_LOGIT_PATH, "rb") as open_f:
        return pickle.load(open_f)
