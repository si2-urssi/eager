"""Different software outcome labellers."""

import pickle
from typing import TYPE_CHECKING

from .tfidf_logit import ARCHIVED_SOFT_SEARCH_TFIDF_LOGIT_PATH

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


def load_soft_search_model() -> "Pipeline":
    with open(ARCHIVED_SOFT_SEARCH_TFIDF_LOGIT_PATH, "rb") as open_f:
        return pickle.load(open_f)
