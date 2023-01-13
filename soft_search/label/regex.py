#!/usr/bin/env python

import re
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..constants import PredictionLabels
from ..data.soft_search_2022 import SoftSearch2022DatasetFields
from ..metrics import EvaluationMetrics

###############################################################################
# Constants

SOFTWARE_LIKE_PATTERNS = (
    r".*(?:software(?:\s(?:code|tool|suite|program|application|framework)s?|"
    r"(?:binar|librar)(?:y|ies))?|algorithms?|tools?).*"
)
COMPILED_SOFTWARE_LIKE_PATTERNS = re.compile(SOFTWARE_LIKE_PATTERNS)
REGEX_LABEL_COL = "regex_match"

###############################################################################


def _apply_regex(text: str) -> str:
    # Try match
    match_or_none = re.match(COMPILED_SOFTWARE_LIKE_PATTERNS, text)

    # Found
    if match_or_none:
        return PredictionLabels.SoftwarePredicted

    # Not Found
    return PredictionLabels.SoftwareNotPredicted


def train(
    df: Union[str, Path, pd.DataFrame],
    text_col: str = SoftSearch2022DatasetFields.abstract_text,
    label_col: str = SoftSearch2022DatasetFields.label,
) -> EvaluationMetrics:
    # Read DataFrame
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df)

    # Eval
    preds = df[text_col].apply(_apply_regex).to_numpy()
    pre, rec, f1, _ = precision_recall_fscore_support(
        df[label_col],
        preds,
        average="weighted",
    )
    acc = accuracy_score(df[label_col], preds)
    return EvaluationMetrics(
        model="regex",
        precision=pre,
        recall=rec,
        f1=f1,
        accuracy=acc,
    )


def label(
    df: pd.DataFrame,
    apply_column: str = "text",
    label_column: str = REGEX_LABEL_COL,
) -> pd.DataFrame:
    """
    In-place add a new column to the provided pandas DataFrame with a label
    of software predicted or not solely based off a regex match for various
    software-like and adjacent terminology.

    Parameters
    ----------
    df: pd.DataFrame
        The pandas DataFrame to in-place add a column with the
        regex matched software outcome labels.
    apply_column: str
        The column to use for "prediction".
        Default: "text"
    label_column: str
        The name of the column to add with outcome "prediction".
        Default: "regex_match"

    Returns
    -------
    pd.DataFrame
        The same pandas DataFrame but with a new column added in-place containing
        the software outcome "prediction".

    See Also
    --------
    soft_search.nsf.get_nsf_dataset
        Function to get an NSF dataset for prediction.
    """
    df[label_column] = df[apply_column].apply(_apply_regex)
    return df
