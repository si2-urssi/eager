#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

import pandas as pd

from ..constants import SoftwareOutcomes

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
        return SoftwareOutcomes.SoftwarePredicted

    # Not Found
    return SoftwareOutcomes.SoftwareNotPredicted


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
