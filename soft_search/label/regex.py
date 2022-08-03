#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pandas as pd
import re

from ..constants import NSFFields

###############################################################################

log = logging.getLogger(__name__)

###############################################################################
# Constants

SOFTWARE_LIKE_PATTERNS = (
    r".*(?:software(?:\s(?:code|tool|suite|program|application|framework)s?|"
    r"(?:binar|librar)(?:y|ies))?|algorithms?|tools?).*"
)
COMPILED_SOFTWARE_LIKE_PATTERNS = re.compile(SOFTWARE_LIKE_PATTERNS)

###############################################################################

def _apply_regex(text: str) -> str:
    # Try match
    match_or_none = re.match(COMPILED_SOFTWARE_LIKE_PATTERNS, text)

    # Found
    if match_or_none:
        return "software-outcome-predicted"
    
    # Not Found
    return "software-outcome-not-predicted"

def label(df: pd.DataFrame, apply_column: str = NSFFields.abstractText, label_column: str = "regex_match",) -> pd.DataFrame:
    df[label_column] = df[apply_column].apply(_apply_regex)
    return df