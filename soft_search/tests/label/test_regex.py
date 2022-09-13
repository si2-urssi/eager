#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from soft_search.constants import PredictionLabels
from soft_search.label import regex

###############################################################################


def test_regex_label() -> None:
    # Starting DataFrame
    df = pd.DataFrame(
        {
            "text": [
                "software",
                "hello",
                "world",
                "algorithm",
            ],
        },
    )
    # Expected values based off above abstractText column
    expected_values = [
        PredictionLabels.SoftwarePredicted,
        PredictionLabels.SoftwareNotPredicted,
        PredictionLabels.SoftwareNotPredicted,
        PredictionLabels.SoftwarePredicted,
    ]

    # Run and compare
    df = regex.label(df)
    assert regex.REGEX_LABEL_COL in df.columns
    assert df[regex.REGEX_LABEL_COL].tolist() == expected_values
