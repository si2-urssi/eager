#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from soft_search.constants import SoftwareOutcomes
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
        SoftwareOutcomes.SoftwarePredicted,
        SoftwareOutcomes.SoftwareNotPredicted,
        SoftwareOutcomes.SoftwareNotPredicted,
        SoftwareOutcomes.SoftwarePredicted,
    ]

    # Run and compare
    df = regex.label(df)
    assert regex.REGEX_LABEL_COL in df.columns
    assert df[regex.REGEX_LABEL_COL].tolist() == expected_values
