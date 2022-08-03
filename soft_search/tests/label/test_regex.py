#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from soft_search.constants import NSFFields, SoftwareOutcomes
from soft_search.label import regex
import pandas as pd

###############################################################################


def test_regex_label() -> None:
    # Starting DataFrame
    df = pd.DataFrame(
        {
            NSFFields.abstractText: [
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
