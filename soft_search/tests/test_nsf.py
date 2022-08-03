#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from typing import List, Union

import pytest

from soft_search.constants import NSFFields
from soft_search.nsf import get_nsf_dataset

###############################################################################


@pytest.mark.parametrize(
    "start_date, end_date, dataset_fields, expected_length",
    [
        ("2017-01-01", "2017-06-01", [NSFFields.id_, NSFFields.abstractText], 27),
        ("01/01/2018", "01/01/2019", [NSFFields.id_, NSFFields.title], 38),
        (
            datetime(2019, 1, 1),
            datetime(2020, 1, 1),
            [NSFFields.id_, NSFFields.projectOutComesReport],
            28,
        ),
        # Fails because weird date format
        pytest.param(
            "2017 01 01",
            "2019 01 01",
            [NSFFields.id_, NSFFields.abstractText],
            10,
            marks=pytest.mark.raises(exception=ValueError),
        ),
    ],
)
def test_get_nsf_dataset(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    dataset_fields: List[str],
    expected_length: int,
) -> None:
    df = get_nsf_dataset(
        start_date=start_date,
        end_date=end_date,
        dataset_fields=dataset_fields,
    )
    assert all([col in dataset_fields for col in df.columns])
    assert len(df) == expected_length
