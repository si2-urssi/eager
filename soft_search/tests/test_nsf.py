#!/usr/bin/env python

from datetime import datetime
from typing import List, Union

import pytest

from soft_search.constants import NSFFields
from soft_search.nsf import get_nsf_dataset


###############################################################################


@pytest.mark.parametrize(
    "start_date, end_date, dataset_fields",
    [
        ("2017-01-01", "2017-02-01", [NSFFields.id_, NSFFields.abstractText]),
        ("01/01/2018", "02/01/2018", [NSFFields.id_, NSFFields.title]),
        (
            datetime(2019, 1, 1),
            datetime(2019, 2, 1),
            [NSFFields.id_, NSFFields.projectOutComesReport],
        ),
        # Fails because weird date format
        pytest.param(
            "2017 01 01",
            "2019 01 01",
            None,
            marks=pytest.mark.raises(exception=ValueError),
        ),
    ],
)
def test_get_nsf_dataset(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    dataset_fields: List[str],
) -> None:
    df = get_nsf_dataset(
        start_date=start_date,
        end_date=end_date,
        dataset_fields=dataset_fields,
    )
    assert all([col in dataset_fields for col in df.columns])
    # Note: we used to check expected length but the NSF API is flakey
    # API returns duplicate awards (by id) and when we drop duplicates
    # it sometimes differs based on the return order
