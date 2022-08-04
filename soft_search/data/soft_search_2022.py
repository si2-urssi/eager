#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pathlib import Path
from typing import Union

import pandas as pd

from ..constants import NSFFields, SoftwareOutcomes

###############################################################################

SOFT_SEARCH_2022_DS_PATH = Path(__file__).parent / "soft-search-2022-labelled.parquet"


class SoftSearch2022DatasetFields:
    id_ = "id"
    url = "url"
    abstractText = NSFFields.abstractText
    projectOutComesReport = NSFFields.projectOutComesReport
    stated_software_will_be_created = "stated_software_will_be_created"
    stated_software_was_created = "stated_software_was_created"


ALL_SOFT_SEARCH_2022_DATASET_FIELDS = [
    getattr(SoftSearch2022DatasetFields, a)
    for a in dir(SoftSearch2022DatasetFields)
    if "__" not in a
]

###############################################################################


def _prepare_soft_search_2022(raw: Union[str, Path, pd.DataFrame]) -> Path:
    """
    Function to prepare the manually labelled data downloaded from Google Drive
    into the stored dataset.

    Parameters
    ----------
    raw: Union[str, Path, pd.DataFrame]
        The path or in-memory pandas DataFrame for the raw manually labelled data.
        Only CSV file format is supported when providing a file path.

    Returns
    -------
    Path
        The Path to the prepared and stored parquet file.
    """
    # Read data
    if isinstance(raw, (str, Path)):
        df = pd.read_csv(raw)
    else:
        df = raw

    # Select columns
    df = df[ALL_SOFT_SEARCH_2022_DATASET_FIELDS]

    # Remove any rows with "unsure" values
    df = df[df[SoftSearch2022DatasetFields.stated_software_was_created] != "unsure"]
    df = df[df[SoftSearch2022DatasetFields.stated_software_will_be_created] != "unsure"]

    # Remap values
    software_values_map = {
        "yes": SoftwareOutcomes.SoftwarePredicted,
        "no": SoftwareOutcomes.SoftwareNotPredicted,
    }
    df[SoftSearch2022DatasetFields.stated_software_was_created] = df[
        SoftSearch2022DatasetFields.stated_software_was_created
    ].map(software_values_map)
    df[SoftSearch2022DatasetFields.stated_software_will_be_created] = df[
        SoftSearch2022DatasetFields.stated_software_will_be_created
    ].map(software_values_map)

    # Store
    df.to_parquet(SOFT_SEARCH_2022_DS_PATH)
    return SOFT_SEARCH_2022_DS_PATH


def load_soft_search_2022() -> pd.DataFrame:
    """
    Load the Software Search 2022 manually labelled dataset.

    Returns
    -------
    pd.DataFrame
        The dataset.
    """
    return pd.read_parquet(SOFT_SEARCH_2022_DS_PATH)


def load_joined_soft_search_2022() -> pd.DataFrame:
    """
    Load the Software Search 2022 manually labelled dataset and then use both
    the "stated_software_was_created" and "stated_software_will_be_created" values
    and the "abstractText" and "projectOutcomesDoc" as data for training.

    Their values will be dumped to "label" and "text" columns respectively.

    Returns
    -------
    pd.DataFrame
        The joined dataset.
    """
    # Load basic
    df = load_soft_search_2022()

    # Select columns of interest
    software_pre = df[
        [
            SoftSearch2022DatasetFields.id_,
            SoftSearch2022DatasetFields.abstractText,
            SoftSearch2022DatasetFields.stated_software_will_be_created,
        ]
    ]
    software_post = df[
        [
            SoftSearch2022DatasetFields.id_,
            SoftSearch2022DatasetFields.projectOutComesReport,
            SoftSearch2022DatasetFields.stated_software_was_created,
        ]
    ]

    # Map column values
    software_pre = software_pre.rename(
        columns={
            SoftSearch2022DatasetFields.abstractText: "text",
            SoftSearch2022DatasetFields.stated_software_will_be_created: "label",
        }
    )
    software_post = software_post.rename(
        columns={
            SoftSearch2022DatasetFields.projectOutComesReport: "text",
            SoftSearch2022DatasetFields.stated_software_was_created: "label",
        }
    )

    # Concat and return
    return pd.concat([software_pre, software_post], ignore_index=True)
