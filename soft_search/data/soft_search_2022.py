#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pathlib import Path
from typing import List, Union

import pandas as pd

from ..constants import NSFFields, PredictionLabels

###############################################################################

SOFT_SEARCH_2022_DS_PATH = Path(__file__).parent / "soft-search-2022-labelled.parquet"
SOFT_SEARCH_2022_IRR_PATH = Path(__file__).parent / "soft-search-2022-irr.parquet"


class SoftSearch2022IRRDatasetFields:
    annotator = "annotator"
    github_link = "github_link"
    include_in_definition = "include_in_definition"
    notes = "notes"
    most_recent_commit_datetime = "most_recent_commit_datetime"


ALL_SOFT_SEARCH_2022_IRR_DATASET_FIELDS = [
    getattr(SoftSearch2022IRRDatasetFields, a)
    for a in dir(SoftSearch2022IRRDatasetFields)
    if "__" not in a
]


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


def _prepare_soft_search_2022_irr(
    all_annos: List[Union[str, Path, pd.DataFrame]],
) -> Path:
    """
    Function to prepare the manually labelled data downloaded from Google Drive
    into the stored dataset.

    Parameters
    ----------
    all_annos: Union[str, Path, pd.DataFrame]
        A list of paths or in-memory pandas DataFrames for the raw
        manually labelled data from annotator one used for calculating
        inter-rater reliability.
        Only CSV file format is supported when providing a file paths.

    Returns
    -------
    Path
        The Path to the prepared and stored parquet file.
    """
    # Fix data
    EXCLUDE_INCLUDE_VALUES_MAP = {
        "exclude": "exclude",
        "include": "include",
        "include ": "include",
        "incldue": "include",
        "exclude ": "exclude",
        "excude": "exclude",
        "include?": "include",
    }

    # Selected data
    columns_subset_frames: List[pd.DataFrame] = []
    for i, anno in enumerate(all_annos):
        # Load the data
        annotator_label: Union[str, int]
        if isinstance(anno, (str, Path)):
            anno_data = pd.read_csv(anno)
            annotator_label = Path(anno).with_suffix("").name
        else:
            anno_data = anno
            annotator_label = i

        # Drop duplicate "notes" column before rename
        anno_data = anno_data.drop(columns=["notes"])

        # Rename columns
        anno_data = anno_data.rename(
            columns={
                "include/exclude": (
                    SoftSearch2022IRRDatasetFields.include_in_definition
                ),
                "link": SoftSearch2022IRRDatasetFields.github_link,
                "Notes (justifications) ": SoftSearch2022IRRDatasetFields.notes,
                "most_recent_commit_datetime": (
                    SoftSearch2022IRRDatasetFields.most_recent_commit_datetime
                ),
            }
        )

        # Subset columns
        subset = anno_data[
            [
                col
                for col in ALL_SOFT_SEARCH_2022_IRR_DATASET_FIELDS
                if col is not SoftSearch2022IRRDatasetFields.annotator
            ]
        ]

        # Sort by link to have semi-consistent order
        subset = subset.sort_values(
            by=[
                SoftSearch2022IRRDatasetFields.github_link,
            ],
        )

        # Rename values
        subset[SoftSearch2022IRRDatasetFields.include_in_definition] = subset[
            SoftSearch2022IRRDatasetFields.include_in_definition
        ].map(EXCLUDE_INCLUDE_VALUES_MAP)

        # Add column for annotator
        subset[SoftSearch2022IRRDatasetFields.annotator] = annotator_label
        columns_subset_frames.append(subset)

    combined = pd.concat(columns_subset_frames).reset_index(drop=True)
    combined.to_parquet(SOFT_SEARCH_2022_IRR_PATH)
    return SOFT_SEARCH_2022_IRR_PATH


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
        "yes": PredictionLabels.SoftwarePredicted,
        "no": PredictionLabels.SoftwareNotPredicted,
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


def load_soft_search_2022_irr() -> pd.DataFrame:
    """
    Load the Software Search 2022 Inter-Rater Reliability labelled dataset.

    Returns
    -------
    pd.DataFrame
        The dataset.
    """
    return pd.read_parquet(SOFT_SEARCH_2022_IRR_PATH)


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
