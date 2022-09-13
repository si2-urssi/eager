#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pathlib import Path
from typing import Union

import pandas as pd

from ..constants import NSFFields, PredictionLabels

###############################################################################

SOFT_SEARCH_2022_DS_PATH = Path(__file__).parent / "soft-search-2022-labelled.parquet"
SOFT_SEARCH_2022_IRR_PATH = Path(__file__).parent / "soft-search-2022-irr-50.parquet"


class SoftSearch2022IRRDatasetFields:
    AwardNumber = "AwardNumber"
    Abstract = "Abstract"
    PromisesSoftware = "PromisesSoftware"
    PromisesModel = "PromisesModel"
    PromisesAlgorithm = "PromisesAlgorithm"
    PromisesDatabase = "PromisesDatabase"
    Notes = "Notes"


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
    anno1: Union[str, Path, pd.DataFrame],
    anno2: Union[str, Path, pd.DataFrame],
) -> Path:
    """
    Function to prepare the manually labelled data downloaded from Google Drive
    into the stored dataset.

    Parameters
    ----------
    anno1: Union[str, Path, pd.DataFrame]
        The path or in-memory pandas DataFrame for the raw manually labelled
        data from annotator one used for calculating inter-rater reliability.
        Only CSV file format is supported when providing a file path.
    anno2: Union[str, Path, pd.DataFrame]
        The path or in-memory pandas DataFrame for the raw manually labelled
        data from annotator two used for calculating inter-rater reliability.
        Only CSV file format is supported when providing a file path.

    Returns
    -------
    Path
        The Path to the prepared and stored parquet file.
    """
    # Read data
    if isinstance(anno1, (str, Path)):
        anno1_data = pd.read_csv(anno1)
    else:
        anno1_data = anno1
    if isinstance(anno2, (str, Path)):
        anno2_data = pd.read_csv(anno2)
    else:
        anno2_data = anno2

    # Select columns
    anno1_data = anno1_data[ALL_SOFT_SEARCH_2022_IRR_DATASET_FIELDS]
    anno2_data = anno2_data[ALL_SOFT_SEARCH_2022_IRR_DATASET_FIELDS]

    # Replace any nan values with "no"
    anno1_data = anno1_data.fillna("no")
    anno2_data = anno2_data.fillna("no")

    # Remap values
    for cat in ["Software", "Model", "Algorithm", "Database"]:
        column = f"Promises{cat}"
        remapper = {
            "yes": getattr(PredictionLabels, f"{cat}Predicted"),
            "no": getattr(PredictionLabels, f"{cat}NotPredicted"),
        }
        anno1_data[column] = anno1_data[column].map(remapper)
        anno2_data[column] = anno2_data[column].map(remapper)

    # Combine to single dataframe
    anno1_data["AnnotatorNum"] = 1
    anno2_data["AnnotatorNum"] = 2
    combined = pd.concat([anno1_data, anno2_data]).reset_index(drop=True)
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
