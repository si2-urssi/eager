#!/usr/bin/env python


from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from tqdm.contrib.concurrent import thread_map

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
    github_link = "github_link"
    nsf_award_id = "nsf_award_id"
    nsf_award_link = "nsf_award_link"
    abstract_text = "abstract_text"
    label = "label"
    from_template_repo = "from_template_repo"
    is_a_fork = "is_a_fork"


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
    exclude_include_values_map = {
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
        ].map(exclude_include_values_map)

        # Add column for annotator
        subset[SoftSearch2022IRRDatasetFields.annotator] = annotator_label
        columns_subset_frames.append(subset)

    combined = pd.concat(columns_subset_frames).reset_index(drop=True)
    combined.to_parquet(SOFT_SEARCH_2022_IRR_PATH)
    return SOFT_SEARCH_2022_IRR_PATH


def _prepare_soft_search_2022(
    linked_nsf_github_repos: Union[str, Path, pd.DataFrame],
    lindsey_data: Union[str, Path, pd.DataFrame],
    richard_data: Union[str, Path, pd.DataFrame],
) -> Path:
    """
    Function to prepare the manually labelled data and store in repo.

    Parameters
    ----------
    linked_nsf_github_repos: Union[str, Path, pd.DataFrame]
        The path or in-memory pandas DataFrame for the linked GitHub repositories
        to the NSF Awards produced by the
        `find-nsf-award-ids-in-github-readmes-and-link` script.
        Only Parquet file format is supported when providing a file path.
    lindsey_data: Union[str, Path, pd.DataFrame]
        The path or in-memory pandas DataFrame for the raw manually labelled
        data from Lindsey. Only CSV file format is supported when providing a file path.
    richard_data: Union[str, Path, pd.DataFrame]
        The path or in-memory pandas DataFrame for the raw manually labelled
        data from Richard. Only CSV file format is supported when providing a file path.

    Returns
    -------
    Path
        The Path to the prepared and stored parquet file.
    """
    # Read data
    if isinstance(linked_nsf_github_repos, (str, Path)):
        linked_nsf_github_df = pd.read_parquet(linked_nsf_github_repos)
    else:
        linked_nsf_github_df = linked_nsf_github_repos
    if isinstance(lindsey_data, (str, Path)):
        lindsey_df = pd.read_csv(lindsey_data)
    else:
        lindsey_df = lindsey_data
    if isinstance(richard_data, (str, Path)):
        richard_df = pd.read_csv(richard_data)
    else:
        richard_df = richard_data

    # Clean Lindsey
    lindsey_df = lindsey_df[["include/exclude", "link"]]
    lindsey_df = lindsey_df[~lindsey_df["include/exclude"].isna()]

    # Clean Richard
    richard_df = richard_df[["include/exclude", "link"]]
    richard_df = richard_df[~richard_df["include/exclude"].isna()]

    # Join and clean after merge
    data_lindsey = lindsey_df.join(
        linked_nsf_github_df.set_index("github_link"),
        on="link",
    )
    data_richard = richard_df.join(
        linked_nsf_github_df.set_index("github_link"),
        on="link",
    )
    data = pd.concat([data_lindsey, data_richard])
    data = data.drop_duplicates(subset=["link", "nsf_award_id"])
    data = data.dropna(subset=["nsf_award_id"])

    def _thread_abstract_text(award_id: int) -> Optional[Dict[str, Union[int, str]]]:
        response_data = requests.get(
            f"https://api.nsf.gov/"
            f"services/v1/awards/{award_id}.json"
            f"?printFields={NSFFields.abstractText}"
        ).json()

        # Handle data existance
        if "response" not in response_data:
            return None
        response_subset = response_data["response"]

        if "award" not in response_subset:
            return None
        award_data = response_subset["award"]

        if len(award_data) == 0:
            return None
        single_award = award_data[0]

        # Return the award id and the abstract text
        return {
            "award_id": award_id,
            "abstract_text": single_award[NSFFields.abstractText],
        }

    # Thread gather texts
    abstract_texts_list = thread_map(
        _thread_abstract_text,
        data.nsf_award_id.unique(),
        desc="Getting NSF Award Abstracts",
    )

    # Filter failed values
    abstract_texts = pd.DataFrame([at for at in abstract_texts_list if at is not None])

    # Join to original data frame
    data = data.join(abstract_texts.set_index("award_id"), on="nsf_award_id")

    # Drop any rows that are missing abstract text
    data = data.dropna(subset=["abstract_text"])

    # Rename to standard set
    data = data.rename(
        columns={
            "include/exclude": SoftSearch2022DatasetFields.label,
            "link": SoftSearch2022DatasetFields.github_link,
            "nsf_link": SoftSearch2022DatasetFields.nsf_award_link,
        },
    )

    # Replace include and exclude with int
    data[SoftSearch2022DatasetFields.label] = data[
        SoftSearch2022DatasetFields.label
    ].replace(
        {
            "exclude": PredictionLabels.SoftwareNotPredicted,
            "include": PredictionLabels.SoftwarePredicted,
        }
    )

    # Store to standard location
    data.to_parquet(SOFT_SEARCH_2022_DS_PATH)

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
