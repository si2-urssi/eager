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
GH_REPOS_WITH_NSF_REF_2022_PATH = (
    Path(__file__).parent / "gh-search-results-duplicates-removed.csv"
)
GH_REPOS_LINKED_TO_NSF_IDS_PATH = (
    Path(__file__).parent / "linked-github-nsf-results.parquet"
)
LINDSEY_GH_REPOS_ANNOTATION_PATH = (
    Path(__file__).parent / "gh-repo-annotations-lindsey.csv"
)
RICHARD_GH_REPOS_ANNOTATION_PATH = (
    Path(__file__).parent / "gh-repo-annotations-richard.csv"
)


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
    project_outcomes = "project_outcomes"
    label = "label"
    from_template_repo = "from_template_repo"
    is_a_fork = "is_a_fork"


ALL_SOFT_SEARCH_2022_DATASET_FIELDS = [
    getattr(SoftSearch2022DatasetFields, a)
    for a in dir(SoftSearch2022DatasetFields)
    if "__" not in a
]

###############################################################################


def load_github_repos_with_nsf_refs_2022() -> pd.DataFrame:
    """
    Load the GitHub repositories with references to NSF dataset.

    Created via the `get-github-repositories-with-nsf-ref` bin script.

    Returns
    -------
    pd.DataFrame
        The dataset.
    """
    return pd.read_csv(GH_REPOS_WITH_NSF_REF_2022_PATH)


def _prepare_soft_search_2022_irr(
    all_annos: List[Union[str, Path, pd.DataFrame]],
) -> Path:
    """
    Prepare and store sample annotation data for use in future IRR calculation.

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


def load_linked_github_repositories_with_nsf_awards_2022() -> pd.DataFrame:
    """
    Load the GitHub repositories linked to specific NSF award IDs dataset.

    Created via the `find-nsf-award-ids-in-github-readmes-and-link` bin script.

    Returns
    -------
    pd.DataFrame
        The dataset.
    """
    return pd.read_parquet(GH_REPOS_LINKED_TO_NSF_IDS_PATH)


def _prepare_soft_search_2022(
    linked_nsf_github_repos: Union[str, Path, pd.DataFrame] = (
        GH_REPOS_LINKED_TO_NSF_IDS_PATH
    ),
    lindsey_data: Union[str, Path, pd.DataFrame] = LINDSEY_GH_REPOS_ANNOTATION_PATH,
    richard_data: Union[str, Path, pd.DataFrame] = RICHARD_GH_REPOS_ANNOTATION_PATH,
) -> Path:
    """
    Prepare the soft search dataset for storage in the package.

    Merge various dataframes together. Fetch NSF fields for each NSF Award ID. Drop
    duplicates. Store to parquet in the project data archive.

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

    # Get both the abstract and the project outcomes report
    get_nsf_fields = ",".join(
        [
            NSFFields.abstractText,
            NSFFields.projectOutComesReport,
        ]
    )

    def _thread_text_prediction_cols(
        award_id: int,
    ) -> Optional[Dict[str, Union[int, str]]]:
        response_data = requests.get(
            f"https://api.nsf.gov/"
            f"services/v1/awards/{award_id}.json"
            f"?printFields={get_nsf_fields}"
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
            "project_outcomes": single_award.get(NSFFields.projectOutComesReport, None),
        }

    # Thread gather texts
    abstract_texts_list = thread_map(
        _thread_text_prediction_cols,
        data.nsf_award_id.unique(),
        desc="Getting NSF Award Abstracts",
    )

    # Filter failed values
    extra_items = pd.DataFrame([at for at in abstract_texts_list if at is not None])

    # Join to original data frame
    data = data.join(extra_items.set_index("award_id"), on="nsf_award_id")

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

    # We want to drop duplicates of nsf award id
    # There should only be 1 example of an NSF award ID
    # no need to duplicate the examples
    # NOTE: before we drop duplicates we sort by label descending so that
    # "if an nsf award id has a label of `software-predicted`" it retains that label
    # i.e. prior to this line, an award may have multiple examples in the dataset
    # some of those examples produce software and some do not produce software
    # if ANY of those examples produce software, we want to label the award as producing
    # software
    data = data.sort_values(by=["label"], ascending=False)
    data = data.drop_duplicates(subset=["nsf_award_id"])

    # Store to standard location
    data.to_parquet(SOFT_SEARCH_2022_DS_PATH)

    return SOFT_SEARCH_2022_DS_PATH


def load_soft_search_2022_training() -> pd.DataFrame:
    """
    Load the Software Search 2022 manually labelled dataset.

    Returns
    -------
    pd.DataFrame
        The dataset.
    """
    return pd.read_parquet(SOFT_SEARCH_2022_DS_PATH)


def load_soft_search_2022_training_irr() -> pd.DataFrame:
    """
    Load the Software Search 2022 Inter-Rater Reliability labelled dataset.

    Returns
    -------
    pd.DataFrame
        The dataset.
    """
    return pd.read_parquet(SOFT_SEARCH_2022_IRR_PATH)
