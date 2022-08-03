#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from typing import List, Optional, Union

import pandas as pd

try:
    import requests
except ImportError:
    raise ImportError(
        "Extra dependencies are needed for the `nsf` submodule of `soft-search`. "
        "Install with `pip install soft-search[nsf]`."
    )

from .constants import ALL_NSF_FIELDS, NSFPrograms

###############################################################################
# Constants

_NSF_API_URL_TEMPLATE = (
    "https://api.nsf.gov/services/v1/awards.json?"
    "fundProgramName={program_name}"
    "&agency={agency}"
    "&dateStart={start_date}"
    "&dateEnd={end_date}"
    "&transType={transaction_type}"
    "&printFields={dataset_fields}"
    "&projectOutcomesOnly={require_project_outcomes}"
    "&offset={offset}"
)

###############################################################################


def _parse_nsf_datetime(dt: Union[str, datetime]) -> str:
    if isinstance(dt, str):
        # Assume "/" means MM/DD/YYYY format
        if "/" in dt:
            return dt

        # Assume "-" means isoformat
        if "-" in dt:
            dt = datetime.fromisoformat(dt)
        # Anything else, raise
        else:
            raise ValueError(
                f"Provided value to `start_date` parameter must be provided as "
                f"either MM/DD/YYYY or YYYY-MM-DD format. Received: '{dt}'"
            )

    # Should either be already formated (from "/")
    # or we had isoformat conversion or provided datetime
    return dt.strftime("%m/%d/%Y")


def _get_nsf_chunk(
    start_date: str,
    end_date: str,
    program_name: str,
    agency: str,
    transaction_type: str,
    dataset_fields: str,
    require_project_outcomes: str,
    offset: int,
) -> pd.DataFrame:
    # Make the request
    response = requests.get(
        _NSF_API_URL_TEMPLATE.format(
            start_date=start_date,
            end_date=end_date,
            program_name=program_name,
            agency=agency,
            transaction_type=transaction_type,
            dataset_fields=dataset_fields,
            require_project_outcomes=require_project_outcomes,
            offset=offset,
        )
    )

    # Parse and return
    return pd.DataFrame(response.json()["response"]["award"])


def get_nsf_dataset(
    start_date: Union[str, datetime],
    end_date: Optional[Union[str, datetime]] = None,
    program_name: str = NSFPrograms.BIO,
    agency: str = "NSF",
    transaction_type: str = "Grant",
    dataset_fields: List[str] = ALL_NSF_FIELDS,
    require_project_outcomes_doc: bool = True,
) -> pd.DataFrame:
    """
    Fetch an NSF awards dataset.
    Wraps the NSF Award Search API:
    https://www.research.gov/common/webapi/awardapisearch-v1.htm

    Parameters
    ----------
    start_date: Union[str, datetime]
        The datetime for which awards were granted after.
        When provided as a string, "MM/DD/YYYY" and "YYYY-MM-DD" formats are accepted.
    end_date: Optional[Union[str, datetime]]
        The datetime for which awards were granted before.
        When provided as a string, "MM/DD/YYYY" and "YYYY-MM-DD" formats are accepted.
        Default: None (no end date)
    program_name: str
        The program to search for awards against.
        Default: "BIO"
    agency: str
        The funding agency.
        Default: "NSF"
    transaction_type: str
        The award type.
        Default: "Grant"
    dataset_fields: List[str]
        The fields to retrieve.
        Default: All fields available in the `soft_search.constants.NSFFields` object.
    require_project_outcomes_doc: bool
        Should only awards that have already returned project outcomes documents
        be requested.
        Default: True (request only projects with outcomes)

    Returns
    -------
    pd.DataFrame
        All awards found as a pandas DataFrame.

    Examples
    --------
    Get all grants funded by the NSF that have project outcomes under the BIO program
    from 2017 onward.

    >>> from soft_search.nsf import get_nsf_dataset
    >>> get_nsf_dataset(start_date="2017-01-01")

    Get all grants funded by the NSF that have project outcomes under the BIO program
    from 2017 onward but only return the id and abstractText fields.

    >>> from soft_search.nsf import get_nsf_dataset
    >>> from soft_search.constants import NSFFields
    >>> get_nsf_dataset(
    ...     start_date="2017-01-01",
    ...     dataset_fields=[
    ...         NSFFields.id_,
    ...         NSFFields.abstractText,
    ...     ]
    ... )

    See Also
    --------
    soft_search.constants.NSFFields
        Available dataset fields to request.
    soft_search.constants.NSFPrograms
        Available programs to request.
    """
    # Parse datetimes
    formatted_start_date = _parse_nsf_datetime(start_date)
    if end_date is None:
        end_date = datetime.utcnow()
    formatted_end_date = _parse_nsf_datetime(end_date)

    # Convert dataset fields to str
    str_dataset_fields = ",".join(dataset_fields)

    # Convert required project outcomes bool to str
    str_require_project_outcomes = str(require_project_outcomes_doc).lower()

    # Run gather
    current_offset = 0
    chunks: List[pd.DataFrame] = []
    while True:
        # Get chunk
        chunk = _get_nsf_chunk(
            start_date=formatted_start_date,
            end_date=formatted_end_date,
            program_name=program_name,
            agency=agency,
            transaction_type=transaction_type,
            dataset_fields=str_dataset_fields,
            require_project_outcomes=str_require_project_outcomes,
            offset=current_offset,
        )
        chunks.append(chunk)

        # Check chunk length
        # The default request size for NSF is 25
        # If we received less than 25 results,
        # we can assume we are done.
        if len(chunk) < 25:
            break

        # Update state
        current_offset += 25

    # Concat all awards
    awards = pd.concat(chunks, ignore_index=True)
    return awards
