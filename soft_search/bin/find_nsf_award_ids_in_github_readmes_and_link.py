#!/usr/bin/env python

import argparse
import logging
import re
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError
from tqdm import tqdm

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Args(argparse.Namespace):
    def __init__(self) -> None:
        self.__parse()

    def __parse(self) -> None:
        p = argparse.ArgumentParser(
            prog="find-nsf-award-ids-in-github-readmes-and-link",
            description=(
                "Scrape each repository's README for NSF award ID, "
                "attempt to verify the ID is valid, "
                "and add to a 'verified' dataset."
            ),
        )
        p.add_argument(
            "gh_data",
            type=Path,
            help=(
                "The path to the GitHub dataset to parse and link with "
                "NSF Awards. Must be provided as a CSV."
            ),
        )
        p.add_argument(
            "-o",
            "--out",
            dest="out",
            default=Path("linked-github-nsf-results.parquet"),
            type=Path,
            help=(
                "The path to the linked GitHub and NSF results. "
                "All other error files will be stored with similar names. "
                "All errors and results will be stored as a parquet file. "
                "Default: linked-github-nsf-results.parquet"
            ),
        )
        p.add_argument(
            "-s",
            "--sleep",
            dest="sleep",
            default=5,
            type=int,
            help=("Number of seconds to sleep between requests to GitHub. Default: 5"),
        )
        p.add_argument(
            "--debug",
            dest="debug",
            action="store_true",
            help="Run with debug logging.",
        )
        p.parse_args(namespace=self)


###############################################################################


def _parse_repos(  # noqa: C901
    gh_data_path: Path,
    out_path: Path,
    sleep_time: int,
) -> None:
    # Load GitHub Data
    github_data = pd.read_csv(gh_data_path)

    # Dedupe based on GitHub Link
    github_data = github_data.drop_duplicates(subset=["link"])

    # Create containers for errored or found data
    linked_rows = []
    failed_rows = []

    # Iter each row in dataset
    # Parse README
    # Check for 7-digit-strings
    # Check those 7-digit-strings with NSF API
    # Sort errors or results into dataframes
    for _, row in tqdm(
        github_data.iterrows(),
        desc="Parsing READMEs and checking found digit-strings",
        total=len(github_data),
    ):
        # Request GitHub page
        try:
            github_page = requests.get(row.link)
            github_page.raise_for_status()
        except HTTPError:
            failed_rows.append(
                {
                    "github_link": row.link,
                    "nsf_award_id": None,
                    "reason": "failed-to-scrape-github",
                }
            )
            continue

        # Find and scrape README text
        soup = BeautifulSoup(github_page.content, "html.parser")
        readme_container = soup.find(id="readme")
        if readme_container is None:
            failed_rows.append(
                {
                    "github_link": row.link,
                    "nsf_award_id": None,
                    "reason": "no-readme",
                }
            )
            continue

        # Check if this repo is a fork or from a template
        possible_generated_from_template_texts = soup.find_all(
            "span",
            class_="text-small lh-condensed-ultra no-wrap mt-1 color-fg-muted",
        )
        mark_template_repo = False
        for possible_gen_from_template in possible_generated_from_template_texts:
            if possible_gen_from_template.text.startswith("generated from"):
                mark_template_repo = True

        possible_forked_texts = soup.find_all(
            "span",
            class_="text-small lh-condensed-ultra no-wrap mt-1",
        )
        mark_forked_repo = False
        for possible_from_fork in possible_forked_texts:
            if possible_from_fork.text.startswith("forked from"):
                mark_forked_repo = True

        # Find all 7 digit numbers
        # This is obviously very inclusive
        # We _hope_ to filter some out in the NSF checking in a second
        likely_nsf_award_references = re.findall(
            r"([0-9]{7})",
            readme_container.text,
            re.MULTILINE,
        )

        # strict_nsf_award_references =
        #
        # I couldn't figure out this regex
        # (NSF|National\ Science\ Foundation){1}(.)+([0-9]{7})+
        #
        # For things like
        # National Science Foundation awards 1043681, 1559691, and 1542736;
        #
        # it only finds the last one
        # I have tried many variants of it...

        # For each award, try to see if it is a valid id
        # by checking with the API
        # This does not guarantee that it is the _correct_ award
        #
        # i.e. if we find a 7 digit number that is a valid award ID but
        # isn't the correct/matching award
        for nsf_award_id in set(likely_nsf_award_references):
            try:
                award_response = requests.get(
                    f"https://api.nsf.gov/services/v1/awards/{nsf_award_id}.json"
                )
                award_response.raise_for_status()

                # Safety check further
                data = award_response.json()
                if "response" in data:
                    response_data = data["response"]
                    if "award" in response_data:
                        award_data = response_data["award"]

                        # No award found with this id
                        if len(award_data) == 0:
                            failed_rows.append(
                                {
                                    "github_link": row.link,
                                    "nsf_award_id": nsf_award_id,
                                    "reason": "no-nsf-award-with-matching-id",
                                }
                            )
                            continue

                    else:
                        failed_rows.append(
                            {
                                "github_link": row.link,
                                "nsf_award_id": nsf_award_id,
                                "reason": "nsf-award-id-request-error",
                            }
                        )
                        continue
                else:
                    failed_rows.append(
                        {
                            "github_link": row.link,
                            "nsf_award_id": nsf_award_id,
                            "reason": "nsf-award-id-request-error",
                        }
                    )
                    continue

                # Seems like a match, add it
                linked_rows.append(
                    {
                        "github_link": row.link,
                        "nsf_award_id": nsf_award_id,
                        "nsf_link": (
                            f"https://www.nsf.gov/awardsearch/"
                            f"showAward?AWD_ID={nsf_award_id}"
                        ),
                        "from_template_repo": mark_template_repo,
                        "is_a_fork": mark_forked_repo,
                    }
                )
            except HTTPError:
                failed_rows.append(
                    {
                        "github_link": row.link,
                        "nsf_award_id": nsf_award_id,
                        "reason": "nsf-award-id-request-error",
                    }
                )

        # Sleep for a bit to not get throttled by GitHub
        time.sleep(sleep_time)

    # Merge all data and store to outputs
    linked_data = pd.DataFrame(linked_rows)
    linked_data.to_parquet(out_path)
    failed_data = pd.DataFrame(failed_rows)
    failed_data.to_parquet(out_path.with_suffix(".errors.parquet"))


def main() -> None:
    # Get args
    args = Args()

    # Determine log level
    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Setup logging
    logging.basicConfig(
        level=log_level,
        format=("[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s"),
    )

    # Run func
    try:
        _parse_repos(args.gh_data, args.out, args.sleep)
    except Exception as e:
        log.error("=============================================")
        log.error("\n\n" + traceback.format_exc())
        log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
