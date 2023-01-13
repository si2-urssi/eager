#!/usr/bin/env python

import argparse
import logging
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastcore.net import HTTP4xxClientError
from ghapi.all import GhApi

###############################################################################

load_dotenv()

###############################################################################

SEARCH_QUERIES_START_PAGE = {
    "National Science Foundation": 0,
    "NSF Award": 0,
    "NSF Grant": 0,
    "Supported by the NSF": 0,
    "Supported by NSF": 0,
}

BATCH_SIZE = 10

###############################################################################


class Args(argparse.Namespace):
    def __init__(self) -> None:
        self.__parse()

    def __parse(self) -> None:
        p = argparse.ArgumentParser(
            prog="get-github-repositories-with-nsf-ref",
            description=("Search for GitHub repositories which reference NSF Awards."),
        )
        p.add_argument(
            "-o",
            "--outdir",
            dest="outdir",
            default=Path("gh-search-results/"),
            type=Path,
            help=(
                "The path to store all paginated results. "
                "Default: gh-search-results/"
            ),
        )
        p.add_argument(
            "-c",
            "--clean",
            dest="clean",
            default=True,
            type=bool,
            help=(
                "Before running the data gathering process, "
                "should any existing outdir be cleaned of existing files. "
                "Default: True (clean existing files)"
            ),
        )
        p.add_argument(
            "-t",
            "--token",
            dest="token",
            default=None,
            type=str,
            help=(
                "GitHub Personal Access Token to use for requests. "
                "If none provided, attempts load from `.env` file. "
                "If none found, uses no-auth requests which will take longer. "
                "Default: None (use .env)"
            ),
        )
        p.add_argument(
            "--debug",
            dest="debug",
            action="store_true",
            help="Run with debug logging.",
        )
        p.parse_args(namespace=self)


###############################################################################


def main() -> None:  # noqa: C901
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
        format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    try:
        # Determine token / api
        if args.token is None:
            load_dotenv()
            api = GhApi()
        else:
            api = GhApi(token=args.token)

        # Clean
        if args.clean:
            if args.outdir.exists():
                shutil.rmtree(args.outdir)

        # Make dir if needed
        args.outdir.mkdir(parents=True, exist_ok=True)

        # Get all results for each term
        query_start_time = time.time()
        for query, page in SEARCH_QUERIES_START_PAGE.items():
            log.info(f"Beginning page requests for: '{query}'")
            complete_query = f'"{query}" filename:README.md'

            # Get initial
            all_gathered = False
            while not all_gathered:
                try:
                    log.debug(f"Querying: '{complete_query}', Page: {page}")
                    page_results = api(
                        "/search/code",
                        "GET ",
                        query={
                            "q": complete_query,
                            "per_page": BATCH_SIZE,
                            "page": page,
                        },
                    )
                    total_count = page_results["total_count"]
                    real_count = total_count if total_count < 1000 else 1000
                    items_returned = page_results["items"]

                    # Unpack results
                    results = []
                    for item in items_returned:
                        repo_details = item["repository"]
                        repo_name = repo_details["name"]
                        owner_details = repo_details["owner"]
                        owner_name = owner_details["login"]
                        full_name = f"{owner_name}/{repo_name}"

                        # Get languages
                        languages = api(f"/repos/{full_name}/languages")

                        # Get latest commit datetime
                        commits = api(f"/repos/{full_name}/commits")
                        most_recent_commit = commits[0]["commit"]
                        most_recent_committer = most_recent_commit["committer"]
                        most_recent_committer_name = most_recent_committer["name"]
                        most_recent_committer_email = most_recent_committer["email"]
                        most_recent_commit_dt = datetime.fromisoformat(
                            # We remove last character because it is 'Z' for "Zulu"
                            # Datetimes are naturally UTC/Zulu
                            most_recent_committer["date"][:-1]
                        )

                        # Append this result to all results
                        results.append(
                            {
                                "owner": owner_name,
                                "name": repo_name,
                                "link": f"https://github.com/{full_name}",
                                "languages": "; ".join(languages.keys()),
                                "most_recent_committer_name": (
                                    most_recent_committer_name
                                ),
                                "most_recent_committer_email": (
                                    most_recent_committer_email
                                ),
                                "most_recent_commit_datetime": (
                                    most_recent_commit_dt.isoformat()
                                ),
                                "most_recent_commit_timestamp": (
                                    most_recent_commit_dt.timestamp()
                                ),
                                "query": query,
                            }
                        )

                    # Store partial results
                    if len(results) != 0:
                        save_name = f"{query.lower().replace(' ', '_')}-page_{page}.csv"
                        pd.DataFrame(results).to_csv(
                            args.outdir / save_name,
                            index=False,
                        )

                    # Increase page and keep going
                    page += 1

                    # Wait to avoid rate limiting
                    log.debug("Sleeping for one minute...")
                    time.sleep(60)

                    # Update time estimate
                    batch_time = time.time()
                    seconds_diff = batch_time - query_start_time
                    seconds_diff_per_page = seconds_diff / page
                    total_pages_required = real_count / BATCH_SIZE
                    remaining_pages = total_pages_required - page
                    estimated_remaining_seconds = (
                        seconds_diff_per_page * remaining_pages
                    )
                    estimated_remained_pages = remaining_pages
                    log.info(
                        f"Remaining pages: {estimated_remained_pages} "
                        f"(of {total_pages_required} -- "
                        f"est. {timedelta(seconds=estimated_remaining_seconds)})"
                    )

                    # Break because we are done
                    # Stop at 1000 results because GitHub limits search
                    # https://github.com/PyGithub/PyGithub/issues/1072#issuecomment-499211486
                    if len(items_returned) == 0 or page * BATCH_SIZE >= 1000:
                        log.info("Reached GitHub max search results.")
                        break

                except HTTP4xxClientError as e:
                    log.error(f"Caught exception: {e}")

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
