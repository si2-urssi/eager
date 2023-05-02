#!/usr/bin/env python

# Import necessary packages and modules
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

# Load environment variables
load_dotenv()

# Set starting page for each search query
SEARCH_QUERIES_START_PAGE = {
    "National Science Foundation": 0,
    "NSF Award": 0,
    "NSF Grant": 0,
    "Supported by the NSF": 0,
    "Supported by NSF": 0,
}

# Set batch size for API requests
BATCH_SIZE = 10

# Define custom argparse namespace to hold program arguments
class Args(argparse.Namespace):
    def __init__(self) -> None:
        self.__parse()

    def __parse(self) -> None:
        # Define command-line argument parser
        p = argparse.ArgumentParser(
            prog="get-github-repositories-with-nsf-ref",
            description=("Search for GitHub repositories which reference NSF Awards."),
        )
        # Define command-line arguments
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
        # Parse command-line arguments and store them in the namespace
        p.parse_args(namespace=self)

# Define main function
def main() -> None:  # noqa: C901
    # Get program arguments from Args namespace
    args = Args()

    # Determine log level based on program arguments
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
        # Determine whether to use authentication token for API requests
        if args.token is None:
            load_dotenv()
            api = GhApi()
        else:
            api = GhApi(token=args.token)

        # Clean output directory before running data gathering process
        if args.clean:
            if args.outdir.exists():
                shutil.rmtree(args.outdir)

        # Create output directory if it doesn't already exist
        args.outdir.mkdir(parents=True, exist_ok=True)

        # Search GitHub for repositories which reference NSF awards
        query_start_time = time.time()
        for
