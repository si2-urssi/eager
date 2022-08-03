#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from soft_search import nsf
from soft_search.constants import NSFPrograms

###############################################################################

START_DATE = datetime(2017, 1, 1)
END_DATE = datetime(2022, 1, 1)


class Args(argparse.Namespace):
    def __init__(self) -> None:
        self.__parse()

    def __parse(self) -> None:
        p = argparse.ArgumentParser(
            prog="get-nsf-award-set-for-soft-search-2022",
            description=(
                "Get the NSF awards set used for manually "
                "labelling software outcomes."
            ),
        )
        p.add_argument(
            "-o",
            "--outfile",
            dest="outfile",
            default=Path("./soft-search-awards.csv"),
            type=Path,
            help="The path to store the dataset CSV.",
        )
        p.add_argument(
            "--debug",
            dest="debug",
            action="store_true",
            help="Run with debug logging.",
        )
        p.parse_args(namespace=self)


###############################################################################


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
        format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    # Get all program chunks
    # Concat
    # Store to CSV
    try:
        # Get chunks
        program_chunks: List[pd.DataFrame] = []
        for program in [NSFPrograms.BIO]:
            log.info(f"Gathering {program} dataset chunk...")
            program_chunks.append(
                nsf.get_nsf_dataset(
                    start_date=START_DATE, end_date=END_DATE, program_name=program
                )
            )

        # Concat and report size
        awards = pd.concat(program_chunks, ignore_index=True)
        log.info(f"Total awards found: {len(awards)}")

        # Store
        outfile = args.outfile.resolve()
        awards.to_csv(outfile, index=False)
        log.info(f"Awards dataset stored to: '{outfile}'.")
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
