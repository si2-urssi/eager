#!/usr/bin/env python

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from soft_search import nsf
from soft_search.constants import ALL_NSF_PROGRAMS

###############################################################################


class Args(argparse.Namespace):
    def __init__(self) -> None:
        self.__parse()

    def __parse(self) -> None:
        p = argparse.ArgumentParser(
            prog="get-nsf-award-set-for-soft-search-2022",
            description=(
                "Get the NSF awards dataset for use in "
                "downstream prediction of software."
            ),
        )
        p.add_argument(
            "-s",
            "--start-date",
            dest="start_date",
            default="2016-01-01",
            type=str,
            help="ISO format string with the date to start gathering awards for.",
        )
        p.add_argument(
            "-e",
            "--end-date",
            dest="end_date",
            default="2023-01-01",
            type=str,
            help="ISO format string with the date to end gathering awards for.",
        )
        p.add_argument(
            "-o",
            "--outfile",
            dest="outfile",
            default=Path("./soft-search-awards.csv"),
            type=Path,
            help="The path to store the dataset CSV (and Parquet with the same name).",
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
        for program in tqdm(ALL_NSF_PROGRAMS, desc="Iterating major programs..."):
            log.info(f"Gathering {program} dataset chunk...")
            chunk = nsf.get_nsf_dataset(
                start_date=args.start_date,
                end_date=args.end_date,
                program_name=program,
                require_project_outcomes_doc=False,
            )
            chunk["majorProgram"] = program
            program_chunks.append(chunk)

        # Concat and report size
        awards = (
            pd.concat(program_chunks, ignore_index=True)
            .drop_duplicates("id")
            .reset_index(drop=True)
        )
        log.info(f"Total awards found: {len(awards)}")

        # Store
        outfile = args.outfile.resolve()
        awards.to_csv(outfile, index=False)
        awards.to_parquet(outfile.with_suffix(".parquet"))
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
