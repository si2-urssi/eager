#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import traceback
from datetime import datetime

from soft_search.label import transformer

###############################################################################

START_DATE = datetime(2017, 1, 1)
END_DATE = datetime(2022, 1, 1)


class Args(argparse.Namespace):
    def __init__(self) -> None:
        self.__parse()

    def __parse(self) -> None:
        p = argparse.ArgumentParser(
            prog="train-and-package-soft-search-2022-transformer",
            description=(
                "Train and package the 2022 Soft Search transformer model "
                "so that it can be shipped with the pip installable wheel."
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

    # Try training and storage
    try:
        transformer._train_and_store_transformer_to_package()
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
