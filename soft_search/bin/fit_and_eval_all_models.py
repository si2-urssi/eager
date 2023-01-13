#!/usr/bin/env python

import argparse
import logging
import sys
import traceback

from soft_search.label.model_selection import fit_and_eval_all_models

###############################################################################


class Args(argparse.Namespace):
    def __init__(self) -> None:
        self.__parse()

    def __parse(self) -> None:
        p = argparse.ArgumentParser(
            prog="fit-and-eval-all-soft-search-2022-models",
            description=(
                "Train and evaluate all the different models with "
                "the soft search 2022 dataset."
            ),
        )
        p.add_argument(
            "-t",
            "--test-size",
            dest="test_size",
            type=float,
            default=0.2,
            help="Test size to use for data splits.",
        )
        p.add_argument("-s", "--seed", type=int, default=0, help="Random seed.")
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
        results = fit_and_eval_all_models(args.test_size, args.seed)
        print(results)
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
