#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from soft_search.data.irr import print_irr_summary_stats

###############################################################################


def test_irr_summary_stats() -> None:
    print_irr_summary_stats()
