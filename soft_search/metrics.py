#!/usr/bin/env python

from dataclasses import dataclass

###############################################################################


@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float