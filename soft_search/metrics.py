#!/usr/bin/env python

from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin

###############################################################################


@dataclass
class EvaluationMetrics(DataClassJsonMixin):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
