# tools.py is a collection of helpers inspired by Python collections.

from datatypes import DataTypes
from utils import date_range  # helper for date_ranges
from utils import xround

assert callable(date_range)
assert callable(xround)


def guess(values):
    """ guesses datatypes"""
    return DataTypes.guess(values)


