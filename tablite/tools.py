# tools.py is a collection of helpers inspired by Python collections.

from tablite.datatypes import DataTypes
from tablite.utils import date_range  # helper for date_ranges
from tablite.utils import xround

assert callable(date_range)
assert callable(xround)


def guess(values):
    """ 
    guesses datatypes 

    example:
    >>> guess(['1','2','3'])
    [1,2,3]
    >>> tools.guess(['1.0','2.0','3.0'])
    [1.0, 2.0, 3.0]
    >>> guess(['true', 'false'])
    [True,False]
    
    """
    return DataTypes.guess(values)


