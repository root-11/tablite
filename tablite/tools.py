# tools.py is a collection of helpers inspired by Python collections.

from tablite.datatypes import DataTypes
from tablite.utils import date_range  # helper for date_ranges
from tablite.utils import unique_name
from tablite.utils import intercept as range_intercept
from tablite.sortation import text_sort, unix_sort, excel_sort
from tablite.file_reader_utils import detect_seperator, get_encoding, get_headers, get_delimiter

assert callable(date_range)

assert callable(unique_name)
assert callable(range_intercept)
assert callable(text_sort)
assert callable(unix_sort)
assert callable(excel_sort)

assert callable(detect_seperator)
assert callable(get_encoding)
assert callable(get_headers)
assert callable(get_delimiter)

guess = DataTypes.guess
xround = DataTypes.round

assert callable(guess)
assert callable(xround)
