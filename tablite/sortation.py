from datetime import datetime,date,time,timedelta
from pyuca import Collator
uca_collator = Collator()

# EXCEL
_excel_typecodes = {  # declares sortation rank: 0 < 1, etc.
    time: 0,
    int: 0,
    float: 0,
    date: 0,
    datetime: 0,
    timedelta: 0,
    str: 1,
    bool: 2,
    type(None): 3
}

_excel_date_epoc = date(1900,1,1)
_excel_datetime_epoc = datetime(1900,1,1)

# excel helpers
def _excel_none(value):
    return float('inf')

def _excel_float(value):
    return value

def _excel_int(value):
    return value

def _excel_time(value):
    return (value.hour * 60 * 60 + value.minute * 60 + value.second + (value.microsecond / 1e6)) / (24 * 60 * 60)

def _excel_date(value):
    dt = value - _excel_date_epoc
    return dt.days + (dt.seconds / (24*60*60))

def _excel_datetime(value):
    dt = value - _excel_datetime_epoc
    return dt.days + (dt.seconds / (24*60*60))

def _excel_timedelta(value):
    return value.days + (value.seconds / (24*60*60))

def _excel_bool(value):
    return int(value)

_excel_value_function = {
    time: _excel_time,
    date: _excel_date,
    datetime: _excel_datetime,
    timedelta: _excel_timedelta,
    bool: _excel_bool,
    float: _excel_float,
    int: _excel_int,
    type(None): _excel_none,
    # str is handled by pyUCA.
}

# UNIX
_unix_typecodes = {
    type(None): 0,
    bool: 1,
    int: 2,
    float: 2,
    time: 3,
    date: 4,
    datetime: 5,
    timedelta: 6,
    str: 7,  # string is handled by pyUCA.
}

_unix_date_epoc = date(1970,1,1)
_unix_datetime_epoc = datetime(1970,1,1)

def _unix_none(value):
    return -float('inf')

def _unix_float(value):
    return value

def _unix_int(value):
    return value

def _unix_time(value):
    return (value.hour * 60 * 60 + value.minute * 60 + value.second + (value.microsecond / 1e6)) / (24 * 60 * 60)

def _unix_date(value):
    dt = value - _unix_date_epoc
    return dt.days + (dt.seconds / (24*60*60))

def _unix_datetime(value):
    dt = value - _unix_datetime_epoc
    return dt.days + (dt.seconds / (24*60*60))

def _unix_timedelta(value):
    return value.days + (value.seconds / (24*60*60))

def _unix_bool(value):
    return int(value)

_unix_value_function = {
    time: _unix_time,
    date: _unix_date,
    datetime: _unix_datetime,
    timedelta: _unix_timedelta,
    bool: _unix_bool,
    float: _unix_float,
    int: _unix_int,
    type(None): _unix_none,
}

def text_sort(values, reverse):
    """
    Sorts everything as text.
    """
    text = {str(i):i for i in values}
    L = list(text.keys())
    L.sort(key=uca_collator.sort_key, reverse=reverse)
    d = {text[value]:ix for ix,value in enumerate(L)}
    return d

def unix_sort(values, reverse):
    """
    Unix sortation sorts by the following order:

    | rank | type      | value                                      |
    +------+-----------+--------------------------------------------+
    |   0  | None      | floating point -infinite                   |
    |   1  | bool      | 0 as False, 1 as True                      |
    |   2  | int       | as numeric value                           |
    |   2  | float     | as numeric value                           |
    |   3  | time      | τ * seconds into the day / (24 * 60 * 60)  |  
    |   4  | date      | as integer days since 1970/1/1             | 
    |   5  | datetime  | as float using date (int) + time (decimal) |
    |   6  | timedelta | as float using date (int) + time (decimal) |
    |   7  | str       | using unicode                              |        
    +------+-----------+--------------------------------------------+
    
    τ = 2 * π

    """
    L = []
    text = [i for i in values if isinstance(i, str)]
    text.sort(key=uca_collator.sort_key, reverse=reverse)
    text_code = _unix_typecodes[str]
    L = [(text_code,ix,v) for ix,v in enumerate(text)]

    for value in (i for i in values if not isinstance(i, str)):
        t = type(value)
        TC = _unix_typecodes[t]
        tf = _unix_value_function[t]
        VC = tf(value)
        L.append((TC,VC,value))
    L.sort(reverse=reverse)
    d = {value:ix for ix,(_,_,value) in enumerate(L)}
    return d


def excel_sort(values, reverse):
    """
    Excel sortation sorts by the following order:

    | rank | type      | value                                      |
    +------+-----------+--------------------------------------------+
    |   1  | int       | as numeric value                           |
    |   1  | float     | as numeric value                           |
    |   1  | time      | as seconds into the day / (24 * 60 * 60)   |
    |   1  | date      | as integer days since 1900/1/1             | 
    |   1  | datetime  | as float using date (int) + time (decimal) |
    |  (1)*| timedelta | as float using date (int) + time (decimal) |
    |   2  | str       | using unicode                              |
    |   3  | bool      | 0 as False, 1 as True                      |
    |   4  | None      | floating point infinite.                   |
    +------+-----------+--------------------------------------------+
    
    * Excel doesn't have timedelta.
    """
    L = []
    text = [i for i in values if isinstance(i, str)]
    
    text.sort(key=uca_collator.sort_key, reverse=reverse)
    L = [(2,ix,v) for ix,v in enumerate(text)]
    
    for value in (i for i in values if not isinstance(i,str)):
        t = type(value)
        TC = _excel_typecodes[t]
        tf = _excel_value_function[t]
        VC = tf(value)
        L.append((TC,VC,value))

    L.sort(reverse=reverse)
    d = {value:ix for ix,(_,_,value) in enumerate(L)}
    return d

modes = {
    'alphanumeric': text_sort,
    'unix': unix_sort,
    'excel': excel_sort
}


def rank(values, reverse, mode):
    """
    values: list of values to sort.
    reverse: bool
    mode: as 'text', as 'numeric' or as 'excel'
    return: dict: d[value] = rank
    """
    if mode not in modes:
        raise ValueError(f"{mode} not in list of modes: {list(modes)}")
    f = modes.get(mode)
    return f(values,reverse)
