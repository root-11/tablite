import numpy as np
from tablite.base import Table
from tablite.config import Config
from tablite.utils import unique_name, type_check


def where(T, criteria, left, right, new):
    """ takes from LEFT where criteria is True else RIGHT.
    :param: T: Table
    :param: criteria: np.array(bool): 
            if True take left column
            else take right column
    :param left: (str) column name
    :param right: (str) column name
    :param new: (str) new name

    :returns: T
    """
    type_check(T, Table)
    if isinstance(criteria, np.ndarray):
        if not criteria.dtype == "bool":
            raise TypeError
    else:
        criteria = np.array(criteria, dtype='bool')
    
    new_name = unique_name(new, list(T.columns))
    T.add_column(new_name)
    col = T[new_name]
    
    for start,end in Config.page_steps(len(criteria)):
        indices = np.arange(start,end)
        left_values = T[left].get_by_indices(indices)
        right_values = T[right].get_by_indices(indices)
        new_values = np.where(criteria, left_values, right_values)
        col.extend(new_values)

    del T[left]
    del T[right]
    if new != new_name and new not in T.columns:
        T[new] = T[new_name]
        del T[new_name]
    return T

