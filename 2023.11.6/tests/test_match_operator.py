import numpy as np
from tablite import Table
from tablite.match import match, find_indices, merge_indices


def test_match():
    
    # boms has no repetitions but is volatile.
    normalized_bom = Table({'bom_id': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'partial_of': [1, 2, 3, 4, 5, 6, 7, 4, 6], 'sku': ['A', 'irrelevant', 'empty carton', 'pkd carton', 'empty pallet', 'pkd pallet', 'pkd irrelevant', 'ppkd carton', 'ppkd pallet'], 'material_id': [None, None, None, 3, None, 5, 3, 3, 5], 'quantity': [1, 1, 1, 1, 1, 1, 1, 1, 1], 'L': [0.25, 0.1, 0.6, 0.6, 1.2, 1.2, 0.6, 0.6, 1.2], 'W': [0.15, 0.1, 0.4, 0.4, 1.0, 1.0, 0.4, 0.4, 1.0], 'H': [0.15, 0.1, 0.3, 0.3, 0.15, 0.15, 0.3, 0.3, 0.15], 'M': [0.13, 1.0, 0.3, 0.3, 12.5, 12.5, 0.3, 0.3, 12.5], 'L2': [None, None, 0.6, 0.6, 1.2, 1.2, 0.6, 0.6, 1.2], 'W2': [None, None, 0.4, 0.4, 1.0, 1.0, 0.4, 0.4, 1.0], 'H2': [None, None, 0.3, 0.3, 1.8, 1.8, 0.3, 0.3, 1.8], 'M2': [None, None, 40.0, 40.0, 1200.0, 1200.0, 40.0, 40.0, 1200.0], 'packtype': ['item', 'item', 'material', 'carton', 'material', 'pallet', 'carton', 'carton', 'pallet'], 'properties': [None, None, None, None, None, None, None, None, None]})
    # 9 is a partially packed pallet of 6

    # multiple values.
    accepts = Table({'bom_id': [3,4,6], 'sku': ['pkd pallet']*3, 'supply_time': [np.timedelta64(3600000000,'us')]*3, 'moq': [1,2,3]})

    products_lookup = normalized_bom.lookup(accepts, ("bom_id", "==", "bom_id"), ("partial_of", "==", "bom_id"), all=False)
    
    # drop items that didn't match.
    products = products_lookup.all(bom_id_1=lambda x: x is not None)
    # +==+======+==========+===========+===========+========+===+===+====+====+===+===+===+======+========+==========+========+==========+===========+===+
    # |# |bom_id|partial_of|    sku    |material_id|quantity| L | W | H  | M  | L2| W2| H2|  M2  |packtype|properties|bom_id_1|  sku_1   |supply_time|moq|
    # +--+------+----------+-----------+-----------+--------+---+---+----+----+---+---+---+------+--------+----------+--------+----------+-----------+---+
    # | 0|     6|         6|pkd pallet |          5|       1|1.2|1.0|0.15|12.5|1.2|1.0|1.8|1200.0|pallet  |None      |       6|pkd pallet|    1:00:00|  1|
    # | 1|     9|         6|ppkd pallet|          5|       1|1.2|1.0|0.15|12.5|1.2|1.0|1.8|1200.0|pallet  |None      |       6|pkd pallet|    1:00:00|  1|
    # +==+======+==========+===========+===========+========+===+===+====+====+===+===+===+======+========+==========+========+==========+===========+===+

    products_matched = match(normalized_bom, accepts, ("bom_id", "==", "bom_id"), ("partial_of", "==", "bom_id"))

    assert products.to_dict() == products_matched.to_dict()


def test_find_indices():  
    """
    finds index of y in x
    """
    import numpy as np
    x = np.array([3, 5, 7,  1,   9, 8, 6, 6])
    y = np.array([2, 1, 5, 10, 100, 6])
    result = find_indices(x,y)
    assert np.all(result == np.array([-1,  3,  1, -1, -1,  6]))
    
def test_merge_indices():
    """
    merges x1 and x2 where 
    """
    A = np.array([-1,  3, -1, 5, -1])
    B = np.array([-1, -1,  4, 5, -1])
    C = np.array([-1, -1, -1, -1, 6])

    AB = merge_indices(A,B)
    assert np.all(AB == np.array([-1,3,4,5,-1]))
    ABC = merge_indices(AB,C)
    assert np.all(ABC == np.array([-1,3,4,5,6]))


