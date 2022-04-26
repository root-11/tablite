from tablite.memory_manager import MemoryManager, SimpleType, MixedType
import numpy as np

def test01():
    A = np.array( [1,2,3])
    assert A.dtype.char == 'l'  # l as in long, not 1 as in one.
    B = np.array( ['a', 'b', 'c'])
    assert B.dtype.char == 'U'
    C = np.concatenate((A,B),dtype='O')  # without dtype declared this is type 'U' (str)
    assert C.dtype.char == 'O'
    D = np.array([123456789.98765432, 0.1])  # without dtype declared this is type 'U' (str)
    assert D.dtype.char == 'd'  # d as in decimal.
    E = np.concatenate((A,B,D), dtype='O')
    assert E.dtype.char == 'O'
    F = np.array( [ np.nan, None])
    G = np.concatenate((A,B,D,F))
    assert G.dtype.char == 'O'

