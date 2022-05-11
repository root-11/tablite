import numpy as np
import h5py
from pathlib import Path

def test_getitem():

    L = [1,2,3,4]
    assert L[:] == [1,2,3,4]  # slice(None,None,None)
    assert L[:0] == []  # slice(None,0,None)
    assert L[0:] == [1,2,3,4]  # slice(0,None,None)
    
    assert L[:2] == [1,2]  # slice(None,2,None)

    assert L[-1:] == [4]  # slice(-1,None,None)
    assert L[-1::1] == [4]
    assert L[-1:None:1] == [4]

    assert L[-1:4:1] == [4]  # slice(-1,4,1)
    assert L[-1:0:-1] == [4,3,2]  # slice(-1,0,-1) -->  slice(4,0,-1) --> for i in range(4,0,-1)
    assert L[-1:0:1] == []  # slice(-1,0,1) --> slice(4,0,-1) --> for i in range(4,0,1)
    assert L[-3:-1:1] == [2,3]  # slice(-3,-1,-1) --> slice(1,3,1)  --> for i in range(1,3,1)

    assert L[0:0] == []  # for i in range(0,0) ...
    assert L[2:2] == []  # for i in range(2,2)
    
    assert L[:10] == [1,2,3,4]  # slice(None,10,None) --> slice(0,4,1) --> for i in range(0,4,1):...
    try:
        assert L[10] == []
    except IndexError:
        assert True
    try:
        assert L[-10] == []
    except IndexError:
        assert True
    assert L[1:2] == [2]  # slice(1,2,1) --> for i in range(1,2):...
    assert L[1:3] == [2,3]
    assert L[0:4:2] == [1,3]
    assert L[0:5:2] == [1,3]
    assert L[::2] == [1,3]
    assert L[1::2] == [2,4]
    assert L[4:0:-1] == [4,3,2] # start included, stop excluded.
    assert L[3:2:-1] == [4]
    
def test_setitem():

    # VALUES! NOT SLICES!
    # ---------------
    L = [1,2,3]
    L[0] = [4,5]   # VALUE: REPLACE position in L with NEW
    assert L == [[4,5],2,3]

    # L = [1,2,3]
    # L[3] = 4
    # IndexError: list assignment index out of range

    L = [1,2,3]
    L[-3] = 4
    assert L ==[4,2,3]

    # L = [1,2,3]
    # L[-4] = 4
    # IndexError: list assignment index out of range
    
    # SLICES - ONE VALUE!
    # -------------------

    L = [1,2,3]
    L[:0] = [4,5]  # SLICE: "REPLACE" L before 0 with NEW
    assert L == [4,5,1,2,3]

    L = [1,2,3]
    L[0:] = [4,5]  # SLICE: REPLACE L after 0 with NEW
    assert L == [4,5]

    L = [1,2,3]
    L[:1] = [4,5]  # SLICE: REPLACE L before 1 with NEW
    assert L == [4,5,2,3]

    L = [1,2,3]
    L[:2] = [4,5]   # SLICE: REPLACE L before 2 with NEW
    assert L == [4,5,3]

    L = [1,2,3]
    L[:3] = [4,5]  # SLICE: REPLACE L before 3 with NEW
    assert L == [4,5]

    # SLICES - TWO VALUES!
    # --------------------
    L = [1,2,3]
    L[0:1] = [4,5]  # SLICE: DROP L between A,B (L[0]=[1]). INSERT NEW starting on 0.
    assert L == [4,5,2,3]

    L = [1,2,3]
    L[1:0] = [4,5]  # SLICE: DROP L between A,B (nothing). INSERT NEW starting on 1.
    assert L == [1,4,5,2,3]

    L = [10,20,30]
    L[1:3] = [4]  # SLICE: DROP L bewteen A,B (L[1:3] = [20,30]). INSERT NEW starting on 1.
    assert L == [10,4]

    L = [1,2,3]
    L[0:3] = [4]
    assert L == [4]  # SLICE: DROP L between A,B (L[0:3] = [1,2,3]). INSERT NEW starting on 0

    # SLICES - THREE VALUES!
    # ----------------------

    L = [1,2,3]
    L[0::2] = [4,5]  # SLICE: for new_index,position in enumerate(range(0,end,step=2)): REPLACE L[position] WITH NEW[ew_index]
    assert L == [4,2,5]

    L = [1,1,1,1,1,1] 
    L[0::2] = [2,3,4]  # SLICE: for new_index,position in enumerate(range(0,end,step=2)): REPLACE L[position] WITH NEW[ew_index]
    assert L == [2, 1, 3, 1, 4, 1]

    L = [1,1,1,1,1,1] 
    L[1::2] = [2,3,4]  # SLICE: for new_index,position in enumerate(range(0,end,step=2)): REPLACE L[position] WITH NEW[ew_index]
    assert L == [1, 2, 1, 3, 1, 4]

    # L = [1,1,1,1,1,1] 
    # L[2::2] = [2,3,4]  
    # ValueError: attempt to assign sequence of size 3 to extended slice of size 2

    # L = [1,1,1,1,1,1] 
    # L[1::-2] = [2,3,4] 
    # ValueError: attempt to assign sequence of size 3 to extended slice of size 1

    # L = [1,1,1,1,1,1] 
    # L[:1:-2] = [2,3,4]  
    # ValueError: attempt to assign sequence of size 3 to extended slice of size 2

    L = [1,1,1,1,1,1] 
    L[None::-2] = [2,3,4]  # SLICE: for new_index, position in enumerate(reversed(range(start,end,-2)): REPLACE L[position] WITH NEW[new_index]
    assert L == [1, 4, 1, 3, 1, 2]  #                                                       ! ----^

    # Note that L[None::-2] becomes range(*slice(None,None,-2).indices(len(L))) == range(5,-1,-2)  !!!!
    L = [1,1,1,1,1,1]
    new = [2,3,4]
    for new_ix,pos in enumerate(range(*slice(None,None,-2).indices(len(L)))):
        L[pos] = new[new_ix] 
    assert L == [1, 4, 1, 3, 1, 2]

    # What happens if we leave out the first : ?
    L = [1,1,1,1,1,1] 
    L[:-2] = [2,3,4]  # SLICE: REPLACE L before -2 with NEW
    assert L == [2,3,4,1,1] 

    # THIS MEANS THAT None is an active OPERATOR that has different meaning depending on the args position.
    L = [1,1,1,1,1,1] 
    L[None:None:-2] = [2,3,4]
    assert L == [1, 4, 1, 3, 1, 2]

    # THAT SETITEM AND GETITEM BEHAVE DIFFERENT DEPENDING ON THE NUMBER OF ARGUMENTS CAN SEEM VERY ARCHAIC !

    L = [1,2,3]
    # L[None] = []  # TypeError: list indices must be integers or slices, not NoneType
    assert L[None : None] == [1,2,3]
    assert L[None : None:   1  ] == [1,2,3]
    assert L[None : None : None] == [1,2,3]
    assert L[  1  : None : None] == [2,3]
    assert L[None :   1  : None] == [1]
    assert L[None : None :  2  ] == [1,3]

    L = [1,2,3]; L[None : None] = [4,5]  # --> L == [4,5]
    L = [1,2,3]; L[None : None:   1  ] = [4,5]  # --> L == [4,5]
    L = [1,2,3]; L[None : None : None] = [4,5]  # --> L == [4,5]
    L = [1,2,3]; L[  1  : None : None] = [4,5]  # --> L == [1,4,5]
    L = [1,2,3]; L[None :   1  : None] = [4,5]  # --> L == [4,5,2,3] 
    L = [1,2,3]; L[None : None :  2  ] = [4,5]  # --> L == [4,2,5]

def test_for_numpy():
    L = [1,2,3]

    L2 = np.array(L)
    L2[:0] == []
    L2[0:] == [1,2,3]
    L2[0::2] == [1,3]

    # Create h5py dataset.
    p = Path('this.h5')
    if p.exists():
        p.unlink()
    with h5py.File(p, 'a') as h5:
        dset = h5.create_dataset(name='L', data=L)
        L3 = dset[:]
        assert (L2 == L3).all()  # numpy for list(L) == list(L2)

    # ONE TO ONE REPLACEMENT.
    L[:2] = [4,5]
    L2[:2] = [4,5]  # numpy
    assert L == L2.tolist()

    with h5py.File(p, 'a') as h5:
        dset = h5['/L']
        L3 = dset[:]
        L3[:2] = [4,5]  # h5py
        assert (L2 == L3).all()  # numpy for list(L) == list(L2)

    # ONE TO TWO REPLACMEENT
    L[:2] = [6]  # --> L == [6,3]
    L2[:2] = [6]  # --> numpy L2 == [6,6,3]
    with h5py.File(p, 'a') as h5:
        dset = h5['/L']
        L3 = dset[:]

        L3[:2] = [6]   # h5py L3 == [6,6,3]    
        assert (L2 == L3).all()  # numpy for list(L) == list(L2)
    assert L != L2.tolist()  # [6,3] != [6,6,3]

    p.unlink()


class Mock(object):
    def __init__(self, *args) -> None:
        self.items = args
        self.last = None
    def __getitem__(self, i):
        self.last = i
    def __setitem__(self, key,value):
        self.last = (key,value)
    # def __getslice__(self,i,j):  
    #     self.last = i,j
    # def __setslice(self,i,j,sequence):  # ONLY APPLY ON SUBCLASSES FOR CPYTHON OBJECTS.
    #     self.last = i,j,sequence
    # def __delslice(self,i,j):
    #     self.last = i,j
    def __delitem__(self, index):
        self.last = index


def test_subclass():
    m = Mock()
    m[:3] # __getitem__
    assert m.last == slice(3)
    m[2:3] 
    assert m.last == slice(2,3)
    m[2:5] = [1,2,3]
    assert m.last == (slice(2, 5), [1,2,3])
    del m[3:5]
    assert m.last == slice(3,5)
