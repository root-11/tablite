from tablite import Table
import numpy as np
from datetime import datetime, timedelta
import pytest

# DESCRIPTION
# The basic tests seeks to cover list like functionality:
# If a table is created, saved, copied, updated or deleted.
# If a column is created, copied, updated or deleted.

# The tests must assure that all common pytypes can be handled.
# This means converting to and from HDF5-bytes format, when HDF5 
# can't handle the datatype natively.


@pytest.fixture(autouse=True) # this resets the HDF5 file for every test.
def refresh():
    Table.reset_storage()
    yield


def test01_compatible_datatypes():
    now = datetime.now().replace(microsecond=0)

    table4 = Table()
    table4['A'] = [-1, 1]
    table4['B'] = [None, 1]     
    table4['C'] = [-1.1, 1.1]
    table4['D'] = ["", "1000"]     
    table4['E'] = [None, "1"]   
    table4['F'] = [False, True]
    table4['G'] = [now, now]
    table4['H'] = [now.date(), now.date()]
    table4['I'] = [now.time(), now.time()]
    table4['J'] = [timedelta(1), timedelta(2, 400)]
    table4['K'] = ['b',"嗨"]  # utf-32
    table4['L'] = [-10**23,10**23]  # int > int64. 
    table4['M'] = [float('inf'), float('-inf')]
    assert table4.columns == ['A','B','C','D','E','F','G','H','I','J','K']  # testing .columns property.

    table4.save = True  # testing that save keeps the data in HDF5.
    del table4  
    
    # recover all active tables from HDF5.
    tables = Table.reload_saved_tables()
    table5 = tables[0]  # this is the content of table4
    assert table5['A'] == [-1,1]  # list test
    assert table5['A'] == np.array([-1,1])  # numpy test
    assert table5['A'] == (-1,1)  # tuple test
    assert table5['A'] == [-1, 1]
    assert table5['B'] == [None, 1]     
    assert table5['C'] == [-1.1, 1.1]
    assert table5['D'] == ["", "1000"]     
    assert table5['E'] == [None, "1"]   
    assert table5['F'] == [False, True]
    assert table5['G'] == [now, now]
    assert table5['H'] == [now.date(), now.date()]
    assert table5['I'] == [now.time(), now.time()]
    assert table5['J'] == [timedelta(1), timedelta(2, 400)]
    assert table5['K'] == ['b',"嗨"]
    assert table5['L'] == [-10**23,10**23]  # int > int64. 
    assert table5['M'] == [float('inf'), float('-inf')]
    rows = [row for row in table5.rows]  # test .rows iterator.
    assert len(rows) == 2
    assert rows[0][0] == -1
    assert rows[0][1] == 1

def test01_confirm_storage_reset():
    tables = Table.reload_saved_tables()
    assert tables == []


def test01_test_numpy_64datetime_and_timedelta():
    raise NotImplementedError
    a = np.datetime64(b'2014-12-12')   # Y-M-D
    b = np.datetime64(b'2014-12-12T12:23:34')   # Y-M-DTh:m:s
    c = np.datetime64(b'2014-12-12T12:23:34.123456')   # Y-M-DTh:m:s.u
    d = np.datetime64('2014-12-12T12:23:34.123456')   # Y-M-DTh:m:s.u
    e = np.timedelta64(dts, 'ms')  # h:m:s
    
    # https://stackoverflow.com/questions/52982056/how-to-convert-numpy-datetime64-ns-to-python-datetime



def test02_verify_garbage_collection():
    # check that the pages are not deleted prematurely.
    table4 = Table()
    table4['A'] = [-1, 1]
    table5 = Table()
    table5['A'] = table4['A']  
    
    del table4['A']
    assert table4.columns == []

    assert table5['A'] == [-1, 1]

    del table4
    del table5
    import gc; gc.collect()  # pytest keeps reference to table4 & 5, so gc must be invoked explicitly.
    # alternatively the explicit call to .__del__ could be made.
    # table4.__del__()  
    # table5.__del__()

    tables = Table.reload_saved_tables()
    assert tables == []


def test03_verify_list_like_operations():
    table4 = Table()
    table4['A'] = [0,1,2,3]  # create page1
    assert list(table4['A'][:]) == [0,1,2,3]
    table4['A'] += [4,5,6]   # append to page1 as ref count == 1.
    assert list(table4['A'][:]) == [0,1,2,3,4,5,6]
    table4['A'][0] = 7  # update as ref count == 1

    table4['A'] += table4['A']  # duplication of pages.
    table4['A'] += [8,9,10]  # append where ref count > 1 creates a new page.
    L = list(table4['A'][:])
    assert L == [7,1,2,3,4,5,6, 7,1,2,3,4,5,6, 8,9,10]
    table4['A'][0] = 11  # unlink page 0, create new page and update record [0] with 10
    assert table4['A'][0] == 11

    table5 = table4.copy()
    table5 += table4
    assert len(table5) == 2 * len(table4)

    table5.clear()
    assert table5.columns == []

def test03_verify_single_page_updates():  
    table4 = Table()
    table4['A'] = list(range(10))
    L = list(range(10))
    C = table4['A']
    L[:0] = [-3,-2,-1]  # insert
    C[:0] = [-3,-2,-1]  # insert
    assert list(C[:]) == L
    L[len(L):] = [10,11,12]  # extend
    C[len(C):] = [10,11,12]  # extend
    assert list(C[:]) == L  # array([-3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
    L[0:2] = [20]  # reduce  # array([20, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
    C[0:2] = [20]  # reduce
    assert list(C[:]) == L  
    L[0:1] = [-3,-2]  # expand
    C[0:1] = [-3,-2]  # expand
    assert list(C[:]) == L
    L[4:8] = [11,12,13,14]  # replace
    C[4:8] = [11,12,13,14]  # replace
    assert list(C[:]) == L
    L[8:4:-1] = [21,22,23,24]  # replace reverse
    C[8:4:-1] = [21,22,23,24]  # replace reverse
    assert list(C[:]) == L


def test03_verify_multi_page_updates():  
    table4 = Table()
    table4['A'] = list(range(5))
    table4['A'] += table4['A']
    L = list(range(10))
    L[:0] = [-3,-2,-1]  # insert
    L[len(L):] = [10,11,12]  # extend
    L[0:2] = [20]  # reduce
    L[0:1] = [-3,-2]  # expand
    L[4:8] = [11,12,13,14]  # replace


def test03_verify_slice_operator_for_uniform_datatype():    
    table4 = Table()
    table4['A'] = L = [0, 10, 20, 3, 4, 5, 100]  # create
    assert L == [0, 10, 20, 3, 4, 5, 100]
    assert table4['A'] == L

    table4['A'][3], L[3] = 30, 30  # update
    assert L == [0, 10, 20, 30, 4, 5, 100]
    assert table4['A'] == L
    
    # operations below are handled by __getitem__ with a slice as a key.
    table4['A'][4:5], L[4:5] = [40,50], [40,50]  # update many as len(4:5)== 1 but len(values)==2
    assert L == [0, 10, 20, 30, 40, 50, 5, 100]
    assert table4['A'] == L
    
    table4['A'][-2:-1], L[-2:-1] = [60,70,80,90],[60,70,80,90]  # update 1, insert 3
    assert L == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert table4['A'] == L
    
    table4['A'][2:4], L[2:4] = [2], [2]  # update + delete 1
    assert L == [0, 10, 2, 40, 50, 60, 70, 80, 90, 100]
    assert table4['A'] == L

    x = len(table4['A'])
    table4['A'][x:], L[len(L):] = [110], [110]  # append
    assert L == [0, 10, 2, 40, 50, 60, 70, 80, 90, 100, 110]
    assert table4['A'] == L
    
    del L[:3]
    del table4['A'][:3]
    assert L == [40, 50, 60, 70, 80, 90, 100, 110]
    assert table4['A'] == L

    L[::3] = [0,0,0]
    table4['A'][::3] = [0,0,0]
    assert L == [0, 50, 60, 0, 80, 90, 0, 110]
    assert table4['A'] == L

    L[4:6] = []
    table4['A'][4:6] = []
    assert L == [0, 50, 60, 0, 0, 110]
    assert table4['A'] == L

    L[None:0] = [20, 30]
    table4['A'][None:0] = [20, 30]
    assert L == [20, 30, 0, 50, 60, 0, 0, 110]
    assert table4['A'] == L

    L[:0] = [10]
    table4['A'][:0] = [10]
    assert L == [10, 20, 30, 0, 50, 60, 0, 0, 110]
    assert table4['A'] == L

    col = table4['A']
    try:
        col[3::3] = [5] * 9
        assert False, f"attempt to assign sequence of size 9 to extended slice of size 3"
    except ValueError:
        assert True

    col.insert(0, -10)
    col.append(120)
    assert list(col) == [-10, 10, 20, 30, 0, 50, 60, 0, 0, 110, 120]
    col.extend([130,140])
    assert list(col) == [-10, 10, 20, 30, 0, 50, 60, 0, 0, 110, 120, 130, 140]
    col.extend(col)
    assert list(col) == 2 * [-10, 10, 20, 30, 0, 50, 60, 0, 0, 110, 120, 130, 140]


def test03_verify_slice_operator_for_multitype_datasets():
    t = Table()
    t['A'] = ["1",2,3.0]
    
    # VALUES! NOT SLICES!
    # ---------------
    L = t['A'].copy()
    assert L == ["1", 2, 3.0]
    try:
        L[0] = [4,5]   # VALUE: REPLACE position in L with NEW
        assert False, "allowing this would insert [4,5] in L as [[4,5],2,3.0] which would break the column format."
    except TypeError:
        assert True
 
    L = t['A'].copy()
    L[-3] = 4
    assert L ==[4,2,3.0]
    
    # SLICES - ONE VALUE!
    # -------------------

    L = t['A'].copy()
    L[:0] = [4,"5"]  # SLICE: "REPLACE" L before 0 with NEW
    assert L == [4,"5","1",2,3.0]

    L = t['A'].copy()
    L[0:] = [4,5]  # SLICE: REPLACE L after 0 with NEW
    assert L == [4,5]  # <-- check that page type becomes simple!

    L = t['A'].copy()
    L[:1] = [4,5]  # SLICE: REPLACE L before 1 with NEW
    assert L == [4,5,2,3.0]

    L = t['A'].copy()
    L[:2] = [4,5]   # SLICE: REPLACE L before 2 with NEW
    assert L == [4,5,3.0]

    L = t['A'].copy()
    L[:3] = [4,5]  # SLICE: REPLACE L before 3 with NEW
    assert L == [4,5] # <-- check that page type becomes simple!

    # SLICES - TWO VALUES!
    # --------------------
    L = t['A'].copy()
    L[0:1] = [4,5]  # SLICE: DROP L between A,B (L[0]=[1]). INSERT NEW starting on 0.
    assert L == [4,5,2,3]

    L = t['A'].copy()
    L[1:0] = [4,"5"]  # SLICE: DROP L between A,B (nothing). INSERT NEW starting on 1.
    assert L == ["1",4,"5",2,3.0]

    L = t['A'].copy()
    L[0:2] = [4,"5",6.0]
    assert L == [4,"5",6.0,3.0]

    L = t['A'].copy()
    L[1:3] = ["4"]  # SLICE: DROP L bewteen A,B (L[1:3] = ["4"]). INSERT NEW starting on 1.
    assert L == ["1","4"]

    L = t['A'].copy()
    L[0:3] = [4]
    assert L == [4]  # SLICE: DROP L between A,B (L[0:3] = [1,2,3]). INSERT NEW starting on 0

    # SLICES - THREE VALUES!
    # ----------------------

    L = t['A'].copy()
    L[0::2] = [4,5]  # SLICE: for new_index,position in enumerate(range(0,end,step=2)): REPLACE L[position] WITH NEW[ew_index]
    assert L == [4,2,5]

    t = Table()
    t['A'] = ["1", 1, 1.0, "1", 1, 1.0]

    L = t['A'].copy()
    L[0::2] = [2,3,4]  # SLICE: for new_index,position in enumerate(range(0,end,step=2)): REPLACE L[position] WITH NEW[ew_index]
    assert L == [2, 1, 3, "1", 4, 1.0]

    L = t['A'].copy()
    L[1::2] = [2,3,4]  # SLICE: for new_index,position in enumerate(range(0,end,step=2)): REPLACE L[position] WITH NEW[ew_index]
    assert L == ["1", 2, 1.0, 3, 1, 4]

    L = t['A'].copy()
    try:
        L[2::2] = [2,3,4]  
    except ValueError as e:
        assert str(e) == "attempt to assign sequence of size 3 to extended slice of size 2"

    L = t['A'].copy()
    try:
        L[1::-2] = [2,3,4] 
    except ValueError as e:
        assert str(e) == "attempt to assign sequence of size 3 to extended slice of size 1"

    L = t['A'].copy()
    try:
        L[:1:-2] = [2,3,4]  
    except ValueError as e:
        assert str(e) == "attempt to assign sequence of size 3 to extended slice of size 2"

    L = t['A'].copy()
    L[None::-2] = [2,3,"4"]  # SLICE: for new_index, position in enumerate(reversed(range(start,end,-2)): REPLACE L[position] WITH NEW[new_index]
    assert L == ["1", "4", 1.0, 3, 1, 2]  #                                                       ! ----^

    # Note that L[None::-2] becomes range(*slice(None,None,-2).indices(len(L))) == range(5,-1,-2)  !!!!
    L = t['A'].copy()
    new = [2,3,"4"]
    for new_ix,pos in enumerate(range(*slice(None,None,-2).indices(len(L)))):
        L[pos] = new[new_ix] 
    assert L == ["1", "4", 1.0, 3, 1, 2]

    # What happens if we leave out the first : ?
    L = t['A'].copy()
    L[:-2] = [2,"3",4]  # SLICE: REPLACE L before -2 with NEW
    assert L == [2,"3",4,1,1.0] 

    # THIS MEANS THAT None is an active OPERATOR that has different meaning depending on the args position.
    L = t['A'].copy()
    L[None:None:-2] = [2,"3",4.0]
    assert L == ["1", 4, 1.0, "3", 1, 2]

    # THAT SETITEM AND GETITEM BEHAVE DIFFERENT DEPENDING ON THE NUMBER OF ARGUMENTS CAN SEEM VERY ARCHAIC !

    t = Table()
    t['A'] = ["1",2,3.0]
    
    L = t['A'].copy()
    # L[None] = []  # TypeError: list indices must be integers or slices, not NoneType
    assert list(L[None : None]) == ["1",2,3.0]
    assert list(L[None : None:   1  ]) == ["1",2,3.0]
    assert list(L[None : None : None]) == ["1",2,3.0]
    assert list(L[  1  : None : None]) == [2,3.0]
    assert list(L[None :   1  : None]) == ["1"]
    assert list(L[None : None :  2  ]) == ["1",3]

    L = t['A'].copy()
    L[None : None] = [4,5]  
    assert L == [4,5]

    L = t['A'].copy()
    L[None : None:   1  ] = [4,5]  
    assert L==[4,5]

    L = t['A'].copy()
    L[None : None : None] = [4,5]  
    assert L == [4,5]

    L = t['A'].copy()
    L[  1  : None : None] = [4,5]  
    assert L == ["1",4,5]

    L = t['A'].copy()
    L[None :   1  : None] = [4,5]  
    assert L == [4,5,2,3.0]

    L = t['A'].copy()
    L[None : None :  2  ] = [4,5]  
    assert L==[4,2,5]

    L = t['A'].copy()
    L.append(4)
    del L[1:3]
    assert L == ["1",4]

    L[:] = [1,2,3.0,4,"5",6,7,8,9]
    del L[::3]
    assert L == [2,3.0,"5",6,8,9]


def test03_verify_column_summaries():  # test special column functions.
    t = Table()
    n,m = 5,3
    t['A'] = list(range(n)) * m
    col = t['A']
    k,v = col.histogram()
    assert len(k) == n
    assert sum(k1*v1 for k1,v1 in zip(k,v)) == sum(col)
    uq = col.unique()
    assert len(uq) == n
    assert sum(uq) == sum(range(n))
    ix = col.index()
    assert len(ix) == n


def test04_verify_add_rows_for_table():
    table4 = Table()
    table4['A', 'B', 'C'] = [ list(range(20)), [str(i) for i in range(20)], [1.1*i for i in range(20)]]  # test multiple assignment.
    
    table5 = table4 * 10
    assert len(table5) == len(table4)*10  # test __mul__

    assert table4['A'] != table5['A']  # test comparison of column.__eq__
    assert table5['A'] == table5['A']
    assert table5 == table5  # test comparison of table.__eq__
    assert table5 != table4

    for ix, row in enumerate(table4.rows,start=1):  # test .rows
        assert len(row) ==3
        a,b,c = row
        assert a.dtype == int
        assert type(b) == str
        assert c.dtype == float
    assert ix == 20, "there are supposed to be 20 rows."

    t = Table()
    t.add_columns('row','A','B','C')
    t.add_rows(1, 1, 2, 3)  # individual values
    t.add_rows([2, 1, 2, 3])  # list of values
    t.add_rows((3, 1, 2, 3))  # tuple of values
    t.add_rows(*(4, 1, 2, 3))  # unpacked tuple
    t.add_rows(row=5, A=1, B=2, C=3)   # keyword - args
    t.add_rows(**{'row': 6, 'A': 1, 'B': 2, 'C': 3})  # dict / json.
    t.add_rows((7, 1, 2, 3), (8, 4, 5, 6))  # two (or more) tuples.
    t.add_rows([9, 1, 2, 3], [10, 4, 5, 6])  # two or more lists
    t.add_rows({'row': 11, 'A': 1, 'B': 2, 'C': 3},
              {'row': 12, 'A': 4, 'B': 5, 'C': 6})  # two (or more) dicts as args - roughly comma sep'd json.
    t.add_rows(*[{'row': 13, 'A': 1, 'B': 2, 'C': 3},
                {'row': 14, 'A': 1, 'B': 2, 'C': 3}])  # list of dicts.
    t.add_rows(row=[15,16], A=[1,1], B=[2,2], C=[3,3])  # kwargs - lists
    assert t['row'] == list(range(1,17))

def test04_verify_multiprocessing_index_in_shared_memory():
    pass  # multi processing index. with shared memory. :Pool.starmap


def test04_verify_table_stacking():
    pass  # test "stacking"


def test05_verify_show_table():
    table4 = Table()
    txt = table4.to_ascii()
    assert txt == "Empty table"  

    table4.add_columns('A', 'B', 'C')
    txt2 = table4.to_ascii()
    assert txt2 == '+=+=+=+\n|A|B|C|\n+-+-+-+\n+=+=+=+'
    for i in range(5):
        table4['A'] += [i]
        table4['B'] += [str(i+9)]
        table4['C'] += [1.1*i]
        txt = table4.to_ascii()
        # +=+==+==================+
        # |A|B |        C         |
        # +-+--+------------------+
        # |0| 9 |               0.0|
        # |1|10 |               1.1|
        # |2|11 |               2.2|
        # |3|12 |3.3000000000000003|
        # |4|13 |               4.4|
        # +=+==+==================+
        assert txt.count('\n') == i+4
        
    table4.show()  # launch the print function.
    table4 *= 10
    table4.show()


def test06_verify_multi_key_indexing_for_tables():
    # doing lookups is supported by indexing
    table6 = Table()
    table6['A'] = ['Alice', 'Bob', 'Bob', 'Ben', 'Charlie', 'Ben','Albert']
    table6['B'] = ['Alison', 'Marley', 'Dylan', 'Affleck', 'Hepburn', 'Barnes', 'Einstein']
    table6.show()

    index = table6.index('A')  # single key.
    assert index[('Bob',)] == {1, 2}
    index2 = table6.index('A', 'B')  # multiple keys.
    assert index2[('Bob', 'Dylan')] == {2}

    table6.copy_to_clipboard()
    t = Table.copy_from_clipboard()
    t.show()

def test07_verify_import_into_table():
    pass  # import data

def test08_verify_filter_table():
    pass  # filter

    t3 = Table.load_file(BIG_HDF5)
    t3_true, t3_false = t3.filter(columns=[('vendor case weight', ">", 2.0)])
    assert t3.columns == t3_true.columns == t3_false.columns
    assert len(t3_true) != 0
    assert len(t3_false) != 0
    assert len(t3_true) + len(t3_false) == len(t3)

def test09_verify_sort():
    pass  # sort  - sort as it appears as string

def test10_verify_join():
    pass  # join + explain join matrix.
    

def test11_verify_loopup_functionality():
    pass  # lookup

def test12_verify_groupby_functionality():
    pass  # groupby

def test13_verify_pivot_table_functionality():
    pass  # pivot table.
              #tabs
              #spaces