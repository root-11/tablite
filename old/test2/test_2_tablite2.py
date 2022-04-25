import time
import pathlib
import numpy as np
from collections import defaultdict
from tablite2 import Table, Column
from tablite2.settings import HDF5_CACHE_DIR, HDF5_CACHE_FILE



GLOBAL_CLEANUP = False
BIG_PATH = r"d:\remove_duplicates2.csv"
BIG_FILE = pathlib.Path(BIG_PATH)
BIG_HDF5 = pathlib.Path(str(BIG_FILE) + '.hdf5')


def setup():
    cache_file = pathlib.Path(HDF5_CACHE_DIR) / HDF5_CACHE_FILE
    if cache_file.exists():
        cache_file.unlink()

def teardown():
    cache_file = pathlib.Path(HDF5_CACHE_DIR) / HDF5_CACHE_FILE
    if cache_file.exists():
        cache_file.unlink()


def test_000_add_all_datatypes():
    from datetime import datetime
    now = datetime.now().replace(microsecond=0)
    table4 = Table()
    table4['A'] = [-1, 1]
    table4['B'] = [None, 1]     # (1)
    table4['C'] = [-1.1, 1.1]
    table4['D'] = ["", "1000"]     # (2)
    table4['E'] = [None, "1"]   # (1,2)
    table4['F'] = [False, True]
    table4['G'] = [now, now]
    table4['H'] = [now.date(), now.date()]
    table4['I'] = [now.time(), now.time()]
    # (1) with `allow_empty=True` `None` is permitted.
    # (2) Empty string is not a None, when datatype is string.
    table4.show()

    for name in 'ABCDEFGHI':
        dt = []
        for v in table4[name]:
            dt.append(type(v))
        print(name, dt)
    # + test for use_disk=True
    table4

def test_000_add_data():
    t = Table()
    t['row'] = []
    t['A'] = []
    t['B', 'C'] = [ [], [] ]

    t.add_row(1, 1, 2, 3)  # individual values
    t.add_row([2, 1, 2, 3])  # list of values
    t.add_row((3, 1, 2, 3))  # tuple of values
    t.add_row(*(4, 1, 2, 3))  # unpacked tuple
    t.add_row(row=5, A=1, B=2, C=3)   # keyword - args
    t.add_row(**{'row': 6, 'A': 1, 'B': 2, 'C': 3})  # dict / json.
    t.add_row((7, 1, 2, 3), (8, 4, 5, 6))  # two (or more) tuples.
    t.add_row([9, 1, 2, 3], [10, 4, 5, 6])  # two or more lists
    t.add_row({'row': 11, 'A': 1, 'B': 2, 'C': 3},
              {'row': 12, 'A': 4, 'B': 5, 'C': 6})  # two (or more) dicts as args.
    t.add_row(*[{'row': 13, 'A': 1, 'B': 2, 'C': 3},
                {'row': 14, 'A': 1, 'B': 2, 'C': 3}])  # list of dicts.


def test_000_slicing():
    table1 = Table()
    base_data = list(range(10_000))
    table1.add_column('A', data=base_data)
    table1.add_column('B', data=[v*10 for v in base_data])
    table1.add_column('C', data=[-v for v in base_data])
    start = time.time()
    big_table = table1 * 10_000  # = 100_000_000
    print(f"it took {time.time()-start} to extend a table to {len(big_table)} rows")
    start = time.time()
    _ = big_table.copy()
    print(f"it took {time.time()-start} to copy {len(big_table)} rows")
    
    a_preview = big_table['A', 'B', 1_000:900_000:700]
    for row in a_preview[3:15:3].rows:
        print(row)
    a_preview.show(format='ascii')


def test_000_columns():  
    pass

def test_001_memory_manager():
    pass

def test_002_task_manager():
    pass

def test_003_worker_functions():  # single proc
    # excel reader
    # ods reader
    # text reader
    # sort
    # filter
    # groupby
    # join
    # lookup
    # NOT append  += tests for that
    # NOT selector  sliceing tests for that.
    # NOT operations 
    pass



def test_004_multiproc_worker_functions():  # multi proc
    global BIG_HDF5
    if BIG_HDF5.exists():
        BIG_HDF5.unlink()

    columns = {  # numpy type codes: https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
        'SKU ID': 'i', # integer
        'SKU description':'S', # np variable length str
        'Shipped date' : 'S', #datetime
        'Shipped time' : 'S', # integer to become time
        'vendor case weight' : 'f'  # float
    }  

    # now use multiprocessing
    start = time.time()
    t1 = Table.import_file(BIG_PATH, import_as='csv', columns=columns, delimiter=',', text_qualifier=None, newline='\n', first_row_has_headers=True)
    end = time.time()
    print(f"import took {round(end-start, 4)} secs.")

    start = time.time()
    t2 = Table.load_file(BIG_HDF5)
    end = time.time()
    print(f"reloading an imported table took {round(end-start, 4)} secs.")
    t1.show()
    print("-"*120)
    t2.show()

    # re-import bypass check
    start = time.time()
    t3 = Table.import_file(BIG_PATH, import_as='csv', columns=columns, delimiter=',', text_qualifier=None, newline='\n', first_row_has_headers=True)
    end = time.time()
    print(f"reloading an already imported table took {round(end-start, 4)} secs.")

    t3.show(slice(3,100,17))


def test_005_top_level_api():
    """
    Table
        .import  creates .hdf5 table
            includes import settings in hash.
        .load  <-- loads external references
        .source {name: (for each column hdf5 source + index)}
        .show  (read slice of source from index)

            sort AZ, ZA  <--- view only! the source doesn't change.
            unique values
            filter by condition [
                is empty, is not empty, 
                text {contains, does not contain, starts with, ends with, is exactly},
                date {is, is before, is after}
                value is {> >= < <= == != between, not between}
                formula (uses eval)
            ]
            filter by values [ unique values ]
            
        .materialize  (create hdf5 from source) 

        .__iter__  reads the view  -- ops uses __iter__ 
    """
    tbl1 = Table.import_file('d:\remove_duplicates.csv', table_key='2345eafd2faf')  # table_key is from datamap or just a counter in the script.
    tbl2 = Table.load_file('d:\remove_duplicates.csv.hdf5')  # no additional storage is needed because the key is the same.
    tbl3 = Table.from_json('d:\some.json')
    tbl3 = tbl1 + tbl2   # append
    tb3_copy = tbl3.copy()  # virtual dataset.
    tbl4 = tbl3.sort(date=True, order=False)  # date ascending, order descending.
    tbl5 = tbl1.join(tbl2, left_keys=['date'], right_key=['date'], left_columns=['date', 'order'], right_columns=['quantity'], type='left')
    tbl6 = tbl1.lookup(tbl2, left_keys=['date'], right_key=['date'], left_columns=['date', 'order'], right_columns=['quantity'], expr='==')
    dates,quantities = tbl6['date'], tbl6['quantity']
    
    def pct(A,B):  # custom ops.
        d = defaultdict(int)
        for a,b in zip(A,B):
            d[a]+=b        
        return [b/d[a] for a,b in zip(A,B)]

    tbl6['pct'] = Column(dtype=np.float, data=pct(dates,quantities))  # adding column to tbl6.
    # tbl6[int]  --> row
    # tbl6[slice] --> n rows
    # tbl6[text]  --> column
    # tbl6[args]  --> columns if text, rows if slice or int
    
    tbl7 = tbl2.filter('date')


def test_006_callisto_guarantee():
    """
    UPLOAD = HDF5 file
    APPEND = virtual dataset
    SORT = make sort index + autosave to HDF5
    GROUPBY = HDF5 file
    PIVOT = GROUPBY view.
    SELECTOR = virtual dataset OR HDF5 file
    JOIN = paired index + autosave to HDF5
    FILTER = mask + autosave to HDF5
    LOOKUP = paired index
    OPERATION = HDF5 update column or create new column.
    """
    pass


if __name__ == "__main__":
    setup()

    for k,v in {k:v for k,v in sorted(globals().items()) if k.startswith('test') and callable(v)}.items():
        print(20 * "-" + k + "-" * 20)
        v()

    teardown()
