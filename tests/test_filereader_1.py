import time
import pathlib
import pytest

from tablite import Table

@pytest.fixture(autouse=True) # this resets the HDF5 file for every test.
def refresh():
    Table.reset_storage()
    yield


def test01_timing():
    table1 = Table()
    base_data = list(range(10_000))
    table1['A'] = base_data
    table1['B'] = [v*10 for v in base_data]
    table1['C'] = [-v for v in base_data]
    start = time.time()
    big_table = table1 * 10_000  # = 100_000_000
    print(f"it took {round(time.time()-start,3)}secs to extend a table to {len(big_table):,} rows")
    start = time.time()
    _ = big_table.copy()
    print(f"it took {round(time.time()-start,3)}secs to copy {len(big_table):,} rows")
    
    a_preview = big_table['A', 'B', 1_000:900_000:700]
    rows = [r for r in a_preview[3:15:3].rows]
    assert rows == [[3100,31000], [5200,52000], [7300, 73000], [9400, 94000]]
    a_preview.show()
  
    
def test01():
    BIG_PATH = r"d:\remove_duplicates2.csv"
    BIG_FILE = pathlib.Path(BIG_PATH)
    BIG_HDF5 = pathlib.Path(str(BIG_FILE) + '.hdf5')
    if not BIG_FILE.exists():
        return

    if BIG_HDF5.exists():
        BIG_HDF5.unlink()

    columns = {  # numpy type codes: https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
        'SKU ID': 'i', # integer
        'SKU description':'S', # np variable length str
        'Shipped date' : 'S', # datetime
        'Shipped time' : 'S', # integer to become time
        'vendor case weight' : 'f'  # float
    }  

    start = time.time()
    t1 = Table.import_file(BIG_PATH, import_as='csv', columns=columns, delimiter=',', text_qualifier=None, newline='\n', first_row_has_headers=True)
    end = time.time()
    print(f"import took {round(end-start, 4)} secs.")
    
    start = time.time()
    t2 = Table.import_file(BIG_PATH, import_as='csv', columns=columns, delimiter=',', text_qualifier=None, newline='\n', first_row_has_headers=True)
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

    if BIG_HDF5.exists():
        BIG_HDF5.unlink()


