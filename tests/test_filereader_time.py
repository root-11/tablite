import time
import pathlib
import pytest
import tempfile
import random

from tqdm import tqdm
from datetime import datetime
from string import ascii_uppercase

from tablite import Table, get_headers


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
  

def make_csv_file(rows=100_000):
    
    d = tempfile.gettempdir()
    path = pathlib.Path(d) / "large.csv"

    headers = ["#", "1","2","3","4","5","6","7","8","9","10","11"]
    with path.open(mode='w', encoding='utf-8') as fo:
        fo.write(",".join(headers) + "\n")  # headers

        for row_no in tqdm(range(1,rows+1), total=int(rows), desc="creating test csv file", unit="rows"):  # rows
            row = [
                row_no,
                random.randint(18_778_628_504, 2277_772_117_504),  # 1 - mock orderid
                datetime.fromordinal(random.randint(738000, 738150)).isoformat(),  # 2 - mock delivery date.
                random.randint(50000, 51000),  # 3 - mock store id.
                random.randint(0, 1),  # 4 - random bit.
                random.randint(3000, 30000),  # 5 - mock product id
                f"C{random.randint(1, 5)}-{random.randint(1, 5)}",  # random weird string
                "".join(random.choice(ascii_uppercase) for _ in range(3)),  # random category
                random.choice(['None', '0°', '6°', '21°']),  # random temperature group.
                random.choice(['ABC', 'XYZ', ""]),  # random choice of category
                random.uniform(0.01, 2.5),  # volume?
                f"{random.uniform(0.1, 25)}\n"  # units?
            ]
            assert len(row) == len(headers)

            fo.write(",".join(str(i) for i in row))
    assert isinstance(path, pathlib.Path)
    return path


def test01():
    path = make_csv_file(1e2)
    if not path.exists():
        return
    
    headers = get_headers(path)
    columns = {h:'f' for h in headers[path.name][0]}

    config = {
        "import_as":'csv', 
        "columns":columns, 
        "delimiter":headers['delimiter'], 
        "text_qualifier":None, 
        "newline":'\n', 
        "first_row_has_headers":True
    }

    start = time.time()
    t1 = Table.import_file(path, **config)
    end = time.time()
    print(f"import took {round(end-start, 4)} secs.")   #  import took 335.1049 secs.
    
    start = time.time()
    t2 = Table.import_file(path, **config)
    end = time.time()
    print(f"reloading an imported table took {round(end-start, 4)} secs.")  # reloading an imported table took 0.177 secs.
    t1.show()
    print("-"*120)
    t2.show()

    # re-import bypass check
    start = time.time()
    t3 = Table.import_file(path, **config)
    end = time.time()
    print(f"reloading an already imported table took {round(end-start, 4)} secs.")  #reloading an already imported table took 0.179 secs.
    t3.show(slice(3,100,17))




