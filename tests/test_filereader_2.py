import tempfile
from pathlib import Path
import random
from datetime import datetime
from tablite import Table
from string import ascii_uppercase
from time import process_time
import pytest



@pytest.fixture(autouse=True) # this resets the HDF5 file for every test.
def refresh():
    Table.reset_storage()
    yield



def test_csv_reader():
    rows = 100_000

    d = tempfile.gettempdir()
    f = Path(d) / "large.csv"

    headers = ["#", "1","2","3","4","5","6","7","8","9","10","11"]
    with f.open(mode='w', encoding='utf-8') as fo:
        fo.write(",".join(headers) + "\n")  # headers

        for row_no in range(1,rows+1):  # rows
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
    assert isinstance(f, Path)

    start = process_time()
    t = Table.import_file(f, import_as='csv', columns={k:'f' for k in headers})
    end = process_time()
    t.show()

    print(f"loading took {round(end - start, 3)} secs. for {len(t)} rows")
    assert len(t) == rows
    assert end-start < 4 * 60  # 4 minutes.

    f.unlink()


