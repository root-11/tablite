import pathlib
import tempfile
import random

import shutil

import logging
log = logging.getLogger()
log.setLevel(logging.INFO)

from datetime import datetime
from string import ascii_uppercase


d = tempfile.gettempdir()
tempdir_name = pathlib.Path(d) / "tablite_synthetic_test_data" 
if not tempdir_name.exists():
    tempdir_name.mkdir()


def remove_synthetic_data():
    """ 
    One call to clean up all synthetic data.
    """    
    if not "tablite_synthetic_test_data" in str(tempdir_name):
        raise ValueError(f"tempdir_name has been changed. Aborting clean up.: {tempdir_name}")

    print(f"removing synthetic data from {tempdir_name}", end='')
    shutil.rmtree(str(tempdir_name))
    print("...done")


def synthetic_order_data_csv(rows=100_000, name=None):
    rows = int(rows)

    if name is None:
        name = f"{rows}.csv"

    path = tempdir_name / name
    if path.exists():
        print("file already exists ...")
        return path

    print(f'creating {path} with {rows:,} rows', end=":")
    headers = ["#", "1","2","3","4","5","6","7","8","9","10","11"]
    with path.open(mode='w', encoding='utf-8') as fo:
        fo.write(",".join(headers) + "\n")  # headers

        L1 = ['None', '0°', '6°', '21°']
        L2 = ['ABC', 'XYZ', ""]
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
                random.choice(L1),  # random temperature group.
                random.choice(L2),  # random choice of category
                random.uniform(0.01, 2.5),  # volume?
                f"{random.uniform(0.1, 25)}\n"  # units?
            ]
            assert len(row) == len(headers)

            fo.write(",".join(str(i) for i in row))
            if row_no % (rows/100) == 0: 
                print(".",end="")

    assert isinstance(path, pathlib.Path)
    return path




