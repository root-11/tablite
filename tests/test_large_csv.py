import tempfile
from pathlib import Path
import random
from datetime import datetime
from tablite import Table
from string import ascii_uppercase
from time import process_time


def test_csv_reader():
    rows = 1_000_000

    d = tempfile.gettempdir()
    f = Path(d) / "large.csv"
    with f.open(mode='w', encoding='utf-8') as fo:
        fo.write(",".join([f'"{i}"' for i in range(1, 13)]) + "\n")  # headers

        for row_no in range(rows):  # rows
            row = [
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

            fo.write(",".join(str(i) for i in row))
    assert isinstance(f, Path)

    start = process_time()
    t = next(Table.from_file(f))
    end = process_time()
    t.show(slice(0, 5, 1))
    # +=============+===================+=====+=====+=====+=====+=====+=====+=====+===================+==================+=====+
    # |      1      |         2         |  3  |  4  |  5  |  6  |  7  |  8  |  9  |         10        |        11        |  12 |
    # |     int     |      datetime     | int | int | int | str | str | str | str |       float       |      float       | str |
    # |    False    |       False       |False|False|False|False|False| True| True|       False       |      False       | True|
    # +-------------+-------------------+-----+-----+-----+-----+-----+-----+-----+-------------------+------------------+-----+
    # |1438288346586|2021-10-04 00:00:00|50118|    0|26152|C2-5 |JXZ  |None |XYZ  | 0.9102364462640427| 3.354410249609248|None |
    # |1023468775534|2021-11-13 00:00:00|50335|    0|17800|C5-3 |PUZ  |None |None |0.27890222331557396|14.154362070905446|None |
    # |1657146367613|2021-11-28 00:00:00|50744|    1|18353|C5-4 |CCE  |6°   |ABC  | 1.8871897850890424|10.592385104357932|None |
    # |1585773266616|2021-09-17 00:00:00|50553|    1| 6309|C3-2 |RDX  |6°   |XYZ  | 1.7898736645664135|24.302329121273914|None |
    # | 651293331020|2021-08-09 00:00:00|50836|    0|10618|C5-5 |YAS  |6°   |ABC  |0.22622667042527606| 7.416767055013624|None |
    # +=============+===================+=====+=====+=====+=====+=====+=====+=====+===================+==================+=====+
    # (showing 5 of 1000000 rows)

    t.show(slice(rows-5, rows))
    # +=============+===================+=====+=====+=====+=====+=====+=====+=====+==================+==================+=====+
    # |      1      |         2         |  3  |  4  |  5  |  6  |  7  |  8  |  9  |        10        |        11        |  12 |
    # |     int     |      datetime     | int | int | int | str | str | str | str |      float       |      float       | str |
    # |    False    |       False       |False|False|False|False|False| True| True|      False       |      False       | True|
    # +-------------+-------------------+-----+-----+-----+-----+-----+-----+-----+------------------+------------------+-----+
    # |1401907043992|2021-10-13 00:00:00|50077|    0| 7717|C2-4 |THK  |6°   |None |0.1920787164614819|   8.8804431343882|None |
    # |1611923334128|2021-10-04 00:00:00|50472|    1|15378|C2-1 |GXE  |6°   |None | 2.343946122395069|14.030124773502914|None |
    # | 851098464124|2021-11-29 00:00:00|50231|    1|29520|C2-3 |ZYW  |None |ABC  |0.6924680964264815|  7.57510405460726|None |
    # |1105092720373|2021-12-08 00:00:00|50461|    0|20645|C1-3 |MUX  |None |XYZ  |1.6628604417682002| 2.522190738627351|None |
    # |1202142426481|2021-09-30 00:00:00|50478|    1|18259|C2-2 |UAP  |6°   |None |1.4528804349872424|14.620770845550924|None |
    # +=============+===================+=====+=====+=====+=====+=====+=====+=====+==================+==================+=====+
    # (showing 5 of 1000000 rows)

    print(f"loading took {round(end - start, 3)} secs. for {len(t)} rows")
    # loading took 64.562 secs. for 1_000_000 rows
    assert len(t) == rows
    assert end-start < 4 * 60  # 4 minutes.

    f.unlink()


