import math
import random
from datetime import datetime
from string import ascii_uppercase
from tablite import Table
from tablite.config import Config


def synthetic_order_data(rows=100_000):
    """Creates a synthetic dataset for testing that looks like this:
    (depending on number of rows)

    ```
    +=========+=======+=============+===================+=====+===+=====+====+===+=====+=====+===================+==================+
    |    ~    |   #   |      1      |         2         |  3  | 4 |  5  | 6  | 7 |  8  |  9  |         10        |        11        |
    |   row   |  int  |     int     |      datetime     | int |int| int |str |str|mixed|mixed|       float       |      float       |
    +---------+-------+-------------+-------------------+-----+---+-----+----+---+-----+-----+-------------------+------------------+
    |0        |      1|1478158906743|2021-10-27 00:00:00|50764|  1|29990|C4-5|APP|21°  |None | 2.0434376837650046|1.3371665497020444|
    |1        |      2|2271295805011|2021-09-13 00:00:00|50141|  0|10212|C4-5|TAE|None |None |  1.010318612835485| 20.94821610676901|
    |2        |      3|1598726492913|2021-08-19 00:00:00|50527|  0|19416|C3-5|QPV|21°  |None |  1.463459515469516|  17.4133659842749|
    |3        |      4|1413615572689|2021-11-05 00:00:00|50181|  1|18637|C4-2|GCL|6°   |ABC  |  2.084002469706324| 0.489481411683505|
    |4        |      5| 245266998048|2021-09-25 00:00:00|50378|  0|29756|C5-4|LGY|6°   |XYZ  | 0.5141579343276079| 8.550780816571438|
    |5        |      6| 947994853644|2021-10-14 00:00:00|50511|  0| 7890|C2-4|BET|0°   |XYZ  | 1.1725893606177542| 7.447314130260951|
    |6        |      7|2230693047809|2021-10-07 00:00:00|50987|  1|26742|C1-3|CFP|0°   |XYZ  | 1.0921267279498004|11.009210185311993|
    |...      |...    |...          |...                |...  |...|...  |... |...|...  |...  |...                |...               |
    |7,999,993|7999994|2047223556745|2021-09-03 00:00:00|50883|  1|15687|C3-1|RFR|None |XYZ  | 1.3467185981566827|17.023443485654845|
    |7,999,994|7999995|1814140654790|2021-08-02 00:00:00|50152|  0|16556|C4-2|WTC|None |ABC  | 1.1517593924478968| 8.201818634721487|
    |7,999,995|7999996| 155308171103|2021-10-14 00:00:00|50008|  1|14590|C1-3|WYM|0°   |None | 2.1273836233717978|23.295943554889195|
    |7,999,996|7999997|1620451532911|2021-12-12 00:00:00|50173|  1|20744|C2-1|ZYO|6°   |ABC  |  2.482509134693724| 22.25375464857266|
    |7,999,997|7999998|1248987682094|2021-12-20 00:00:00|50052|  1|28298|C5-4|XAW|None |XYZ  |0.17923757926558143|23.728160892974252|
    |7,999,998|7999999|1382206732187|2021-11-13 00:00:00|50993|  1|24832|C5-2|UDL|None |ABC  |0.08425329763360942|12.707735293126758|
    |7,999,999|8000000| 600688069780|2021-09-28 00:00:00|50510|  0|15819|C3-4|IGY|None |ABC  |  1.066241687256579|13.862069804070295|
    +=========+=======+=============+===================+=====+===+=====+====+===+=====+=====+===================+==================+
    ```

    Args:
        rows (int, optional): number of rows wanted. Defaults to 100_000.

    Returns:
        Table (Table): Populated table.
    """  # noqa
    rows = int(rows)

    L1 = ["None", "0°", "6°", "21°"]
    L2 = ["ABC", "XYZ", ""]

    t = Table()
    assert isinstance(t, Table)
    for page_n in range(math.ceil(rows / Config.PAGE_SIZE)):  # n pages
        start = (page_n * Config.PAGE_SIZE)
        end = min(start + Config.PAGE_SIZE, rows)
        ro = range(start, end)

        t2 = Table()
        t2["#"] = [v+1 for v in ro]
        # 1 - mock orderid
        t2["1"] = [random.randint(18_778_628_504, 2277_772_117_504) for i in ro]
        # 2 - mock delivery date.
        t2["2"] = [datetime.fromordinal(random.randint(738000, 738150)).isoformat() for i in ro]
        # 3 - mock store id.
        t2["3"] = [random.randint(50000, 51000) for _ in ro]
        # 4 - random bit.
        t2["4"] = [random.randint(0, 1) for _ in ro]
        # 5 - mock product id
        t2["5"] = [random.randint(3000, 30000) for _ in ro]
        # 6 - random weird string
        t2["6"] = [f"C{random.randint(1, 5)}-{random.randint(1, 5)}" for _ in ro]
        # 7 - # random category
        t2["7"] = ["".join(random.choice(ascii_uppercase) for _ in range(3)) for _ in ro]
        # 8 -random temperature group.
        t2["8"] = [random.choice(L1) for _ in ro]
        # 9 - random choice of category
        t2["9"] = [random.choice(L2) for _ in ro]
        # 10 - volume?
        t2["10"] = [random.uniform(0.01, 2.5) for _ in ro]
        # 11 - units?
        t2["11"] = [f"{random.uniform(0.1, 25)}" for _ in ro]

        if len(t) == 0:
            t = t2
        else:
            t += t2

    return t
