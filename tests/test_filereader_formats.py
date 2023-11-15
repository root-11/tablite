from tablite.config import Config
from tablite import Table
from tablite.file_reader_utils import TextEscape, get_headers, ENCODING_GUESS_BYTES
from tablite.datatypes import DataTypes
from time import process_time_ns
from datetime import date, datetime
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)  # this resets the HDF5 file for every test.
def refresh():
    yield


def test_text_escape():
    text_escape = TextEscape(delimiter=";", openings=None, closures=None)

    te = text_escape('"t"')
    assert te == ["t"]

    te = text_escape('"t";"3";"2"')
    assert te == ["t", "3", "2"]

    te = text_escape('"this";"123";234;"this";123;"234"')
    assert te == ["this", "123", "234", "this", "123", "234"]

    te = text_escape('"this";"123";"234"')
    assert te == ["this", "123", "234"]

    te = text_escape('"this";123;234')
    assert te == ["this", "123", "234"]

    te = text_escape('"this";123;"234"')
    assert te == ["this", "123", "234"]

    te = text_escape('123;"1\'3";234')
    assert te == ["123", "1'3", "234"], te

    te = text_escape('"1000627";"MOC;SEaert;pás;krk;XL;černá";"2.180,000";"CM3";2')
    assert te == ["1000627", "MOC;SEaert;pás;krk;XL;černá", "2.180,000", "CM3", "2"]

    te = text_escape('"1000294";"S2417DG 24"" LED monitor (210-AJWM)";"47.120,000";"CM3";3')
    assert te == ["1000294", 'S2417DG 24" LED monitor (210-AJWM)', "47.120,000", "CM3", "3"]


def test_text_escape2():
    # set up
    text_escape = TextEscape(openings="({[", closures="]})", text_qualifier='"', delimiter=",")
    s = 'this,is,a,,嗨,"(comma,sep\'d)","text"'
    L = text_escape(s)
    assert L == ["this", "is", "a", "", "嗨", "(comma,sep'd)", "text"]


def test_text_escape_without_text_qualifier():
    text_escape = TextEscape(openings="({[", closures="]})", delimiter=",")

    s = "this,is,a,,嗨,\"(comma,sep'd)\",text"
    L = text_escape(s)
    assert L == ["this", "is", "a", "", "嗨", "(comma,sep'd)", "text"]


def test_get_headers():
    """this test does not look for content. It merely checks that the reader doesn't fail."""
    folder = Path(__file__).parent / "data"
    for fname in folder.iterdir():
        d = get_headers(fname)
        assert isinstance(d, dict)
        assert d
        print(fname)
        print(d)

def test_get_headers_single_column():
    """this test does not look for content. It merely checks that the reader doesn't fail."""
    fname = Path(__file__).parent / "data" / "simple.csv"

    headers = get_headers(fname, header_row_index=1, linecount=0)
    assert headers["simple.csv"] == [["header"]]

def test_filereader_empty_fields():
    for i in range(5):
        csv_file = Path(__file__).parent / "data" / "bad_empty.csv"

        from tablite.config import Config

        Config.MULTIPROCESSING_MODE = Config.FALSE

        t = Table.from_file(csv_file, text_qualifier='"')
        expected = Table(
            columns={
                "A": [None, 3, 1, None, None],
                "B": [None, 3333, None, 2, None],
                "C": [None, 3, None, None, 3],
            }
        )
        t.show()
        assert t == expected, i #t.show()


def test_filereader_123csv():
    csv_file = Path(__file__).parent / "data" / "123.csv"

    table7 = Table(
        columns={
            "A": [1, None, 8, 3, 4, 6, 5, 7, 9],
            "B": [10, 100, 1, 1, 1, 1, 10, 10, 10],
            "C": [0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    sort_order = {"B": False, "C": False, "A": False}
    table7.sort(sort_order)

    headers = ",".join([c for c in table7.columns])
    data = [headers]
    for row in table7.rows:
        data.append(",".join(str(v) for v in row))

    s = "\n".join(data)
    print(s)
    csv_file.write_text(s)  # write
    tr_table = Table.from_file(csv_file, columns=["A", "B", "C"])
    csv_file.unlink()  # cleanup

    tr_table.show()
    for c in tr_table.columns:
        col = tr_table[c]
        col[:] = DataTypes.guess(col)

    tr_table.show()

    assert tr_table == table7


def test_filereader_csv_f12():
    path = Path(__file__).parent / "data" / "f12.csv"
    columns = [
        "Prod Slbl",
        "Prod Tkt Descp Txt",
        "Case Qty",
        "Height",
        "Width",
        "Length",
        "Weight",
        "sale_date",
        "cust_nbr",
        "Case Qty_1",
        "EA Location",
        "CDY/Cs",
        "EA/Cs",
        "EA/CDY",
        "Ordered As",
        "Picked As",
        "Cs/Pal",
        "SKU",
        "Order_Number",
        "cases",
    ]
    data = Table.from_file(path, columns=columns)
    assert len(data) == 13
    # fmt:off
    for val, gt in zip(list(data.rows), [
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 1365660, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "1365660_2012/01/01", 1],
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 1696753, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "1696753_2012/01/01", 1],
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 1828693, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "1828693_2012/01/01", 1],
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 2211182, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "2211182_2012/01/01", 2],
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 2229312, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "2229312_2012/01/01", 1],
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 2414206, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "2414206_2012/01/01", 1],
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 266791, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "266791_2012/01/01", 2],
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 2, 0, 0), 1017988, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "1017988_2012/01/02", 1],
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 2, 0, 0), 1020158, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "1020158_2012/01/02", 2],
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 2, 0, 0), 1032132, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "1032132_2012/01/02", 1],
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 2, 0, 0), 1048323, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "1048323_2012/01/02", 1],
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 2, 0, 0), 1056865, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "1056865_2012/01/02", 2],
        [52609, "3.99 BTSZ RDS TO", 7, 16, 16, 22, 20, datetime(2012, 1, 2, 0, 0), 1057577, 7, "EA", 0, 7, 0, "Each", "Each", 72, 587, "1057577_2012/01/02", 0],
    ]):
        assert len(val) == len(gt)
        for v, g in zip(val, gt):
            assert v == g, f"'{v}' != '{g}'"
    # fmt:on


def test_filereader_book1csv():
    path = Path(__file__).parent / "data" / "book1.csv"
    assert path.exists()
    table = Table.from_file(path, columns=["a", "b", "c", "d", "e", "f"])
    table.show(slice(0, 10))
    for name in table.columns:
        table[name] = DataTypes.guess(table[name])

    assert table["a"].types() == {int: 45}
    for name in list("bcdef"):
        assert table[name].types() == {float: 45}

    assert len(table) == 45


def test_filereader_book1tsv():
    path = Path(__file__).parent / "data" / "book1.tsv"
    assert path.exists()
    table = Table.from_file(path, columns=["a", "b", "c", "d", "e", "f"], delimiter="\t", text_qualifier=None)
    table.show(slice(0, 10))
    assert len(table) == 45


def test_filereader_gdocs1csv():
    path = Path(__file__).parent / "data" / "gdocs1.csv"
    assert path.exists()
    table = Table.from_file(path, columns=["a", "b", "c", "d", "e", "f"], text_qualifier=None)
    table.show(slice(0, 10))
    assert len(table) == 45


def test_filereader_book1txt():
    path = Path(__file__).parent / "data" / "book1.txt"
    assert path.exists()
    table = Table.from_file(path, columns=["a", "b", "c", "d", "e", "f"], delimiter="\t", text_qualifier=None)
    table.show(slice(0, 10))
    assert len(table) == 45


def test_filereader_book1_txt_chunks():
    path = Path(__file__).parent / "data" / "book1.txt"
    assert path.exists()
    table1 = Table.from_file(path, columns=["a", "b", "c", "d", "e", "f"], delimiter="\t", text_qualifier=None)
    start = 0
    table2 = None
    while True:
        tmp = Table.from_file(
            path, columns=["a", "b", "c", "d", "e", "f"], delimiter="\t", text_qualifier=None, start=start, limit=5
        )
        if len(tmp) == 0:
            break
        start += len(tmp)

        if table2 is None:
            table2 = tmp
        else:
            table2 += tmp

    assert table1 == table2


def test_filereader_book1_txt_chunks_and_offset():
    path = Path(__file__).parent / "data" / "book1.txt"
    assert path.exists()

    start = 2

    table1 = Table.from_file(
        path, columns=["a", "b", "c", "d", "e", "f"], delimiter="\t", text_qualifier=None, start=0
    )

    table1 = Table.from_file(
        path, columns=["a", "b", "c", "d", "e", "f"], delimiter="\t", text_qualifier=None, start=start
    )

    table2 = None
    while True:
        tmp = Table.from_file(
            path, columns=["a", "b", "c", "d", "e", "f"], delimiter="\t", text_qualifier=None, start=start, limit=5
        )
        if len(tmp) == 0:
            break
        start += len(tmp)
        if table2 is None:
            table2 = tmp
        else:
            table2 += tmp

    assert table1 == table2


def test_filereader_gdocsc1tsv():
    path = Path(__file__).parent / "data" / "gdocs1.tsv"
    assert path.exists()
    table = Table.from_file(path, columns=["a", "b", "c", "d", "e", "f"], text_qualifier=None, delimiter="\t")
    table.show(slice(0, 10))
    assert len(table) == 45
    for name in table.columns:
        table[name] = DataTypes.guess(table[name])

    assert table["a"].types() == {int: 45}
    for name in list("bcdef"):
        assert table[name].types() == {float: 45}


def test_filereader_gdocsc1ods():
    path = Path(__file__).parent / "data" / "gdocs1.ods"
    assert path.exists()

    sheet1 = Table.from_file(path, sheet="Sheet1")
    for name in sheet1.columns:
        sheet1[name] = DataTypes.guess(sheet1[name])
        assert sheet1[name].types() == {int: 45}

    sheet2 = Table.from_file(path, sheet="Sheet2")
    for name in sheet2.columns:
        sheet2[name] = DataTypes.guess(sheet2[name])
        if name == "a":
            assert sheet2[name].types() == {int: 45}
        else:
            assert sheet2[name].types() == {float: 45}


def test_filereader_gdocs1xlsx():
    path = Path(__file__).parent / "data" / "gdocs1.xlsx"
    assert path.exists()
    sheet1 = Table.from_file(path, sheet="Sheet1", columns=["a", "b", "c", "d", "e", "f"])
    sheet1.show(slice(0, 10))

    for name in sheet1.columns:
        sheet1[name] = DataTypes.guess(sheet1[name])
        assert sheet1[name].types() == {int: 45}
    assert len(sheet1) == 45


def test_filereader_utf8csv():
    path = Path(__file__).parent / "data" / "utf8_test.csv"
    assert path.exists()

    columns = ["Item", "Materiál", "Objem", "Jednotka objemu", "Free Inv Pcs"]
    table = Table.from_file(path, delimiter=";", columns=columns, text_qualifier='"')
    table.show(slice(0, 10))
    table.show(slice(-15, None))

    types = {"Item": int, "Materiál": str, "Objem": float, "Jednotka objemu": str, "Free Inv Pcs": int}

    for name in table.columns:
        table[name] = DataTypes.guess(table[name])
        tt = table[name].types()
        assert max(tt, key=tt.get) == types[name]

    assert len(table) == 99, len(table)


def test_filereader_utf16csv():
    path = Path(__file__).parent / "data" / "utf16_test.csv"
    assert path.exists()
    col_names = ['"Item"', '"Materiál"', '"Objem"', '"Jednotka objemu"', '"Free Inv Pcs"']
    table = Table.from_file(path, delimiter=";", columns=col_names)
    table.show(slice(0, 10))
    # +===+============+=======================================+============+=================+==============+
    # | # |   "Item"   |               "Materiál"              |  "Objem"   |"Jednotka objemu"|"Free Inv Pcs"|
    # |row|    str     |                  str                  |    str     |       str       |     str      |
    # +---+------------+---------------------------------------+------------+-----------------+--------------+
    # |0  |"1000028"   |"SL 70"                                |"1.248,000" |"CM3"            |21            |
    # |1  |"1000031"   |"Karibik 12,5 kg"                      |"41.440,000"|"CM3"            |2             |
    # |2  |"1000036"   |"IH 26"                                |"6.974,100" |"CM3"            |2             |
    # |3  |"1000062"   |"IL 35"                                |"6.557,300" |"CM3"            |15            |
    # |4  |"1000078"   |"Projektor Tomáš se zvukem"            |"8.742,400" |"CM3"            |11            |
    # |5  |"1000081"   |"FC 48"                                |"2.667,600" |"CM3"            |29            |
    # |6  |"1000087004"|"HG Ar Racer Tank Blk Met Sil L"       |"2.552,000" |"CM3"            |2             |
    # |7  |"1000091"   |"MG 520"                               |"18.581,000"|"CM3"            |5             |
    # |8  |"1000094001"|"HG Ar Racer Tank Abs Green Met Sil XS"|"1.386,000" |"CM3"            |4             |
    # |9  |"1000094002"|"HG Ar Racer Tank Abs Green Met Sil S" |"1.216,000" |"CM3"            |7             |
    # +===+============+=======================================+============+=================+==============+
    table.show(slice(-15))
    assert len(table) == 99, len(table)


def test_filereader_win1251_encoding_csv():
    path = Path(__file__).parent / "data" / "win1250_test.csv"
    assert path.exists()
    col_names = ['"Item"', '"Materiál"', '"Objem"', '"Jednotka objemu"', '"Free Inv Pcs"']
    table = Table.from_file(path, delimiter=";", columns=col_names)
    table.show(slice(0, 10))
    table.show(slice(None, -15))
    assert len(table) == 99, len(table)


def test_filereader_utf8sig_encoding_csv():
    path = Path(__file__).parent / "data" / "utf8sig.csv"
    assert path.exists()
    col_names = ["432", "1"]
    table = Table.from_file(path, delimiter=",", columns=col_names)
    table.show(slice(0, 10))
    table.show(slice(-15))
    assert len(table) == 2, len(table)


def test_filereader_saptxt():
    path = Path(__file__).parent / "data" / "sap.txt"
    assert path.exists()

    header = "    | Delivery |  Item|Pl.GI date|Route |SC|Ship-to   |SOrg.|Delivery quantity|SU| TO Number|Material    |Dest.act.qty.|BUn|Typ|Source Bin|Cty"  # noqa
    col_names = [w.strip(" ").rstrip(" ") for w in header.split("|")]

    table = Table.from_file(
        path,
        delimiter="|",
        columns=[k for k in col_names if k != ""],
        strip_leading_and_tailing_whitespace=True,
        guess_datatypes=False,
    )

    for name in table.columns:
        table[name] = DataTypes.guess(table[name])

    table.show()
    # fmt: off
    # +==+====+=========+====+==========+======+==+==========+=====+=================+==+==========+=========+=============+===+===+==========+===+====+
    # |# |    | Delivery|Item|Pl.GI date|Route |SC| Ship-to  |SOrg.|Delivery quantity|SU|TO Number | Material|Dest.act.qty.|BUn|Typ|Source Bin|Cty| _1 |
    # +--+----+---------+----+----------+------+--+----------+-----+-----------------+--+----------+---------+-------------+---+---+----------+---+----+
    # | 0|None|255332458|  10|2016-03-01|KR-SSH|S1|N193799SEA|GB20 |                1|EA|2110950757|LR034266 |            1|EA |122|2406130101|KR |None|
    # | 1|None|255337984|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                9|EA|2110933207|LR069697 |            9|EA |66L|6605020402|KR |None|
    # | 2|None|255337999|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                1|EA|2110933208|LR006253 |            1|EA |400|F103310000|KR |None|
    # | 3|None|255342585|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                1|EA|2110933209|LR076144 |            1|EA |104|130637A001|KR |None|
    # | 4|None|255342838|  10|2016-03-01|JP-SA |S1|N269899AIR|GB20 |               20|EA|2110940618|LR072969 |            0|EA |209|340649C001|JP |None|
    # | 5|None|255342838|  10|2016-03-01|JP-SA |S1|N269899AIR|GB20 |               20|EA|2110938640|LR072969 |           14|EA |209|340649C001|JP |None|
    # | 6|None|255342838|  10|2016-03-01|JP-SA |S1|N269899AIR|GB20 |               20|EA|2110938641|LR072969 |            6|EA |209|340649C001|JP |None|
    # | 7|None|255342842|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                3|EA|2110933210|LR045184 |            3|EA |122|2406060201|KR |None|
    # | 8|None|255342846|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                6|EA|2110933211|LR045184 |            6|EA |122|2406060201|KR |None|
    # | 9|None|255343550|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                1|EA|2110933212|LR006253 |            1|EA |400|F103310000|KR |None|
    # |10|None|255345406|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                1|EA|2110933213|LR006253 |            1|EA |400|F103310000|KR |None|
    # |11|None|255347459|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                1|EA|2110933214|LR076133 |            0|EA |211|130842B002|KR |None|
    # |12|None|255347459|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                1|EA|2110938506|LR076133 |            1|EA |111|131125A002|KR |None|
    # |13|None|255347460|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                2|EA|2110933215|LR076133 |            0|EA |211|130842B002|KR |None|
    # |14|None|255347460|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                2|EA|2110938507|LR076133 |            2|EA |111|131125A002|KR |None|
    # |15|None|255347461|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                2|EA|2110933216|LR076133 |            0|EA |211|130842B002|KR |None|
    # |16|None|255347461|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                2|EA|2110938508|LR076133 |            2|EA |111|131125A002|KR |None|
    # |17|None|255349073|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                1|EA|2110933217|VPLCT0147|            1|EA |205|RS0933A02X|KR |None|
    # |18|None|255352616|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                6|EA|2110933218|LR086385 |            6|EA |100|1505450201|KR |None|
    # |19|None|255352619|  10|2016-03-01|KR-SS |S1|N193799SEA|GB20 |                2|EA|2110933219|LR072471 |            2|EA |100|1680050203|KR |None|
    # +==+====+=========+====+==========+======+==+==========+=====+=================+==+==========+=========+=============+===+===+==========+===+====+
    # fmt: on
    assert len(table) == 20, len(table)


def test_filereader_book1xlsx():
    path = Path(__file__).parent / "data" / "book1.xlsx"
    assert path.exists()
    start = process_time_ns()
    sheet1 = Table.from_file(path, sheet="Sheet1", columns=["a", "b", "c", "d", "e", "f"])
    sheet2 = Table.from_file(
        path, columns=["a", "b", "c", "d", "e", "f"], sheet="Sheet2 "
    )  # there's a deliberate white space at the end!)
    end = process_time_ns()

    tables = [sheet1, sheet2]
    fields = sum(len(t) * len(t.columns) for t in tables)
    print("{:,} fields/seccond".format(round(1e9 * fields / max(1, end - start), 0)))

    for name in "abcdef":
        sheet1[name] = DataTypes.guess(sheet1[name])
        assert sheet1[name].types() == {int: len(sheet1)}

    sheet2["a"] == DataTypes.guess(sheet2["a"])
    assert sheet2["a"].types() == {int: len(sheet2)}

    for name in list("bcdef"):
        sheet2[name] = DataTypes.guess(sheet2[name])
        assert sheet2[name].types() == {float: len(sheet2)}


def test_filereader_exceldatesxlsx():
    path = Path(__file__).parent / "data" / "excel_dates.xlsx"
    assert path.exists()
    try:
        _ = Table.from_file(path, sheet=None, columns=None)
        assert False
    except ValueError as e:
        assert "available sheets" in str(e)

    table2 = Table.from_file(path, sheet="Sheet1", columns=None)
    sample = get_headers(path)
    columns = [k for k in sample["Sheet1"][0]]

    table = Table.from_file(path, sheet="Sheet1", columns=columns)
    assert table == table2

    table.show()
    # +===+===================+=============+==========+=====+
    # | # |        Date       |numeric value|  string  | bool|
    # |row|      datetime     |     int     |   str    | bool|
    # +---+-------------------+-------------+----------+-----+
    # |0  |1920-01-01 00:00:00|            0|1920/01/01| True|
    # |1  |2016-10-31 00:00:00|        42674|2016/10/31|False|
    # +===+===================+=============+==========+=====+
    assert len(table) == 2, len(table)
    assert table["Date"].types() == {datetime: len(table)}
    assert table["numeric value"].types() == {int: len(table)}
    assert table["bool"].types() == {bool: len(table)}

    assert table["string"].types() == {str: len(table)}
    table["string"] = DataTypes.guess(table["string"])
    # table.show()
    # +===+===================+=============+==========+=====+
    # | # |        Date       |numeric value|  string  | bool|
    # |row|      datetime     |     int     |   date   | bool|
    # +---+-------------------+-------------+----------+-----+
    # |0  |1920-01-01 00:00:00|            0|1920-01-01| True|
    # |1  |2016-10-31 00:00:00|        42674|2016-10-31|False|
    # +===+===================+=============+==========+=====+
    assert table["string"].types() == {date: len(table)}


def test_filereader_gdocs1csv_no_header():
    path = Path(__file__).parent / "data" / "gdocs1.csv"
    assert path.exists()
    table = Table.from_file(path, first_row_has_headers=False)
    assert list(table.columns) == ["0", "1", "2", "3", "4", "5"]

    table = Table.from_file(path, first_row_has_headers=False, columns=[str(n) for n in [0, 1, 2, 3, 4, 5]])
    # ^--- this uses the import_file shortcut as it has the same config as the previous import.

    table.show(slice(0, 10))
    # +===+===+=============+=============+============+============+============+
    # | # | 0 |      1      |      2      |     3      |     4      |     5      |
    # |row|str|     str     |     str     |    str     |    str     |    str     |
    # +---+---+-------------+-------------+------------+------------+------------+
    # |0  |a  |b            |c            |d           |e           |f           |  <--- strings!
    # |1  |1  |0.06060606061|0.09090909091|0.1212121212|0.1515151515|0.1818181818|
    # |2  |2  |0.1212121212 |0.2424242424 |0.4848484848|0.9696969697|1.939393939 |
    # |3  |3  |0.2424242424 |0.4848484848 |0.9696969697|1.939393939 |3.878787879 |
    # |4  |4  |0.4848484848 |0.9696969697 |1.939393939 |3.878787879 |7.757575758 |
    # |5  |5  |0.9696969697 |1.939393939  |3.878787879 |7.757575758 |15.51515152 |
    # |6  |6  |1.939393939  |3.878787879  |7.757575758 |15.51515152 |31.03030303 |
    # |7  |7  |3.878787879  |7.757575758  |15.51515152 |31.03030303 |62.06060606 |
    # |8  |8  |7.757575758  |15.51515152  |31.03030303 |62.06060606 |124.1212121 |
    # |9  |9  |15.51515152  |31.03030303  |62.06060606 |124.1212121 |248.2424242 |
    # +===+===+=============+=============+============+============+============+
    assert len(table) == 46
    assert table.types() == {str(i): {str: len(table)} for i in range(6)}


def test_filereader_gdocs1xlsx_no_header():
    path = Path(__file__).parent / "data" / "gdocs1.xlsx"
    assert path.exists()
    tables = []
    for sheet in ("Sheet1", "Sheet2", "Sheet3"):
        table = Table.from_file(
            path, sheet=sheet, first_row_has_headers=False, columns=[str(n) for n in [0, 1, 2, 3, 4, 5]]
        )
        table.show(slice(0, 10))
        # +===+===+=============+=============+============+============+============+
        # | # | 0 |      1      |      2      |     3      |     4      |     5      |
        # |row|str|     str     |     str     |    str     |    str     |    str     |
        # +---+---+-------------+-------------+------------+------------+------------+
        # |0  |a  |b            |c            |d           |e           |f           |  <--- strings!
        # |1  |1  |0.06060606061|0.09090909091|0.1212121212|0.1515151515|0.1818181818|
        # |2  |2  |0.1212121212 |0.2424242424 |0.4848484848|0.9696969697|1.939393939 |
        # |3  |3  |0.2424242424 |0.4848484848 |0.9696969697|1.939393939 |3.878787879 |
        # |4  |4  |0.4848484848 |0.9696969697 |1.939393939 |3.878787879 |7.757575758 |
        # |5  |5  |0.9696969697 |1.939393939  |3.878787879 |7.757575758 |15.51515152 |
        # |6  |6  |1.939393939  |3.878787879  |7.757575758 |15.51515152 |31.03030303 |
        # |7  |7  |3.878787879  |7.757575758  |15.51515152 |31.03030303 |62.06060606 |
        # |8  |8  |7.757575758  |15.51515152  |31.03030303 |62.06060606 |124.1212121 |
        # |9  |9  |15.51515152  |31.03030303  |62.06060606 |124.1212121 |248.2424242 |
        # +===+===+=============+=============+============+============+============+

        if sheet == "Sheet1":
            assert len(table) == 46
            assert table.types() == {
                "0": {int: 44, float: 1, str: 1},
                "1": {int: 43, float: 2, str: 1},
                "2": {int: 43, float: 2, str: 1},
                "3": {int: 43, float: 2, str: 1},
                "4": {int: 43, float: 2, str: 1},
                "5": {int: 43, float: 2, str: 1},
            }
        elif sheet == "Sheet2":
            assert len(table) == 46
            assert table.types() == {
                "0": {int: 45, str: 1},
                "1": {int: 11, float: 34, str: 1},
                "2": {int: 12, float: 33, str: 1},
                "3": {int: 13, float: 32, str: 1},
                "4": {int: 14, float: 31, str: 1},
                "5": {int: 15, float: 30, str: 1},
            }
        elif sheet == "Sheet3":
            assert len(table) == 0
            assert table.types() == {}
        # assert table.types() == {str(i): {str: len(table)} for i in range(6)}
        tables.append(table)



def test_keep_some_columns_only():
    path = Path(__file__).parent / "data" / "book1.csv"
    assert path.exists()
    table = Table.from_file(path, columns=["a", "b"])
    assert set(table.columns) == {"a", "b"}
    assert len(table) == 45

def test_misaligned_pages_1():
    path = Path(__file__).parent / "data" / "detect_misalignment.csv"
    assert path.exists()

    expected_table = {
        "a": [0, 2, 0, 0, 6],
        "b": [1, 3, 0, 0, 7],
        "c": [None, 4, 0, 0, 8],
        "d": [None, 5, 0, 0, None],
        "e": ["None", "a", "b", "c", "None"]
    }

    table = Table.from_file(path, columns=list(expected_table.keys()))

    assert set(table.columns) == set(expected_table.keys())
    assert len(table) == 5

    for k, v in expected_table.items():
        assert list(table[k]) == v

def test_misaligned_pages_2():
    path = Path(__file__).parent / "data" / "detect_misalignment.csv"
    assert path.exists()

    expected_table = {
        "a": [0, 2, 0, 0, 6],
        "d": [None, 5, 0, 0, None],
        "e": ["None", "a", "b", "c", "None"]
    }

    table = Table.from_file(path, columns=list(expected_table.keys()))

    assert set(table.columns) == set(expected_table.keys())
    assert len(table) == 5

    for k, v in expected_table.items():
        assert list(table[k]) == v
    

def test_number_locales():
    path = Path(__file__).parent / "data" / "floats.csv"
    assert path.exists()
    table = Table.from_file(path, text_qualifier="\"", columns=[
        "us_floats", 
        "eu_floats",
        "us_thousands",
        "eu_thousands",
        "us_thousands_floats",
        "eu_thousands_floats",
        "us_eu_mixed"
    ])
    assert set(table.columns) == {
        "us_floats", 
        "eu_floats",
        "us_thousands",
        "eu_thousands",
        "us_thousands_floats",
        "eu_thousands_floats",
        "us_eu_mixed"
    }
    assert len(table) == 4
    assert table["us_floats"] == [1.23, 1.23, 1.23, 1.23]
    assert table["eu_floats"] == [1.23, 1.23, 1.23, 1.23]
    assert table["us_thousands"] == [1123456, 1123456, 1123456, 1123456]
    assert table["eu_thousands"] == [1123456, 1123456, 1123456, 1123456]
    assert table["us_thousands_floats"] == [1123456.78, 1123456.78, 1123456.78, 1123456.78]
    assert table["eu_thousands_floats"] == [1123456.78, 1123456.78, 1123456.78, 1123456.78]
    assert table["us_eu_mixed"] == [1123456.78, 1123456.78, 1123456.78, 1123456.78]

def test_split_lines():
    path = Path(__file__).parent / "data" / "split_lines.csv"
    assert path.exists()
    table = Table.from_file(path, text_qualifier="\"", columns=["a", "c"])
    assert set(table.columns) == {"a", "c"}
    assert len(table) == 3
    assert table["a"] == ['aaa\\nbbb', 'ccc\\nddd', 'eee']
    assert table["c"] == [0, 0, 0]



def test_long_texts():
    path = Path(__file__).parent / "data" / "long_text_test.csv"
    assert path.exists()

    columns = [
        "sharepointid",
        "Rank",
        "ITEMID",
        "docname",
        "doctype",
        "application",
        "APPNO",
        "ARTICLES",
        "violation",
        "nonviolation",
        "CONCLUSION",
        "importance",
        "ORIGINATING BODY ID",
        "typedescription",
        "kpdate",
        "kpdateAsText",
        "documentcollectionid",
        "documentcollectionid2",
        "languageisocode",
        "extractedappno",
        "isplaceholder",
        "doctypebranch",
        "RESPONDENT",
        "respondentOrderEng",
        "scl",
        "ECLI",
        "ORIGINATING BODY",
        "YEAR",
        "FULLTEXT",
        "judges",
        "courts",
    ]

    t = Table.from_file(path, text_qualifier='"')
    first_col = list(t.columns)[0]
    last_col = list(t.columns)[-1]
    t[first_col, last_col].show()

    t = Table.from_file(path, columns=columns[:-1], text_qualifier='"')
    selection = tuple(columns[:5])
    t[selection].show()


def test_no_commas():
    table = Table.from_file(Path(__file__).parent / "data" / "no_separator.csv")

    assert len(table) == 25


def test_multi_charset():
    tbl = Table.from_file(Path(__file__).parent / "data" / "formats.csv")

    assert len(tbl) == ENCODING_GUESS_BYTES + 3

def test_header_offset_text():
    tbl = Table.from_file(Path(__file__).parent / "data" / "simple.csv", header_row_index=1)

    assert list(tbl.columns.keys()) == ["header"]
    assert list(tbl["header"]) == [1, 2, 3, 4, 5]

def test_header_offset_xlsx():
    tbl = Table.from_file(Path(__file__).parent / "data" / "simple.xlsx", header_row_index=1, sheet="simple")

    assert list(tbl.columns.keys()) == ["header"]
    assert list(tbl["header"]) == [1, 2, 3, 4, 5]

def test_header_offset_ods():
    tbl = Table.from_file(Path(__file__).parent / "data" / "simple.ods", header_row_index=1, sheet="simple")

    assert list(tbl.columns.keys()) == ["header"]
    assert list(tbl["header"]) == [1, 2, 3, 4, 5]

def test_booleans():
    tbl = Table.from_file(Path(__file__).parent / "data" / "booleans.csv")

    assert len(tbl) == 1
    assert next(tbl.rows) == [False, True, False, True, False, True]