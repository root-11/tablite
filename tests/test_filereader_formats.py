from tablite import Table
from tablite.file_reader_utils import TextEscape
from tablite.datatypes import DataTypes
from time import process_time_ns
from datetime import date, datetime
from pathlib import Path
import pytest


@pytest.fixture(autouse=True) # this resets the HDF5 file for every test.
def refresh():
    Table.reset_storage()
    yield


def test_text_escape():
    text_escape = TextEscape(delimiter=';',openings=None,closures=None)

    te = text_escape('"t"')
    assert te ==["t"]

    te = text_escape('"t";"3";"2"')
    assert te == ["t", "3", "2"]

    te = text_escape('"this";"123";234;"this";123;"234"')
    assert te == ['this', '123', '234', 'this', '123', '234']

    te = text_escape('"this";"123";"234"')
    assert te == ["this", "123", "234"]

    te = text_escape('"this";123;234')
    assert te == ["this", "123", "234"]

    te = text_escape('"this";123;"234"')
    assert te == ["this", "123", "234"]

    te = text_escape('123;"1\'3";234')
    assert te == ["123", "1'3", "234"], te

    te = text_escape('"1000627";"MOC;SEaert;pás;krk;XL;černá";"2.180,000";"CM3";2')
    assert te == ["1000627", "MOC;SEaert;pás;krk;XL;černá", "2.180,000", "CM3", '2']

    te = text_escape('"1000294";"S2417DG 24"" LED monitor (210-AJWM)";"47.120,000";"CM3";3')
    assert te == ['1000294', 'S2417DG 24"" LED monitor (210-AJWM)', '47.120,000', 'CM3', '3']


def test_filereader_123csv():
    csv_file = Path(__file__).parent / "data" / "123.csv"

    table7 = Table()
    # table7.metadata['filename'] = '123.csv'
    table7.add_column('A', data=[1, None, 8, 3, 4, 6, 5, 7, 9])
    table7.add_column('B', data=[10, 100, 1, 1, 1, 1, 10, 10, 10])
    table7.add_column('C', data=[0, 1, 0, 1, 0, 1, 0, 1, 0])
    sort_order = {'B': False, 'C': False, 'A': False}
    table7 = table7.sort(**sort_order)

    headers = ",".join([c for c in table7.columns])
    data = [headers]
    for row in table7.rows:
        data.append(",".join(str(v) for v in row))

    s = "\n".join(data)
    print(s)
    csv_file.write_text(s)  # write
    tr_table = Table.import_file(csv_file, import_as='csv', columns={v:'?' for v in 'ABC'})
    csv_file.unlink()  # cleanup

    tr_table.show()
    for c in tr_table.columns:
        col = tr_table[c]
        col[:] = DataTypes.guess(col)
    
    tr_table.show()

    assert tr_table == table7


def test_filereader_csv_f12():
    path = Path(__file__).parent / "data" / 'f12.csv'
    columns = ['Prod Slbl', 'Prod Tkt Descp Txt', 'Case Qty', 'Height', 'Width', 'Length', 'Weight', 'sale_date', 'cust_nbr', 'Case Qty_1', 'EA Location', 'CDY/Cs', 'EA/Cs', 'EA/CDY', 'Ordered As', 'Picked As', 'Cs/Pal', 'SKU', 'Order_Number', 'cases']
    data = Table.import_file(path, import_as='csv', columns={n:'f' for n in columns})
    assert len(data) == 13
    for name in data.columns:
        data[name] = DataTypes.guess(data[name])
    assert list(data.rows) == [
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 1365660, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '1365660_2012/01/01', 1],
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 1696753, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '1696753_2012/01/01', 1],
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 1828693, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '1828693_2012/01/01', 1],
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 2211182, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '2211182_2012/01/01', 2],
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 2229312, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '2229312_2012/01/01', 1],
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 2414206, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '2414206_2012/01/01', 1],
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 1, 0, 0), 266791, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '266791_2012/01/01', 2],
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 2, 0, 0), 1017988, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '1017988_2012/01/02', 1],
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 2, 0, 0), 1020158, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '1020158_2012/01/02', 2],
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 2, 0, 0), 1032132, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '1032132_2012/01/02', 1],
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 2, 0, 0), 1048323, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '1048323_2012/01/02', 1],
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 2, 0, 0), 1056865, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '1056865_2012/01/02', 2],
        [52609, '3.99 BTSZ RDS TO', 7, 16, 16, 22, 20, datetime(2012, 1, 2, 0, 0), 1057577, 7, 'EA', 0, 7, 0, 'Each', 'Each', 72, 587, '1057577_2012/01/02', 0]
    ], list(data.rows)


def test_filereader_book1csv():
    path = Path(__file__).parent / "data" / 'book1.csv'
    assert path.exists()
    table = Table.import_file(path, import_as='csv', columns={n:'f' for n in ['a', 'b', 'c', 'd', 'e', 'f']})
    table.show(slice(0, 10))
    for name in table.columns:
        table[name] = DataTypes.guess(table[name])

    assert table['a'].types() == {int:45}
    for name in list('bcdef'):
        assert table[name].types() == {float:45}

    assert len(table) == 45


def test_filereader_book1tsv():
    path = Path(__file__).parent / "data" / 'book1.tsv'
    assert path.exists()
    table = Table.import_file(path, import_as='csv', columns={n:'f' for n in ['a', 'b', 'c', 'd', 'e', 'f']}, delimiter='\t', text_qualifier=None)
    table.show(slice(0, 10))
    assert len(table) == 45


def test_filereader_gdocs1csv():
    path = Path(__file__).parent / "data" / 'gdocs1.csv'
    assert path.exists()
    table = Table.import_file(path, import_as='csv', columns={n:'f' for n in ['a', 'b', 'c', 'd', 'e', 'f']}, text_qualifier=None)
    table.show(slice(0, 10))
    assert len(table) == 45


def test_filereader_book1txt():
    path = Path(__file__).parent / "data" / 'book1.txt'
    assert path.exists()
    table = Table.import_file(path, import_as='csv', columns={n:'f' for n in ['a', 'b', 'c', 'd', 'e', 'f']}, delimiter='\t', text_qualifier=None)
    table.show(slice(0, 10))
    assert len(table) == 45


def test_filereader_book1_txt_chunks():
    path = Path(__file__).parent / "data" / 'book1.txt'
    assert path.exists()
    table1 = Table.import_file(path, import_as='csv', columns={n:'f' for n in ['a', 'b', 'c', 'd', 'e', 'f']}, delimiter='\t', text_qualifier=None)
    start = 0
    table2 = None
    while True:
        tmp = Table.import_file(path, import_as='csv', columns={n:'f' for n in ['a', 'b', 'c', 'd', 'e', 'f']}, delimiter='\t', text_qualifier=None, start=start, limit=5)
        if len(tmp)==0:
            break
        start += len(tmp) 

        if table2 is None:
            table2 = tmp
        else:
            table2 += tmp
        
    assert table1 == table2
        

def test_filereader_book1_txt_chunks_and_offset():
    path = Path(__file__).parent / "data" / 'book1.txt'
    assert path.exists()

    start = 2

    table1 = Table.import_file(path, import_as='csv', columns={n:'f' for n in ['a', 'b', 'c', 'd', 'e', 'f']}, delimiter='\t', text_qualifier=None,start=start)
    
    table2 = None
    while True:
        tmp = Table.import_file(path, import_as='csv', columns={n:'f' for n in ['a', 'b', 'c', 'd', 'e', 'f']}, delimiter='\t', text_qualifier=None, start=start, limit=5)
        if len(tmp)==0:
            break
        start += len(tmp) 
        if table2 is None:
            table2 = tmp
        else:
            table2 += tmp
    
    assert table1 == table2


def test_filereader_gdocsc1tsv():
    path = Path(__file__).parent / "data" / 'gdocs1.tsv'
    assert path.exists()
    table = Table.import_file(path, import_as='csv', columns={n:'f' for n in ['a', 'b', 'c', 'd', 'e', 'f']}, text_qualifier=None, delimiter='\t')
    table.show(slice(0, 10))
    assert len(table) == 45
    for name in table.columns:
        table[name] = DataTypes.guess(table[name])
    
    assert table['a'].types() == {int:45}
    for name in list('bcdef'):
        assert table[name].types() == {float:45}


def test_filereader_gdocsc1ods():
    path = Path(__file__).parent / "data" / 'gdocs1.ods'
    assert path.exists()

    sheet1 = Table.import_file(path, import_as='ods', sheet='Sheet1')
    for name in sheet1.columns:
        sheet1[name] = DataTypes.guess(sheet1[name])
        assert sheet1[name].types() == {int:45}

    sheet2 = Table.import_file(path, import_as='ods', sheet='Sheet2')
    for name in sheet2.columns:
        sheet2[name] = DataTypes.guess(sheet2[name])
        if name == 'a':
            assert sheet2[name].types() == {int:45}
        else:
            assert sheet2[name].types() == {float:45}


def test_filereader_gdocs1xlsx():
    path = Path(__file__).parent / "data" / 'gdocs1.xlsx'
    assert path.exists()
    sheet1 = Table.import_file(path, import_as='xlsx', sheet='Sheet1', columns={k:'f' for k in ['a', 'b', 'c', 'd', 'e', 'f']})
    sheet1.show(slice(0, 10))

    for name in sheet1.columns:
        sheet1[name] = DataTypes.guess(sheet1[name])
        assert sheet1[name].types() == {int:45}
    assert len(sheet1) == 45


def test_filereader_utf8csv():
    path = Path(__file__).parent / "data" / 'utf8_test.csv'
    assert path.exists()

    columns = ["Item","Materiál","Objem","Jednotka objemu","Free Inv Pcs"]
    table = Table.import_file(path, import_as='csv', delimiter=';', columns={k:'f' for k in columns}, text_qualifier='"')
    table.show(slice(0, 10))
    table.show(slice(-15,None))

    types = {
        'Item': int,
        'Materiál': str,
        'Objem': float,
        'Jednotka objemu': str,
        'Free Inv Pcs': int
    }

    for name in table.columns:
        table[name] = DataTypes.guess(table[name])
        assert table[name].types() == {types[name]: 99}

    assert len(table) == 99, len(table)


def test_filereader_utf16csv():
    path = Path(__file__).parent / "data" / 'utf16_test.csv'
    assert path.exists()
    col_names = ['"Item"', '"Materiál"', '"Objem"', '"Jednotka objemu"', '"Free Inv Pcs"']
    table = Table.import_file(path, import_as='csv', delimiter=';', columns={k:'f' for k in col_names})
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
    path = Path(__file__).parent / "data" / 'win1250_test.csv'
    assert path.exists()
    col_names = ['"Item"', '"Materiál"', '"Objem"', '"Jednotka objemu"', '"Free Inv Pcs"']
    table = Table.import_file(path, import_as='csv', delimiter=';', columns={k:'f' for k in col_names})
    table.show(slice(0, 10))
    table.show(slice(None, -15))
    assert len(table) == 99, len(table)


def test_filereader_utf8sig_encoding_csv():
    path = Path(__file__).parent / "data" / 'utf8sig.csv'
    assert path.exists()
    col_names = ['432', '1']
    table = Table.import_file(path, import_as='csv', delimiter=',', columns={k:'f' for k in col_names})
    table.show(slice(0, 10))
    table.show(slice(-15))
    assert len(table) == 2, len(table)


def test_filereader_saptxt():
    path = Path(__file__).parent / "data" / 'sap.txt'
    assert path.exists()
    
    header = "    | Delivery |  Item|Pl.GI date|Route |SC|Ship-to   |SOrg.|Delivery quantity|SU| TO Number|Material    |Dest.act.qty.|BUn|Typ|Source Bin|Cty"
    col_names = [w.strip(" ").rstrip(" ") for w in header.split("|")]

    table = Table.import_file(path, delimiter="|", import_as='txt', columns={k:'f' for k in col_names if k!=""}, strip_leading_and_tailing_whitespace=True)
    
    for name in table.columns:
        table[name] = DataTypes.guess(table[name])
    
    table.show()

    assert len(table) == 20, len(table)


def test_filereader_book1xlsx():
    path = Path(__file__).parent / "data" / 'book1.xlsx'
    assert path.exists()
    start = process_time_ns()
    sheet1 = Table.import_file(path, import_as='xls', sheet='Sheet1', columns={k:'f' for k in ['a', 'b', 'c', 'd', 'e', 'f']})
    sheet2 = Table.import_file(path, import_as='xls', columns={k:'f' for k in ['a', 'b', 'c', 'd', 'e', 'f']}, sheet='Sheet2 ')  # there's a deliberate white space at the end!)
    end = process_time_ns()

    tables = [sheet1,sheet2]
    fields = sum(len(t) * len(t.columns) for t in tables)
    print("{:,} fields/seccond".format(round(1e9 * fields / max(1, end - start), 0)))

    for name in 'abcdef':
        sheet1[name] = DataTypes.guess(sheet1[name])
        assert sheet1[name].types() == {int: len(sheet1)}

    sheet2['a'] == DataTypes.guess(sheet2['a'])
    assert sheet2['a'].types() == {int: len(sheet2)}
    
    for name in list('bcdef'):
        sheet2[name] = DataTypes.guess(sheet2[name])
        assert sheet2[name].types() == {float: len(sheet2)}


def test_filereader_exceldatesxlsx():
    path = Path(__file__).parent / "data" / 'excel_dates.xlsx'
    assert path.exists()
    table = Table.import_file(path, import_as='xls', sheet='Sheet1', columns={k:'f' for k in ['Date', 'numeric value', 'string', 'bool']})
    table.show()
    # +===+===================+=============+==========+=====+
    # | # |        Date       |numeric value|  string  | bool|
    # |row|      datetime     |     int     |   str    | bool|
    # +---+-------------------+-------------+----------+-----+
    # |0  |1920-01-01 00:00:00|            0|1920/01/01| True|
    # |1  |2016-10-31 00:00:00|        42674|2016/10/31|False|
    # +===+===================+=============+==========+=====+
    assert len(table) == 2, len(table)
    assert table['Date'].types() == {datetime: len(table)}
    assert table['numeric value'].types() == {int: len(table)}
    assert table['bool'].types() == {bool: len(table)}

    assert table['string'].types() == {str: len(table)}
    table['string'] = DataTypes.guess(table['string'])
    # table.show()
    # +===+===================+=============+==========+=====+
    # | # |        Date       |numeric value|  string  | bool|
    # |row|      datetime     |     int     |   date   | bool|
    # +---+-------------------+-------------+----------+-----+
    # |0  |1920-01-01 00:00:00|            0|1920-01-01| True|
    # |1  |2016-10-31 00:00:00|        42674|2016-10-31|False|
    # +===+===================+=============+==========+=====+
    assert table['string'].types() == {date:len(table)}


def test_filereader_gdocs1csv_no_header():
    path = Path(__file__).parent / "data" / 'gdocs1.csv'
    assert path.exists()
    try:
        table = Table.import_file(path, import_as='csv', first_row_has_headers=False)
        assert False, "headers missing"
    except ValueError:
        assert True
    table = Table.import_file(path, import_as='csv', first_row_has_headers=False, columns={str(n):'f' for n in [0, 1, 2, 3, 4, 5]})
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
    path = Path(__file__).parent / "data" / 'gdocs1.xlsx'
    assert path.exists()
    tables = []
    for sheet in ('Sheet1', 'Sheet2', 'Sheet3'):

        table = Table.import_file(path, import_as='xls', sheet=sheet, first_row_has_headers=False, columns={str(n):'f' for n in [0, 1, 2, 3, 4, 5]})
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
                '0': {int: 44, float: 1, str: 1}, 
                '1': {int: 43, float: 2, str: 1}, 
                '2': {int: 43, float: 2, str: 1}, 
                '3': {int: 43, float: 2, str: 1}, 
                '4': {int: 43, float: 2, str: 1}, 
                '5': {int: 43, float: 2, str: 1}
                }
        elif sheet == "Sheet2":
            assert len(table) == 46
            assert table.types() == {
                '0': {int: 45, str: 1}, 
                '1': {int: 11, float: 34, str: 1}, 
                '2': {int: 12, float: 33, str: 1}, 
                '3': {int: 13, float: 32, str: 1}, 
                '4': {int: 14, float: 31, str: 1},
                '5': {int: 15, float: 30, str: 1}
                }
        elif sheet == "Sheet3":
            assert len(table) == 0
            assert table.types() == {}
        # assert table.types() == {str(i): {str: len(table)} for i in range(6)}
        tables.append(table)


def test_keep():
    path = Path(__file__).parent / "data" / 'book1.csv'
    assert path.exists()
    table = Table.import_file(path, columns={'a':'f','b':'f'}, import_as='csv')
    assert set(table.columns) == {'a', 'b'}
    assert len(table) == 45


