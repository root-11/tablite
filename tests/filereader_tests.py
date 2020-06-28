from table import *
from time import process_time_ns


def text_escape_tests():
    te = text_escape('"t"')
    assert te == ("t",)

    te = text_escape('"t";"3";"2"')
    assert te == ("t", "3", "2")

    te = text_escape('"this";"123";234;"this";123;"234"')
    assert te == ('this', '123', '234', 'this', '123', '234')

    te = text_escape('"this";"123";"234"')
    assert te == ("this", "123", "234")

    te = text_escape('"this";123;234')
    assert te == ("this", "123", "234")

    te = text_escape('"this";123;"234"')
    assert te == ("this", "123", "234")

    te = text_escape('123;"1\'3";234')
    assert te == ("123", "1'3", "234"), te

    te = text_escape('"1000627";"MOC;SEaert;pás;krk;XL;černá";"2.180,000";"CM3";2')
    assert te == ("1000627", "MOC;SEaert;pás;krk;XL;černá", "2.180,000", "CM3", '2')

    te = text_escape('"1000294";"S2417DG 24"" LED monitor (210-AJWM)";"47.120,000";"CM3";3')
    assert te == ('1000294', 'S2417DG 24"" LED monitor (210-AJWM)', '47.120,000', 'CM3', '3')


def test_filereader_123csv():
    csv_file = Path(__file__).parent / "data" / "123.csv"

    table7 = Table(filename=csv_file.name)
    table7.metadata['filename'] = '123.csv'
    table7.add_column('A', int, data=[1, None, 8, 3, 4, 6, 5, 7, 9], allow_empty=True)
    table7.add_column('B', int, data=[10, 100, 1, 1, 1, 1, 10, 10, 10])
    table7.add_column('C', int, data=[0, 1, 0, 1, 0, 1, 0, 1, 0])
    sort_order = {'B': False, 'C': False, 'A': False}
    table7.sort(**sort_order)

    headers = ", ".join([c for c in table7.columns])
    data = [headers]
    for row in table7.rows:
        data.append(", ".join(str(v) for v in row))

    s = "\n".join(data)
    print(s)
    csv_file.write_text(s)  # write
    tr_table = file_reader(csv_file)[0]  # read
    csv_file.unlink()  # cleanup

    tr_table.show()
    find_format(tr_table)

    assert tr_table == table7


def test_filereader_book1csv():
    path = Path(__file__).parent / "data" / 'book1.csv'
    assert path.exists()
    table = file_reader(path)[0]
    table.show(slice(0, 10))

    book1_csv = Table(filename=path.name)
    book1_csv.add_column('a', int)
    for float_type in list('bcdef'):
        book1_csv.add_column(float_type, float)

    assert table.compare(book1_csv), table.compare(book1_csv)
    assert len(table) == 45


def test_filereader_gdocs1csv():
    path = Path(__file__).parent / "data" / 'gdocs1.csv'
    assert path.exists()
    table = file_reader(path)[0]
    table.show(slice(0, 10))

    book1_csv = Table(filename=path.name)
    book1_csv.add_column('a', int)
    for float_type in list('bcdef'):
        book1_csv.add_column(float_type, float)

    assert table.compare(book1_csv), table.compare(book1_csv)
    assert len(table) == 45


def test_filereader_book1txt():
    path = Path(__file__).parent / "data" / 'book1.txt'
    assert path.exists()
    table = file_reader(path)[0]
    table.show(slice(0, 10))

    book1_csv = Table(filename=path.name)
    book1_csv.add_column('a', int)
    for float_type in list('bcdef'):
        book1_csv.add_column(float_type, float)

    assert table.compare(book1_csv), table.compare(book1_csv)
    assert len(table) == 45


def test_filereader_gdocsc1tsv():
    path = Path(__file__).parent / "data" / 'gdocs1.tsv'
    assert path.exists()
    table = file_reader(path)[0]
    table.show(slice(0, 10))

    book1_csv = Table(filename=path.name)
    book1_csv.add_column('a', int)
    for float_type in list('bcdef'):
        book1_csv.add_column(float_type, float)

    assert table.compare(book1_csv), table.compare(book1_csv)
    assert len(table) == 45


def test_filereader_gdocsc1ods():
    path = Path(__file__).parent / "data" / 'gdocs1.ods'
    assert path.exists()
    tables = file_reader(path)

    sheet1 = Table(filename=path.name, sheet_name='Sheet1')
    for int_type in list('abcdef'):
        sheet1.add_column(int_type, int)

    sheet2 = Table(filename=path.name, sheet_name='Sheet2')
    sheet2.add_column('a', int)
    for float_type in list('bcdef'):
        sheet2.add_column(float_type, float)

    sheets = [sheet1, sheet2]

    for sheet, table in zip(sheets, tables):
        table.compare(sheet)
        assert len(table) == 45


def test_filereader_gdocs1xlsx():
    path = Path(__file__).parent / "data" / 'gdocs1.xlsx'
    assert path.exists()
    table = file_reader(path)[0]
    table.show(slice(0, 10))

    gdocs1xlsx = Table(filename=path.name, sheet_name='Sheet1')
    for float_type in list('abcdef'):
        gdocs1xlsx.add_column(float_type, int)

    assert table.compare(gdocs1xlsx), table.compare(gdocs1xlsx)
    assert len(table) == 45



def test_filereader_utf8csv():
    path = Path(__file__).parent / "data" / 'utf8_test.csv'
    assert path.exists()
    table = file_reader(path, sep=';')[0]
    table.show(slice(0, 10))
    table.show(slice(-15))

    book1_csv = Table(filename=path.name)
    book1_csv.add_column('Item', int)
    book1_csv.add_column('Materiál', str)
    book1_csv.add_column('Objem', float)
    book1_csv.add_column('Jednotka objemu', str)
    book1_csv.add_column('Free Inv Pcs', int)

    assert table.compare(book1_csv), table.compare(book1_csv)
    assert len(table) == 99, len(table)


def test_filereader_utf16csv():
    path = Path(__file__).parent / "data" / 'utf16_test.csv'
    assert path.exists()
    table = file_reader(path, sep=';')[0]
    table.show(slice(0, 10))
    table.show(slice(-15))

    book1_csv = Table(filename=path.name)
    book1_csv.add_column('Item', int)
    book1_csv.add_column('Materiál', str)
    book1_csv.add_column('Objem', float)
    book1_csv.add_column('Jednotka objemu', str)
    book1_csv.add_column('Free Inv Pcs', int)

    assert table.compare(book1_csv), table.compare(book1_csv)
    assert len(table) == 99, len(table)


def test_filereader_win1251_encoding_csv():
    path = Path(__file__).parent / "data" / 'win1250_test.csv'
    assert path.exists()
    table = file_reader(path, sep=';')[0]
    table.show(slice(0, 10))
    table.show(slice(-15))

    book1_csv = Table(filename=path.name)
    book1_csv.add_column('Item', int)
    book1_csv.add_column('Materiál', str)
    book1_csv.add_column('Objem', float)
    book1_csv.add_column('Jednotka objemu', str)
    book1_csv.add_column('Free Inv Pcs', int)
    assert table.compare(book1_csv), table.compare(book1_csv)
    assert len(table) == 99, len(table)


def test_filereader_saptxt():
    path = Path(__file__).parent / "data" / 'sap.txt'
    assert path.exists()
    # test part 1: split using user defined sequence.
    header = "    | Delivery |  Item|Pl.GI date|Route |SC|Ship-to   |SOrg.|Delivery quantity|SU| TO Number|Material    |Dest.act.qty.|BUn|Typ|Source Bin|Cty"
    split_sequence = ["|"] * header.count('|')
    table = file_reader(path, split_sequence=split_sequence)[0]
    table.show(slice(5))

    sap_sample = Table(filename=path.name)
    sap_sample.add_column('None', str, True)
    sap_sample.add_column('Delivery', int, False)
    sap_sample.add_column('Item', int, False)
    sap_sample.add_column('Pl.GI date', date, False)
    sap_sample.add_column('Route', str, False)
    sap_sample.add_column('SC', str, False)
    sap_sample.add_column('Ship-to', str, False)
    sap_sample.add_column('SOrg.', str, False)
    sap_sample.add_column('Delivery quantity', int, False)
    sap_sample.add_column('SU', str, False)
    sap_sample.add_column('TO Number', int, False)
    sap_sample.add_column('Material', str, False)
    sap_sample.add_column('Dest.act.qty.', int, False)
    sap_sample.add_column('BUn', str, False)
    sap_sample.add_column('Typ', str, False)
    sap_sample.add_column('Source Bin', str, False)
    sap_sample.add_column('Cty|', str, False)

    assert table.compare(sap_sample)
    assert len(table) == 20, len(table)


def test_filereader_book1xlsx():
    path = Path(__file__).parent / "data" / 'book1.xlsx'
    assert path.exists()
    start = process_time_ns()
    tables = file_reader(path)
    end = process_time_ns()

    fields = sum(len(t) * len(t.columns) for t in tables)
    print("{:,} fields/seccond".format(round(1e9 * fields / max(1, end - start), 0)))

    sheet1 = Table(filename=path.name, sheet_name='Sheet1')
    for column_name in list('abcdef'):
        sheet1.add_column(column_name, int, False)

    sheet2 = Table(filename=path.name, sheet_name='Sheet2 ')  # there's a deliberate white space at the end!
    sheet2.add_column('a', int, False)
    for column_name in list('bcdef'):
        sheet2.add_column(column_name, float, False)

    books = [sheet1, sheet2]

    for book, table in zip(books, tables):
        table.show(slice(5))
        assert table.compare(book)
        assert len(table) == 45, len(table)


def test_filereader_exceldatesxlsx():
    path = Path(__file__).parent / "data" / 'excel_dates.xlsx'
    assert path.exists()
    table = file_reader(path)[0]
    table.show()

    sheet1 = Table(filename=path.name, sheet_name='Sheet1')
    sheet1.add_column('Date', date, False)
    sheet1.add_column('numeric value', int, False)
    sheet1.add_column('string', date, False)
    sheet1.add_column('bool', bool, False)

    assert table.compare(sheet1)
    assert len(table) == 2, len(table)


