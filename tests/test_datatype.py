from tablite.datatypes import DataTypes, Rank
from datetime import date, time, datetime, timedelta
import numpy as np
import math


def test_np_types():
    for name in dir(np):
        obj = getattr(np, name)
        if hasattr(obj, "dtype"):
            try:
                if "time" in name:
                    npn = obj(0, "D")
                else:
                    npn = obj(0)
                nat = npn.item()
                print("{0} ({1!r}) -> {2}".format(name, npn.dtype.char, type(nat)))
            except:
                pass
    pass  # NOTE: Just use .tolist() to convert 'O' type data to native python.


def test_dt_ranks():
    r = Rank("A", "B", "C")
    r.match("C")
    assert list(r) == ["C", "A", "B"]
    assert [i for i in r] == ["C", "A", "B"]
    r.match("B")
    assert list(r) == ["C", "B", "A"]
    r.match("C")
    assert list(r) == ["C", "B", "A"]
    r.match("B")
    assert list(r) == ["C", "B", "A"]
    r.match("A")
    assert list(r) == ["C", "B", "A"]
    r.match("A")
    assert list(r) == ["C", "B", "A"]
    r.match("A")
    assert list(r) == ["A", "C", "B"]
    for i in range(5):
        r.match("A")
    assert list(r) == ["A", "C", "B"]


def test_datatype_inference():
    # integers
    assert DataTypes.infer(1, int) == 1
    assert DataTypes.infer(0, int) == 0
    assert DataTypes.infer(-1, int) == -1
    assert DataTypes.infer("1", int) == 1
    assert DataTypes.infer("0", int) == 0
    assert DataTypes.infer("-1", int) == -1
    assert (
        DataTypes.infer('"1000028234565432345676542345676543342345675432"', int)
        == 1000028234565432345676542345676543342345675432
    )
    assert DataTypes.infer('"1000028"', int) == 1000028
    assert DataTypes.infer('"1,000,028"', int) == 1000028
    try:
        DataTypes.infer("1.0", int)
        assert False, "1.0 is a float."
    except ValueError:
        assert True

    try:
        DataTypes.infer("1.0", float)
        assert True, "1.0 is a float."
    except ValueError:
        assert False

    # floats
    assert DataTypes.infer("2932,500", float) == 2932.5
    assert DataTypes.infer("2,932.500", float) == 2932.5
    assert DataTypes.infer("2932.500", float) == 2932.5
    assert DataTypes.infer("-2932.500", float) == -2932.5
    assert DataTypes.infer("2.932,500", float) == 2932.5
    assert DataTypes.infer("2.932e5", float) == 2.932e5
    assert DataTypes.infer("-2.932e5", float) == -2.932e5
    assert DataTypes.infer("10e5", float) == 10e5
    assert DataTypes.infer("-10e5", float) == -10e5
    assert DataTypes.infer("-10e-5", float) == -10e-5
    try:
        DataTypes.infer("100126495100211788-1", float)
        assert False, "this is a corrupted string."
    except ValueError:
        assert True

    # booleans
    assert DataTypes.infer("true", bool) is True
    assert DataTypes.infer("True", bool) is True
    assert DataTypes.infer("TRUE", bool) is True
    assert DataTypes.infer("false", bool) is False
    assert DataTypes.infer("False", bool) is False
    assert DataTypes.infer("FALSE", bool) is False

    # strings
    assert DataTypes.infer(7, str) == "7"

    # dates
    isodate = date(1990, 1, 1)
    assert DataTypes.infer(isodate, date) == isodate
    assert DataTypes.infer(isodate.isoformat(), date) == isodate
    assert DataTypes.infer("1990-01-01", date) == date(1990, 1, 1)  # date with minus
    assert DataTypes.infer("2003-09-25", date) == date(2003, 9, 25)

    assert DataTypes.infer("25-09-2003", date) == date(2003, 9, 25)  # year last.
    assert DataTypes.infer("10-09-2003", date) == date(2003, 9, 10)

    assert DataTypes.infer("1990.01.01", date) == date(1990, 1, 1)  # date with dot.
    assert DataTypes.infer("2003.09.25", date) == date(2003, 9, 25)
    assert DataTypes.infer("25.09.2003", date) == date(2003, 9, 25)
    assert DataTypes.infer("10.09.2003", date) == date(2003, 9, 10)

    assert DataTypes.infer("1990/01/01", date) == date(1990, 1, 1)  # date with slash
    assert DataTypes.infer("2003/09/25", date) == date(2003, 9, 25)
    assert DataTypes.infer("25/09/2003", date) == date(2003, 9, 25)
    assert DataTypes.infer("10/09/2003", date) == date(2003, 9, 10)

    assert DataTypes.infer("1990 01 01", date) == date(1990, 1, 1)  # date with space
    assert DataTypes.infer("2003 09 25", date) == date(2003, 9, 25)
    assert DataTypes.infer("25 09 2003", date) == date(2003, 9, 25)
    assert DataTypes.infer("10 09 2003", date) == date(2003, 9, 10)

    assert DataTypes.infer("20030925", date) == date(2003, 9, 25)  # "iso stripped format strip"
    assert DataTypes.infer("19760704", date) == date(1976, 7, 4)  # "random format"),

    assert DataTypes.infer("7 4 1976", date) == date(1976, 4, 7)
    # assert DataTypes.infer("14 jul 1976", date) == date(1976, 7, 14)
    # assert DataTypes.infer("4 Jul 1976", date) == date(1976, 7, 4)

    # NOT HANDLED - ambiguous formats due to lack of 4 digits for year.
    # ("10 09 03", date(2003, 10, 9), "date with space"),
    # ("25 09 03", date(2003, 9, 25), "date with space"),
    # ("03 25 Sep", date(2003, 9, 25), "strangely ordered date"),
    # ("25 03 Sep", date(2025, 9, 3), "strangely ordered date"),
    # "10-09-03", date(2003, 10, 9),
    # ("10.09.03", date(2003, 10, 9), "date with dot"),
    # ("10/09/03", date(2003, 10, 9), "date with slash"),

    # NOT HANDLED - MDY formats are US locale.
    # assert DataTypes.infer("09-25-2003", date) == date(2003, 9, 25)
    # assert DataTypes.infer("09.25.2003", date) == date(2003, 9, 25)
    # assert DataTypes.infer("09/25/2003", date) == date(2003, 9, 25)
    # assert DataTypes.infer("09 25 2003", date) == date(2003, 9, 25)
    # assert DataTypes.infer('13NOV2017', date) == date(2017, 11, 13) # GH360

    # times
    isotime = time(23, 12, 11)
    assert DataTypes.infer(isotime, time) == isotime
    assert DataTypes.infer(isotime.isoformat(), time) == time(23, 12, 11)
    assert DataTypes.infer("23:12:11", time) == time(23, 12, 11)
    assert DataTypes.infer("23:12:11.123456", time) == time(23, 12, 11, 123456)

    # datetimes
    isodatetime = datetime.now()
    assert DataTypes.infer(isodatetime, datetime) == isodatetime
    assert DataTypes.infer(isodatetime.isoformat(), datetime) == isodatetime
    dirty_date = datetime(1990, 1, 1, 23, 12, 11, int(0.003 * 10**6))
    assert DataTypes.infer("1990-01-01T23:12:11.003000", datetime) == dirty_date  # iso minus T microsecond
    assert DataTypes.infer("1990-01-01T23:12:11.003", datetime) == dirty_date  #
    assert DataTypes.infer("1990-01-01 23:12:11.003", datetime) == dirty_date  # iso space
    assert DataTypes.infer("1990/01/01T23:12:11.003", datetime) == dirty_date  # iso slash T
    assert DataTypes.infer("1990/01/01 23:12:11.003", datetime) == dirty_date  # iso slash
    assert DataTypes.infer("1990.01.01T23:12:11.003", datetime) == dirty_date  # iso dot T
    assert DataTypes.infer("1990.01.01 23:12:11.003", datetime) == dirty_date  # iso dot
    assert DataTypes.infer("10/04/2007 00:00", datetime) == datetime(2007, 4, 10, 0, 0)
    assert DataTypes.infer("1990 01 01T23:12:11.003", datetime) == dirty_date  # iso space T
    assert DataTypes.infer("1990 01 01 23:12:11.003", datetime) == dirty_date  # iso space

    assert DataTypes.infer("2003-09-25T10:49:41", datetime) == datetime(
        2003, 9, 25, 10, 49, 41
    )  # iso minus T fields omitted.
    assert DataTypes.infer("2003-09-25T10:49", datetime) == datetime(2003, 9, 25, 10, 49)
    assert DataTypes.infer("2003-09-25T10", datetime) == datetime(2003, 9, 25, 10)

    assert DataTypes.infer("20080227T21:26:01.123456789", datetime) == datetime(
        2008, 2, 27, 21, 26, 1, 123456
    )  # high precision seconds
    assert DataTypes.infer("20030925T104941", datetime) == datetime(
        2003, 9, 25, 10, 49, 41
    )  # iso nospace T fields omitted.
    assert DataTypes.infer("20030925T1049", datetime) == datetime(2003, 9, 25, 10, 49, 0)
    assert DataTypes.infer("20030925T10", datetime) == datetime(2003, 9, 25, 10)

    assert DataTypes.infer("199709020908", datetime) == datetime(1997, 9, 2, 9, 8)
    assert DataTypes.infer("19970902090807", datetime) == datetime(1997, 9, 2, 9, 8, 7)
    assert DataTypes.infer("2003-09-25 10:49:41,502", datetime) == datetime(
        2003, 9, 25, 10, 49, 41, 502000
    )  # python logger format
    assert DataTypes.infer("0099-01-01T00:00:00", datetime) == datetime(99, 1, 1, 0, 0)  # 99 ad
    assert DataTypes.infer("0031-01-01T00:00:00", datetime) == datetime(31, 1, 1, 0, 0)  # 31 ad

    # NOT HANDLED. ambiguous format. Year is not 4 digits.
    # ("950404 122212", datetime(1995, 4, 4, 12, 22, 12), "random format"),
    # ("04.04.95 00:22", datetime(1995, 4, 4, 0, 22), "random format"),

    # NOT HANDLED. Month and Day names are locale dependent.
    # ("Thu Sep 25 10:36:28 2003", datetime(2003, 9, 25, 10, 36, 28), "date command format strip"),
    # ("Thu Sep 25 2003", datetime(2003, 9, 25), "date command format strip"),
    # ("  July   4 ,  1976   12:01:02   am  ", datetime(1976, 7, 4, 0, 1, 2), "extra space"),
    # ("Wed, July 10, '96", datetime(1996, 7, 10, 0, 0), "random format"),
    # ("1996.July.10 AD 12:08 PM", datetime(1996, 7, 10, 12, 8), "random format"),
    # ("7-4-76", datetime(1976, 7, 4), "random format"),
    # ("0:01:02 on July 4, 1976", datetime(1976, 7, 4, 0, 1, 2), "random format"),
    # ("July 4, 1976 12:01:02 am", datetime(1976, 7, 4, 0, 1, 2), "random format"),
    # ("Mon Jan  2 04:24:27 1995", datetime(1995, 1, 2, 4, 24, 27), "random format"),
    # ("Jan 1 1999 11:23:34.578", datetime(1999, 1, 1, 11, 23, 34, 578000), "random format"),


def test_round():
    xround = DataTypes.round
    # round up
    assert xround(0, 1, True) == 0
    assert xround(1.6, 1, True) == 2
    assert xround(1.4, 1, True) == 2
    # round down
    assert xround(0, 1, False) == 0
    assert xround(1.6, 1, False) == 1
    assert xround(1.4, 1, False) == 1
    # round half
    assert xround(0, 1) == 0
    assert xround(1.6, 1) == 2
    assert xround(1.4, 1) == 1

    # round half
    assert xround(16, 10) == 20
    assert xround(14, 10) == 10

    # round half
    assert xround(-16, 10) == -20
    assert xround(-14, 10) == -10

    # round to odd multiples
    assert xround(6, 3.1415, 1) == 2 * 3.1415

    assert xround(1.2345, 0.001, True) == 1.2349999999999999 and math.isclose(1.2349999999999999, 1.235)
    assert xround(1.2345, 0.001, False) == 1.234

    assert xround(123, 100, False) == 100
    assert xround(123, 100, True) == 200

    assert xround(123, 5.07, False) == 24 * 5.07

    dt = datetime(2022, 8, 18, 11, 14, 53, 440)

    td = timedelta(hours=0.5)
    assert xround(dt, td, up=False) == datetime(2022, 8, 18, 11, 0)
    assert xround(dt, td, up=None) == datetime(2022, 8, 18, 11, 0)
    assert xround(dt, td, up=True) == datetime(2022, 8, 18, 11, 30)

    td = timedelta(hours=24)
    assert xround(dt, td, up=False) == datetime(2022, 8, 18)
    assert xround(dt, td, up=None) == datetime(2022, 8, 18)
    assert xround(dt, td, up=True) == datetime(2022, 8, 19)

    td = timedelta(days=0.5)
    assert xround(dt, td, up=False) == datetime(2022, 8, 18)
    assert xround(dt, td, up=None) == datetime(2022, 8, 18, 12)
    assert xround(dt, td, up=True) == datetime(2022, 8, 18, 12)

    td = timedelta(days=1.5)
    assert xround(dt, td, up=False) == datetime(2022, 8, 18)
    assert xround(dt, td, up=None) == datetime(2022, 8, 18)
    assert xround(dt, td, up=True) == datetime(2022, 8, 19, 12)

    td = timedelta(seconds=0.5)
    assert xround(dt, td, up=False) == datetime(2022, 8, 18, 11, 14, 53, 0)
    assert xround(dt, td, up=None) == datetime(2022, 8, 18, 11, 14, 53, 0)
    assert xround(dt, td, up=True) == datetime(2022, 8, 18, 11, 14, 53, 500000)

    td = timedelta(seconds=40000)
    assert xround(dt, td, up=False) == datetime(2022, 8, 18, 6, 40)
    assert xround(dt, td, up=None) == datetime(2022, 8, 18, 6, 40)
    assert xround(dt, td, up=True) == datetime(2022, 8, 18, 17, 46, 40)
