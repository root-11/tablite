from table import *
from datetime import date, time, datetime


def test_datatype_inference():
    # integers
    assert DataTypes.infer(1, int) == 1
    assert DataTypes.infer(0, int) == 0
    assert DataTypes.infer(-1, int) == -1
    assert DataTypes.infer('1', int) == 1
    assert DataTypes.infer('0', int) == 0
    assert DataTypes.infer('-1', int) == -1
    assert DataTypes.infer('"1000028"', int) == 1000028
    assert DataTypes.infer('"1,000,028"', int) == 1000028

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

    # booleans
    assert DataTypes.infer('true', bool) is True
    assert DataTypes.infer('True', bool) is True
    assert DataTypes.infer('TRUE', bool) is True
    assert DataTypes.infer('false', bool) is False
    assert DataTypes.infer('False', bool) is False
    assert DataTypes.infer('FALSE', bool) is False

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
    dirty_date = datetime(1990, 1, 1, 23, 12, 11, int(0.003 * 10 ** 6))
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

    assert DataTypes.infer("2003-09-25T10:49:41", datetime) == datetime(2003, 9, 25, 10, 49, 41)  # iso minus T fields omitted.
    assert DataTypes.infer("2003-09-25T10:49", datetime) == datetime(2003, 9, 25, 10, 49)
    assert DataTypes.infer("2003-09-25T10", datetime) == datetime(2003, 9, 25, 10)

    assert DataTypes.infer("20080227T21:26:01.123456789", datetime) == datetime(2008, 2, 27, 21, 26, 1, 123456)  # high precision seconds
    assert DataTypes.infer("20030925T104941", datetime) == datetime(2003, 9, 25, 10, 49, 41)  # iso nospace T fields omitted.
    assert DataTypes.infer("20030925T1049", datetime) == datetime(2003, 9, 25, 10, 49, 0)
    assert DataTypes.infer("20030925T10", datetime) == datetime(2003, 9, 25, 10)

    assert DataTypes.infer("199709020908", datetime) == datetime(1997, 9, 2, 9, 8)
    assert DataTypes.infer("19970902090807", datetime) == datetime(1997, 9, 2, 9, 8, 7)
    assert DataTypes.infer("2003-09-25 10:49:41,502", datetime) == datetime(2003, 9, 25, 10, 49, 41, 502000)  # python logger format
    assert DataTypes.infer('0099-01-01T00:00:00', datetime) == datetime(99, 1, 1, 0, 0)  # 99 ad
    assert DataTypes.infer('0031-01-01T00:00:00', datetime) == datetime(31, 1, 1, 0, 0)  # 31 ad

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