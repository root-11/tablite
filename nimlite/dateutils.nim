import std/times
from utils import divmod
from std/math import splitDecimal

const DAYS_PER_MONTH_TABLE* = [
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], # not leap
    [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # leap
]

type YearRange* = range[1..9999]
type MicrosecondRange* = range[0..999_999]

proc isLeapYear*(year: int): bool {.inline.} = year mod 4 == 0 and (year mod 100 != 0 or year mod 400 == 0)
proc getDaysInMonth*(year, month: int): int {.inline.} = DAYS_PER_MONTH_TABLE[int isLeapYear(year)][month - 1]

proc toTimedelta*(
    weeks = 0, days = 0, hours = 0, minutes = 0, seconds = 0, milliseconds = 0, microseconds: int = 0
): (int, int, int) {.inline.} =
    var d, s, us: int

    var v_weeks = weeks
    var v_days = days
    var v_hours = hours
    var v_minutes = minutes
    var v_seconds = seconds
    var v_milliseconds = milliseconds
    var v_microseconds = microseconds

    # Normalize everything to days, seconds, microseconds.
    v_days += v_weeks*7
    v_seconds += v_minutes*60 + v_hours*3600
    v_microseconds += v_milliseconds*1000

    d = v_days

    (v_days, v_seconds) = divmod(v_seconds, 24*3600)

    d += v_days
    s += int(v_seconds) # can't overflow

    v_microseconds = int(v_microseconds)
    (v_seconds, v_microseconds) = divmod(v_microseconds, 1000000)
    (v_days, v_seconds) = divmod(v_seconds, 24*3600)
    d += v_days
    s += v_seconds

    # Just a little bit of carrying possible for microseconds and seconds.
    (v_seconds, us) = divmod(v_microseconds, 1000000)
    s += v_seconds
    (v_days, s) = divmod(s, 24*3600)
    d += v_days

    return (d, s, us)

proc extractUnit(d: int, unit: int): (int, int) {.inline.} =
    var (idiv, imod) = divmod(d, unit)

    if (imod < 0):
        imod += unit
        idiv -= 1

    return (idiv, imod)

proc days2YearsDays(days: int): (int, int) =
    let days_per_400years = (400*365 + 100 - 4 + 1)
    # Adjust so it's relative to the year 2000 (divisible by 400)
    var tdays = days - (365*30 + 7)

    # Break down the 400 year cycle to get the year and day within the year
    var year: int
    (year, tdays) = extractUnit(tdays, days_per_400years)
    year = 400 * year

    # Work out the year/day within the 400 year cycle
    if (tdays >= 366):
        year = year + (100 * int ((tdays-1) / (100*365 + 25 - 1)))
        tdays = (tdays-1) mod (100*365 + 25 - 1)
        if (tdays >= 365):
            year += 4 * int ((tdays+1) / (4*365 + 1))
            tdays = (tdays+1) mod (4*365 + 1)
            if (tdays >= 366):
                year = year + int ((tdays-1) / 365)
                tdays = (tdays-1) mod 365

    return (year + 2000, tdays)

proc days2Components(days: int): (int, Month, MonthdayRange) =
    var (dts_year, idays) = days2YearsDays(days)
    let month_lengths = DAYS_PER_MONTH_TABLE[int isLeapYear(dts_year)]

    for i in 0..11:
        if (idays < month_lengths[i]):
            let dts_month = Month(i + 1)
            let dts_day = MonthdayRange (idays + 1)
            return (dts_year, dts_month, dts_day)
        else:
            idays = (idays - month_lengths[i])

    raise newException(IndexDefect, "failed")

proc days2Date*(days: int): DateTime =
    let (dts_year, dts_month, dts_day) = days2Components(days)

    return dateTime(dts_year, dts_month, dts_day, zone=utc())

proc delta2Date*(
    weeks = 0, days = 0, hours = 0, minutes = 0, seconds = 0, milliseconds = 0, microseconds: int = 0
): DateTime =
    let (d, s, us) = toTimedelta(weeks, days, hours, minutes, seconds, milliseconds, microseconds)
    let date = days2Date(d)
    let durr = initDuration(seconds=s, microseconds=us)

    let final = date + durr

    return final

proc date2NimDateTime*(year: int, month: int, day: int): DateTime {.inline.} =
    return dateTime(year, Month(month), MonthdayRange(day), zone=utc())

proc datetime2NimDatetime*(year: int, month: int, day: int, hour: int, minute: int, second: int, microsecond: int): DateTime {.inline.} =
    return dateTime(year, Month(month), MonthdayRange(day), hour, minute, second, microsecond * 1000, zone=utc())

proc time2NimDuration*(hour: int, minute: int, second: int, microsecond: int): Duration {.inline.} =
    return initDuration(hours=hour, minutes=minute, seconds=second, microseconds=microsecond)

proc duration2Time*(self: Duration): Time {.inline.} = Time() + self
proc time2Duration*(self: Time): Duration {.inline.} = self - Time()
proc seconds2Duration*(seconds: float): Duration {.inline.} =
    let (secs, frac) = seconds.splitDecimal()
    let us = frac * 1_000_000
    return initDuration(seconds=int secs, microseconds=int us)

proc duration2Seconds*(dur: Duration): float {.inline.} = dur.inMicroseconds / 1_000_000

proc duration2Date*(dur: Duration): DateTime {.inline.} = dateTime(1970, mJan, 1, zone=utc()) + dur
proc seconds2Date*(seconds: float): DateTime {.inline.} = duration2Date(seconds2Duration(seconds))
proc datetime2Date*(self: DateTime): DateTime {.inline.} = dateTime(self.year, self.month, self.monthday, zone=utc())