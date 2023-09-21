from std/math import floor
from std/unicode import runeLen
import std/strutils
import pickling


const DAYS_IN_MONTH = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

proc inferBool*(str: ptr string): bool =
    if str[].toLower() == "true":
        return true
    elif str[].toLower() == "false":
        return false

    raise newException(ValueError, "not a boolean value")

proc inferInt*(str: ptr string): int =
    let sstr = str[]
    return parseInt(sstr)
proc inferFloat*(str: ptr string): float = parseFloat(str[])

proc parseDateWords(str: ptr string, allow_time: bool): (array[3, string], int) =
    const accepted_tokens = [' ', '.', '-', '/']

    var has_tokens = false
    let str_len = str[].runeLen

    if str_len != str[].len: # datetimes are not in unicode
        raise newException(ValueError, "not a value")


    for i in 0..4: # date will have tokens in first 5 characters YYYY-/DD-/MM-
        var ch: char

        try:
            ch = char str[i]
        except Exception: # bad encoding
            raise newException(ValueError, "not a date")

        if ch.isDigit:
            continue

        if not (ch in accepted_tokens): # not a digit, nor an accepted token
            if ch in [':', 'T'] and allow_time: # time token but we allow parsin time
                continue
            raise newException(ValueError, "not a date: '" & ch & "'" & $allow_time)

        has_tokens = true
        echo $ch
        break

    var substrings: array[3, string]


    if has_tokens:
        var active_token = '\x00'
        var slice_start: int
        var was_digit = false
        var substring_count: int
        var idx = 0

        while idx < str_len:
            let ch = str[idx]

            if idx == 0 and not ch.isDigit: # dates always start with a digit
                raise newException(ValueError, "not a date")

            if ch.isDigit:
                if not was_digit:
                    slice_start = idx
                was_digit = true
                inc idx
                continue

            if active_token == '\x00':
                active_token = ch
            elif active_token != ch: # date tokens do should not change
                if substring_count == 2 and allow_time and ch in [' ', 'T']: # time token and we can parse time
                    break
                raise newException(ValueError, "not a date: '" & $ch & "'")

            substrings[substring_count] = $str[].substr(slice_start, idx-1)
            inc substring_count

            was_digit = false
            inc idx

        if substring_count != 2 or (idx - slice_start) == 0:
            raise newException(ValueError, "not a date") # should have 2 substrings and some leftover

        substrings[substring_count] = $str[].substr(slice_start, idx-1)

        return (substrings, idx)

    # YYYYMMDD
    if str_len < 8:
        raise newException(ValueError, "not a date")

    substrings[0] = str[].substr(0, 3)
    substrings[1] = str[].substr(4, 5)
    substrings[2] = str[].substr(6, 7)

    return (substrings, 8)

proc isLeapYear(year: int): bool = year mod 4 == 0 and (year mod 100 != 0 or year mod 400 == 0)
proc getDaysInMonth(year, month: int): int =
    if month == 2 and isLeapYear(year):
        return 29
    return DAYS_IN_MONTH[month]

proc wordsToDate(date_words: ptr array[3, string], is_american: bool): PY_Date =
    var year, month, day: int
    var month_or_day: array[2, int]

    if date_words[0].len == 4:
        year = parseInt(date_words[0])
        month_or_day[0] = parseInt(date_words[1])
        month_or_day[1] = parseInt(date_words[2])

        if is_american:
            raise newException(ValueError, "invalid date")

        # if YYYY first it's always YYYY-MM-DD format
        return newPyDate(uint16 year, uint8 month_or_day[0], uint8 month_or_day[1])
    elif date_words[2].len == 4:
        year = parseInt(date_words[2])
        month_or_day[0] = parseInt(date_words[0])
        month_or_day[1] = parseInt(date_words[1])

    if year < 0 or year > 9999:
        raise newException(ValueError, "date out of range")

    if month_or_day[0] <= 0 or month_or_day[1] <= 0 or month_or_day[0] > 12 and month_or_day[1] > 12:
        raise newException(ValueError, "date out of range")

    echo $month_or_day

    if month_or_day[0] <= 12 and month_or_day[1] <= 12: # MMDDYYYY/DDMMYYYY
        # if both under 12, use tie breaker
        if is_american:
            month = month_or_day[0]
            day = month_or_day[1]
        else:
            month = month_or_day[1]
            day = month_or_day[0]
    elif month_or_day[0] < 12: # MMDDYYYY
        if not is_american: # must be american
            raise newException(ValueError, "invalid format")
        # day
        day = month_or_day[0]
        month = month_or_day[1]
    elif month_or_day[1] < 12: # DDMMYYYY
        if is_american: # cannot be american
            raise newException(ValueError, "invalid format")
        # month
        day = month_or_day[0]
        month = month_or_day[1]
    else:
        raise newException(ValueError, "date out of range")

    if getDaysInMonth(year, month) < day:
        raise newException(ValueError, "day out of range")

    return newPyDate(uint16 year, uint8 month, uint8 day)

proc inferDate*(str: ptr string, is_american: bool): PY_Date =
    let str_len = str[].runeLen

    if str_len > 10 or str_len < 8: # string len will never match
        raise newException(ValueError, "not a date")

    let (date_words, _) = str.parseDateWords(false)

    return wordsToDate(date_words.unsafeAddr, is_american)

proc divmod(x: int, y: int): (int, int) =
    let z = int(floor(x / y))

    return (z, x - y * z)

proc toTimedelta(
    weeks = 0, days = 0, hours = 0, minutes = 0, seconds = 0, milliseconds = 0, microseconds: int = 0
): (int, int, int) =
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

proc parse_hh_mm_ss_ff(tstr: ptr string): (uint8, uint8, uint8, uint32) =
    # Parses things of the form HH[:MM[:SS[.fff[fff]]]]
    let len_str = tstr[].len

    var time_comps: array[4, int]
    var pos = 0

    for comp in 0..2:
        if (len_str - pos) < 2:
            raise newException(ValueError, "Incomplete time component")

        let substr = tstr[].substr(pos, pos+1)

        time_comps[comp] = parseInt(substr)

        pos += 2

        if pos >= len_str or comp >= 2:
            break

        let next_char = tstr[pos]

        if next_char != ':':
            raise newException(ValueError, "Invalid time separator: " & $next_char)

        pos += 1

    if pos < len_str:
        if tstr[pos] != '.':
            raise newException(ValueError, "Invalid microsecond component")
        else:
            pos += 1

            let len_remainder = len_str - pos
            if not (len_remainder in [3, 6]):
                raise newException(ValueError, "Invalid microsecond component")

            time_comps[3] = parseInt(tstr[].substr(pos))
            if len_remainder == 3:
                time_comps[3] *= 1000

    return (uint8 time_comps[0], uint8 time_comps[1], uint8 time_comps[2], uint32 time_comps[3])

proc inferTime*(str: ptr string): PY_Time =
    let str_len = str[].len

    if str_len < 2:
        raise newException(ValueError, "not a time")

    if not (":" in str[]):
        # Format supported is HH[MM[SS]]
        if str_len in [2, 4, 6]:
            var hour, minute, second: uint8

            hour = uint8 parseInt(str[].substr(0, 1))

            if str_len >= 4:
                minute = uint8 parseInt(str[].substr(2, 3))

            if str_len >= 6:
                second = uint8 parseInt(str[].substr(4, 5))

            return newPyTime(hour, minute, second, uint32 0)

        raise newException(ValueError, "not a time")

    # Format supported is HH[:MM[:SS[.fff[fff]]]][+HH:MM[:SS[.ffffff]]]
    let tz_pos_minus = str[].find("-")
    let tz_pos_plus = str[].find("+")

    var tz_pos = -1

    if tz_pos_minus != -1 and tz_pos_plus != -1:
        raise newException(ValueError, "not a time")
    elif tz_pos_plus != -1:
        tz_pos = tz_pos_plus
    elif tz_pos_minus != -1:
        tz_pos = tz_pos_minus

    var timestr = (if tz_pos == -1: str[] else: str[].substr(0, tz_pos-1))

    let (hour, minute, second, microsecond) = parse_hh_mm_ss_ff(timestr.unsafeAddr)

    if tz_pos >= 0:
        let tzstr = str[].substr(tz_pos + 1)

        if not (tzstr.len in [5, 8, 15]):
            raise newException(Exception, "invalid timezone")

        echo $str[tz_pos]

        let tz_sign: int8 = (if str[tz_pos] == '-': -1 else: 1)
        let (tz_hours_p, tz_minutes_p, tz_seconds_p, tz_microseconds_p) = parse_hh_mm_ss_ff(tzstr.unsafeAddr)
        var (tz_days, tz_seconds, tz_microseconds) = toTimedelta(
            hours = tz_sign * int tz_hours_p,
            minutes = tz_sign * int tz_minutes_p,
            seconds = tz_sign * int tz_seconds_p,
            microseconds = tz_sign * int tz_microseconds_p
        )

        return newPyTime(hour, minute, second, microsecond, int32 tz_days, int32 tz_seconds, int32 tz_microseconds)

    return newPyTime(hour, minute, second, microsecond)


proc inferDatetime*(str: ptr string, is_american: bool): PY_DateTime =
    echo $str[]

    let str_len = str[].runeLen

    if str_len > 42 or str_len < 10: # string len will never match
        raise newException(ValueError, "not a datetime: " & $str_len)

    let (date_words, toffset) = str.parseDateWords(true)
    let first_tchar = str[toffset]
    var tstr {.noinit.}: string

    if(first_tchar.isDigit):
        tstr = str[].substr(toffset)
    elif first_tchar in [' ', 'T']:
        tstr = str[].substr(toffset + 1)
    else:
        raise newException(ValueError, "not a datetime")

    let date = wordsToDate(date_words.unsafeAddr, is_american)
    let time = inferTime(tstr.unsafeAddr)

    return newPyDateTime(date, time)
