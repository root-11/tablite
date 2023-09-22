from std/enumerate import enumerate
import insertsort
import infertypes

type DataTypes* = enum
    # sort by difficulty
    DT_NONE, DT_BOOL,
    DT_DATE_SHORT,
    DT_INT, DT_FLOAT,
    DT_DATETIME, DT_DATETIME_US, DT_DATE, DT_DATE_US, DT_TIME,
    DT_STRING,
    DT_MAX_ELEMENTS

type Rank* = array[int(DataTypes.DT_MAX_ELEMENTS), (DataTypes, uint)]

proc newRank*(): Rank =
    var rank {.noinit.}: Rank

    for i in 0..(int(DataTypes.DT_MAX_ELEMENTS)-1):
        rank[i] = (DataTypes(i), uint 0)

    return rank

iterator iter*(rank: var Rank): ptr (DataTypes, uint) {.closure.} =
    var x = 0
    let max = int(DataTypes.DT_MAX_ELEMENTS)
    while x < max:
        yield rank[x].unsafeAddr
        inc x
    raise newException(ResourceExhaustedError, "stop iteration")

proc cmpDtypes(a: (DataTypes, uint), b: (DataTypes, uint)): bool = 
    return a[1] > b[1]

proc cmpDtypesStringless(a: (DataTypes, uint), b: (DataTypes, uint)): bool = 
    # puts strings at the end of non-zero sequence
    if a[0] == DataTypes.DT_STRING and b[1] > 0: return false
    elif b[0] == DataTypes.DT_STRING and a[1] > 0: return true
    return cmpDtypes(a, b)

proc sortRanks*(rank: var Rank, stringless: bool): void =
    if stringless:
        rank.insertSort(cmpDtypesStringless)
    else:
        rank.insertSort(cmpDtypes)

proc updateRank*(rank: var Rank, str: ptr string): DataTypes =
    var rank_dtype: DataTypes
    var index: int
    var rank_count: uint

    for i, r_addr in enumerate(rank.iter()):
        try:
            case r_addr[0]:
                of DataTypes.DT_INT:
                    discard str.inferInt()
                of DataTypes.DT_FLOAT:
                    discard str.inferFloat()
                of DataTypes.DT_DATE:
                    discard str.inferDate(false, false)
                of DataTypes.DT_DATE_SHORT:
                    discard str.inferDate(true, false)
                of DataTypes.DT_DATE_US:
                    discard str.inferDate(false, true)
                of DataTypes.DT_TIME:
                    discard str.inferTime()
                of DataTypes.DT_DATETIME:
                    discard str.inferDatetime(false)
                of DataTypes.DT_DATETIME_US:
                    discard str.inferDatetime(true)
                of DataTypes.DT_STRING:
                    discard
                of DataTypes.DT_BOOL:
                    discard str.inferBool()
                of DataTypes.DT_NONE:
                    discard str.inferNone()
                of DataTypes.DT_MAX_ELEMENTS:
                    raise newException(Exception, "not a type")
        except ValueError as e:
            # echo $e.msg & " | " & $e.getStackTrace()
            continue

        rank_dtype = r_addr[0]
        rank_count = r_addr[1]
        index = i
        break

    rank[index] = (rank_dtype, rank_count + 1)
    rank.sortRanks(true)

    return rank_dtype