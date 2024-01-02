from std/random import randomize, sample
from std/math import floor
from std/sugar import collect
from std/tables import Table

randomize()

const rand_chars = {'a'..'z','A'..'Z', '0'..'9'}

type Iterable[T] = proc: iterator: T
template corrupted*(): void = raise newException(IOError, "file corrupted")
template implement*(name: string = ""): void = raise newException(Exception, if name.len == 0: "not yet imlemented" else: "'" & name & "' not yet imlemented")

const isLittleEndian* = system.cpuEndian == littleEndian
const endiannessMark* = if isLittleEndian: "<" else: ">"

proc uniqueName*(desired_name: string, name_list: openArray[string] | seq[string]): string {.inline.} =
    var name = desired_name
    var idx = 1

    while name in name_list:
        name = desired_name & "_" & $idx
        inc idx

    name

proc unescapeSeq*(str: string): string {.inline.} = # nim has no true unescape
    case str:
        of "\\n": return "\n"
        of "\\t": return "\t"
        of "\\\"": return "\""
        of "\\\\": return "\\"

    return str

proc divmod*(x: int, y: int): (int, int) {.inline.} =
    let z = int(floor(x / y))

    return (z, x - y * z)

proc generateRandomString*(len: int): string {.inline.} =
    var str = newString(len)

    for i in 0..<len:
        str[i] = sample(rand_chars)

    return str

proc extractUnit*(d: int, unit: int): (int, int) {.inline.} =
    var (idiv, imod) = divmod(d, unit)

    if (imod < 0):
        imod += unit
        idiv -= 1

    return (idiv, imod)

proc toIterKeys*[K, V](self: Table[K, V]): iterator(): K =
    return iterator(): K =
        for k in self.keys:
            yield k

proc toIterValues*[K, V](self: Table[K, V]): iterator(): V =
    return iterator(): V =
        for v in self.values:
            yield v

proc toIterPairs*[K, V](self: Table[K, V]): iterator(): (K, V) =
    return iterator(): (K, V) =
        for (k, v) in self.pairs:
            yield (k, v)

proc toSeqKeys*[K, V](self: Table[K, V]): seq[K] =
    return collect:
        for e in self.toIterKeys():
            e
proc toSeqValues*[K, V](self: Table[K, V]): seq[V] =
    return collect:
        for e in self.toIterValues():
            e
proc toSeqPairs*[K, V](self: Table[K, V]): seq[(K, V)] =
    return collect:
        for e in self.toIterPairs():
            e