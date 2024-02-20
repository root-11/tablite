import std/options
from std/random import Rand, initRand, sample
from std/math import floor

var rng = none[Rand]()

const randChars = {'a'..'z','A'..'Z', '0'..'9'}

template corrupted*(d: typedesc = IOError): void = raise newException(d, "file corrupted")
template implement*(name: string = ""): void = raise newException(Exception, if name.len == 0: "not yet imlemented" else: "'" & name & "' not yet imlemented")

const isLittleEndian* = system.cpuEndian == littleEndian
const endiannessMark* = if isLittleEndian: "<" else: ">"

proc uniqueName*(desiredName: string, nameList: openArray[string] | seq[string]): string {.inline.} =
    ## makes sure that desiredName is unique to the given list
    var name = desiredName
    var idx = 1

    while name in nameList:
        name = desiredName & "_" & $idx
        inc idx

    name

proc unescapeSeq*(str: string): string {.inline.} =
    ## removes escape sequences from string because nim doesn't have true unescape
    case str:
    of "\\n": return "\n"
    of "\\t": return "\t"
    of "\\\"": return "\""
    of "\\\\": return "\\"
    else: return str

proc divmod*(x: int, y: int): (int, int) {.inline.} =
    ## function returns a tuple containing the quotient and the remainder
    let z = int(floor(x / y))

    return (z, x - y * z)

proc generateRandomString*(len: int): string {.inline.} =
    if rng.isNone:
        rng = some(initRand())

    ## generates a random string of len
    var str = newString(len)

    for i in 0..<len:
        str[i] = rng.get().sample(randChars)

    return str