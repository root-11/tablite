from std/random import randomize, sample
from std/math import floor

randomize()

const rand_chars = {'a'..'z','A'..'Z', '0'..'9'}

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
    ## generates a random string of len
    var str = newString(len)

    for i in 0..<len:
        str[i] = sample(rand_chars)

    return str