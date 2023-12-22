from std/math import floor

proc uniqueName*(desired_name: string, name_list: seq[string]): string {.inline.} =
    var name = desired_name
    var idx = 1

    while name in name_list:
        name = desired_name & "_" & $idx
        inc idx

    return name

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

proc extractUnit*(d: int, unit: int): (int, int) {.inline.} =
    var (idiv, imod) = divmod(d, unit)

    if (imod < 0):
        imod += unit
        idiv -= 1

    return (idiv, imod)