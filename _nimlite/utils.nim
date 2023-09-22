proc uniqueName*(desired_name: string, name_list: seq[string]): string =
    var name = desired_name
    var idx = 1

    while name in name_list:
        name = desired_name & "_" & $idx
        inc idx

    return name

proc unescapeSeq*(str: string): string = # nim has no true unescape
    case str:
        of "\\n": return "\n"
        of "\\t": return "\t"
        of "\\\"": return "\""
        of "\\\\": return "\\"

    return str
