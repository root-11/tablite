import std/unicode
# from std/strutils import parseInt, parseFloat
import infertypes

proc writeNumpyHeader*(fh: File, dtype: string, shape: uint): void =
    const magic = "\x93NUMPY"
    const major = "\x01"
    const minor = "\x00"

    let header = "{'descr': '" & dtype & "', 'fortran_order': False, 'shape': (" & $shape & ",)}"
    let header_len = len(header)
    let padding = (64 - ((len(magic) + len(major) + len(minor) + 2 + header_len)) mod 64)

    let padding_header = uint16 (padding + header_len)

    fh.write(magic)
    fh.write(major)
    fh.write(minor)

    discard fh.writeBuffer(padding_header.unsafeAddr, 2)

    fh.write(header)

    for i in 0..padding-2:
        fh.write(" ")
    fh.write("\n")

proc writeNumpyUnicode*(fh: var File, str: var string, unicode_len: uint): void =
    for rune in str.toRunes():
        var ch = uint32(rune)
        discard fh.writeBuffer(ch.unsafeAddr, 4)

    let dt = unicode_len - (uint str.runeLen)

    for i in 1..dt:
        fh.write("\x00\x00\x00\x00")

proc writeNumpyInt*(fh: var File, str: var string): void =
    let parsed = inferInt(addr str)
    discard fh.writeBuffer(parsed.unsafeAddr, 8)

proc writeNumpyFloat*(fh: var File, str: var string): void =
    let parsed = inferFloat(addr str)
    discard fh.writeBuffer(parsed.unsafeAddr, 8)

proc writeNumpyBool*(fh: var File, str: var string): void =
    fh.write((if str.toLower() == "true": '\x01' else: '\x00'))
