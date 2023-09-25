type Encodings* {.pure.} = enum ENC_UTF8, ENC_UTF16

type BaseEncodedFile* = ref object of RootObj
    fh: File

type FileUTF8 = ref object of BaseEncodedFile
type FileUTF16 = ref object of BaseEncodedFile
    endianness: Endianness

proc endOfFile*(f: BaseEncodedFile): bool = f.fh.endOfFile()
proc getFilePos*(f: BaseEncodedFile): uint = uint f.fh.getFilePos()
proc setFilePos*(f: BaseEncodedFile, pos: int64, relativeTo: FileSeekPos): void = f.fh.setFilePos(pos, relativeTo)
proc close*(f: BaseEncodedFile): void = f.fh.close()

proc readLine(f: FileUTF8, str: var string): bool = f.fh.readLine(str)
proc readLine(f: FileUTF16, str: var string): bool = 
    var ch_arr {.noinit.}: array[2, uint8]
    var ch: uint16

    let newline_char: uint16 = 0x000a
    var wchar_seq {.noinit.} = newSeqOfCap[uint16](80)

    while unlikely(not f.endOfFile):
        if f.fh.readBuffer(addr ch_arr, 2) != ch_arr.len:
            raise newException(Exception, "malformed file")

        if f.endianness == bigEndian: # big if true
            (ch_arr[0], ch_arr[1]) = (ch_arr[1], ch_arr[0])

        ch = cast[uint16](ch_arr)

        if newline_char == ch:
            break

        wchar_seq.add(ch)

    var wstr {.noinit.} = newWideCString(wchar_seq.len)

    if wchar_seq.len > 0:
        copyMem(wstr[0].unsafeAddr, wchar_seq[0].unsafeAddr, wchar_seq.len * 2)
    else:
        return false

    str = $wstr

    return true

proc readLine*(f: BaseEncodedFile, str: var string): bool = 
    if f of FileUTF8:
        return readLine(cast[FileUTF8](f), str)
    elif f of FileUTF16:
        return readLine(cast[FileUTF16](f), str)
    else:
        raise newException(Exception, "encoding not implemented")

proc newFileUTF16(filename: string): FileUTF16 =
    var fh = open(filename, fmRead)

    if fh.getFileSize() mod 2 != 0:
        raise newException(Exception, "invalid size")

    var bom_bytes: array[2, uint16]
    
    if fh.readBuffer(addr bom_bytes, bom_bytes.len) != bom_bytes.len:
        raise newException(Exception, "cannot find bom")

    var bom = cast[uint16](bom_bytes)
    var endianness: Endianness;

    if bom == 0xfeff:
        endianness = Endianness.littleEndian
    elif bom == 0xfffe:
        endianness = Endianness.bigEndian
    else:
        raise newException(Exception, "invalid bom")

    return FileUTF16(fh: fh, endianness: endianness)

proc newFileUTF8(filename: string): FileUTF8 =
    let fh = open(filename, fmRead)

    var bom: array[3, uint8]
    var bom_bytes = fh.readBytes(bom, 0, 3)

    # detect bom
    if bom_bytes != 3:
        fh.setFilePos(0, FileSeekPos.fspSet)
    elif bom[0] != 0xEF or bom[1] != 0xBB or bom[2] != 0xBF:
        fh.setFilePos(0, FileSeekPos.fspSet)

    return FileUTF8(fh: fh)

proc newFile*(filename: string, encoding: Encodings): BaseEncodedFile =
    case encoding:
        of ENC_UTF8:
            return newFileUTF8(filename)
        of ENC_UTF16:
            return newFileUTF16(filename)

proc findNewlines*(fh: BaseEncodedFile): (seq[uint], uint) =
    var newline_offsets = newSeq[uint](1)
    var total_lines: uint = 0
    var str: string

    newline_offsets[0] = fh.getFilePos()

    while likely(fh.readLine(str)):
        inc total_lines

        newline_offsets.add(fh.getFilePos())

    return (newline_offsets, total_lines)

proc findNewlines*(path: string, encoding: Encodings): (seq[uint], uint) =
    let fh = newFile(path, encoding)
    try:
        return findNewlines(fh)
    finally:
        fh.close()