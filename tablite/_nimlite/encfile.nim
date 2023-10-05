import std/[encodings, strutils]

type Encodings* {.pure.} = enum ENC_UTF8, ENC_UTF16, ENC_CONV
type FileEncoding* = object
    encoding: Encodings
    conv: string

type BaseEncodedFile* = ref object of RootObj
    fh: File

type FileUTF8 = ref object of BaseEncodedFile
type FileConvertable = ref object of BaseEncodedFile
    encoder: EncodingConverter
type FileUTF16 = ref object of BaseEncodedFile
    endianness: Endianness

proc endOfFile*(f: BaseEncodedFile): bool = f.fh.endOfFile()
proc getFilePos*(f: BaseEncodedFile): uint = uint f.fh.getFilePos()
proc setFilePos*(f: BaseEncodedFile, pos: int64, relativeTo: FileSeekPos): void = f.fh.setFilePos(pos, relativeTo)

proc close*(f: FileUTF8): void = discard
proc close*(f: FileUTF16): void = discard
proc close*(f: FileConvertable): void = f.encoder.close()
proc close*(f: BaseEncodedFile): void =
    f.fh.close()

    if f of FileUTF8:
        (cast[FileUTF8](f)).close()
    elif f of FileUTF16:
        (cast[FileUTF16](f)).close()
    elif f of FileConvertable:
        (cast[FileConvertable](f)).close()
    else:
        raise newException(Exception, "encoding not implemented")

proc readLine(f: FileUTF8, str: var string): bool = f.fh.readLine(str)
proc readLine(f: FileConvertable, str: var string): bool =
    let res = f.fh.readLine(str)

    if res:
        str = f.encoder.convert(str)

    return res

proc readLine(f: FileUTF16, str: var string): bool =
    var ch_arr: array[2048, uint8] # must be divisible by 2
    var ch: uint16

    let newline_char: uint16 = 0x000a
    var wchar_seq = newSeqOfCap[uint16](80)

    var file_offset = f.fh.getFilePos()
    var elements = f.fh.readBuffer(addr ch_arr, 2)
    var el_iter = 0

    if (elements mod 2) != 0:
        raise newException(Exception, "malformed file")

    while likely(el_iter < elements):
        if f.endianness == bigEndian: # big if true
            (ch_arr[el_iter], ch_arr[el_iter+1]) = (ch_arr[el_iter+1], ch_arr[el_iter])

        ch = cast[uint16](ch_arr)

        el_iter = el_iter + 2

        if newline_char == ch:
            if wchar_seq.len == 0:
                str = "" # empty line
                return true
            break

        wchar_seq.add(ch)

        if el_iter >= elements:
            file_offset = f.fh.getFilePos()
            elements = f.fh.readBuffer(addr ch_arr, 2)
            el_iter = 0

            if (elements mod 2) != 0:
                raise newException(Exception, "malformed file")

    f.fh.setFilePos(file_offset + el_iter, fspSet)

    var wstr = newWideCString(wchar_seq.len)

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
    elif f of FileConvertable:
        return readLine(cast[FileConvertable](f), str)
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

proc newFileConvertable(filename: string, format: string): FileConvertable =
    let fh = open(filename, fmRead)
    let encoder = open("UTF-8", format)

    return FileConvertable(fh: fh, encoder: encoder)

proc newFile*(filename: string, encoding: FileEncoding): BaseEncodedFile =
    case encoding.encoding:
        of ENC_UTF8: return newFileUTF8(filename)
        of ENC_UTF16: return newFileUTF16(filename)
        of ENC_CONV: return newFileConvertable(filename, encoding.conv)

proc findNewlines*(fh: BaseEncodedFile): (seq[uint], uint) =
    var newline_offsets = newSeq[uint](1)
    var total_lines: uint = 0
    var str: string

    newline_offsets[0] = fh.getFilePos()

    while likely(fh.readLine(str)):
        inc total_lines

        newline_offsets.add(fh.getFilePos())

    return (newline_offsets, total_lines)

proc findNewlines*(path: string, encoding: FileEncoding): (seq[uint], uint) =
    let fh = newFile(path, encoding)
    try:
        return findNewlines(fh)
    finally:
        fh.close()

proc str2Enc*(encoding: string): FileEncoding {.inline.} =
    let upper = encoding.toUpper()
    
    case upper:
        of $ENC_UTF8: return FileEncoding(encoding: ENC_UTF8)
        of $ENC_UTF16: return FileEncoding(encoding: ENC_UTF16)
        else:
            let header = $ENC_CONV & "|"
            if upper.startsWith(header):
                return FileEncoding(encoding: ENC_CONV, conv: encoding.substr(header.len))
            else: raise newException(IOError, "invalid encoding: " & encoding)

proc str2ConvEnc*(encoding: string): FileEncoding {.inline.} =
    return str2Enc($ENC_CONV & "|" & encoding)

proc `$`*(encoding: FileEncoding): string {.inline.} =
    case encoding.encoding:
        of ENC_UTF8, ENC_UTF16: return $encoding.encoding
        of ENC_CONV: $encoding.encoding & "|" & encoding.conv