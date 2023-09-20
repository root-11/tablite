proc writeNumpyHeader(fh: File, dtype: string, shape: uint): void =
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