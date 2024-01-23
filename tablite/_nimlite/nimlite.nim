import nimpy
import pymodules
import std/[paths]
import std/[os, options, tables]
# import encfile, table, csvparse

template isLib(): bool =
    isMainModule and appType == "lib" or true

when isLib:
    include includes/column_selector
    include includes/text_reader
