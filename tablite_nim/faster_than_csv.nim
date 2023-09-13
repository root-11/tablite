import lexbase, json, tables, streams, strutils, os, osproc, xmltree, httpclient, terminal, experimental/diff, nimpy

type CsvParser = object of BaseLexer # Custom copypasted from Nim stdlib "parsecsv.nim"
    row, headers: seq[string]
    sep, quote, esc: char
    currRow: int

template close(self: var CsvParser) = lexbase.close(self)

template open(self: var CsvParser, input: Stream, separator = ',', quote = '"', escape = '\0') =
    lexbase.open(self, input, 64)
    self.sep = separator
    self.quote = quote
    self.esc = escape

template open(self: var CsvParser, filename: string, separator = ',', quote = '"', escape = '\0') =
    open(self, newFileStream(filename, fmRead), separator, quote, escape)

template open(self: var CsvParser, file: File, separator = ',', quote = '"', escape = '\0') =
    open(self, newFileStream(file), separator, quote, escape)


proc parseField(self: var CsvParser, a: var string) =
    var pos = create int
    pos[] = self.bufpos
    while self.buf[pos[]] in {' ', '\t'}: inc pos[]
    setLen(a, 0) # reuse memory
    if self.buf[pos[]] == self.quote and self.quote != '\0':
        inc pos[]
        while true:
            let c = self.buf[pos[]]
            if c == '\0':
                self.bufpos = pos[]
                raise newException(IOError, "CSV parse error, missing quotes at index " & $pos[])
            elif c == self.quote:
                if self.esc == '\0' and self.buf[pos[] + 1] == self.quote:
                    add(a, self.quote)
                    inc pos[], 2
                else:
                    inc(pos[])
                    break
            elif c == self.esc:
                add(a, self.buf[pos[] + 1])
                inc pos[], 2
            else:
                case c
                of '\c':
                    pos[] = handleCR(self, pos[])
                    add(a, '\n')
                of '\l':
                    pos[] = handleLF(self, pos[])
                    add(a, '\n')
                else:
                    add(a, c)
                    inc(pos[])
    else:
        while true:
            let c = self.buf[pos[]]
            if c == self.sep: break
            if c in {'\c', '\l', '\0'}: break
            add(a, c)
            inc pos[]
    self.bufpos = pos[]
    if likely(pos != nil): dealloc pos

proc readRow(self: var CsvParser): bool =
    var col {.noalias.} = create int
    while true:
        case self.buf[self.bufpos]
        of '\c': self.bufpos = handleCR(self, self.bufpos)
        of '\l': self.bufpos = handleLF(self, self.bufpos)
        else: break
    while self.buf[self.bufpos] != '\0':
        let oldlen = self.row.len
        if oldlen < col[] + 1:
            setLen(self.row, col[] + 1)
            self.row[col[]] = ""
        parseField(self, self.row[col[]])
        inc col[]
        if self.buf[self.bufpos] == self.sep: inc(self.bufpos)
        else:
            case self.buf[self.bufpos]
            of '\c', '\l':
                while true:
                    case self.buf[self.bufpos]
                    of '\c': self.bufpos = handleCR(self, self.bufpos)
                    of '\l': self.bufpos = handleLF(self, self.bufpos)
                    else: break
            of '\0': discard
            else: raise newException(IOError, "CSV parse error, missing separators at column " & $col[])
            break
    setLen(self.row, col[])
    result = col[] > 0
    inc(self.currRow)
    if likely(col != nil): dealloc col

template rowEntry(self: var CsvParser; entry: var string) =
    let index = create int
    index[] = self.headers.find(entry)
    if likely(index[] >= 0): entry = self.row[index[]] else: echo "ERROR: Key not found: " & entry
    if likely(index != nil): dealloc index

template readHeaderRow(self: var CsvParser) =
    if likely(self.readRow()): self.headers = self.row

# ^ CSV Parser ##################################### v CSV functions for Python


const html_table_header = """<!DOCTYPE html>
<html style="background-color:lightcyan">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.2/css/bulma.min.css" async defer >
</head>
<body><br><br>
  <div class="container is-fluid">
    <table class="table is-bordered is-striped is-hoverable is-fullwidth">"""

const karax_header = """
include karax/prelude  # nimble install karax http://github.com/pragmagic/karax

proc createDom(): VNode {.discardable.} =
  result = buildHtml(table):
    thead(class="thead"):
      tr(class="has-background-grey-light"):
"""

proc csv2list*(csv_file_path: string; separator: char = ',';
    quote: char = '"'; escape: char = '\x00'): seq[string] {.exportpy.} =
    ## Stream Read CSV to a list of strings.
    let parser {.noalias.} = create CsvParser
    parser[].open(csv_file_path, separator, quote, escape)
    parser[].readHeaderRow()
    while parser[].readRow():
        for column in parser[].headers.mitems:
            parser[].rowEntry(column)
            result.add column
    parser[].close()
    if likely(parser != nil): dealloc parser

proc csv2dict*(csv_file_path: string; separator: char = ',';
  quote: char = '"'; escape: char = '\x00'): seq[Table[string, string]] {.exportpy.} =
    ## Stream Read CSV to a list of dictionaries. This is very similar to ``pandas.read_csv(filename)``.
    let parser {.noalias.} = create CsvParser
    parser[].open(csv_file_path, separator, quote, escape)
    parser[].readHeaderRow()
    while parser[].readRow():
        for column in parser[].headers.items:
            var col = column
            parser[].rowEntry(col)
            result.add {column: col}.toTable
    parser[].close()
    if parser != nil: dealloc parser

proc read_clipboard*(has_header: bool = true, separator: char = ',',
  quote: char = '"', escape: char = '\x00'): seq[(string, string)] {.exportpy.} =
    ## Stream Read CSV to a list of dictionaries. This is very similar to ``pandas.read_clipboard()``.
    let parser {.noalias.} = create(CsvParser)
    let (output, exitCode) = execCmdEx(
      when defined(linux): "xclip -out"
    elif defined(macos): "pbpaste"
    elif defined(windows): "Get-Clipboard"
    else: "")
    if likely(exitCode == 0):
        parser[].open(newStringStream(output), separator, quote, escape)
        if has_header:
            parser[].readHeaderRow()
            while parser[].readRow():
                for column in parser[].headers.items:
                    var col = column
                    parser[].rowEntry(col)
                    result.add {column: col}
        else:
            var counter = 0
            while parser[].readRow():
                for value in parser[].row.items:
                    result.add {$counter: $value}
                inc counter
    parser[].close()
    if parser != nil: dealloc parser

proc url2csv*(url: string; separator: char = ','; quote: char = '"'; escape: char = '\x00';
  agent: string = defUserAgent; maxRedirects: int = 5; timeout: int = -1; http_method: string = "GET", body: string = ""): seq[(string, string)] {.exportpy.} =
    ## Stream Read URL to CSV to a list of dictionaries. This is very similar to ``pandas.read_csv(url)``.
    assert url.startsWith"http", "URL must be a valid, non empty string, HTTP URL Link"
    let parser {.noalias.} = create(CsvParser)
    let client = newHttpClient(userAgent = agent, maxRedirects = maxRedirects, timeout = timeout)
    parser[].open(newStringStream(client.request(url, http_method, body).body), separator, quote, escape)
    client.close()
    parser[].readHeaderRow()
    while parser[].readRow():
        for column in parser.headers.items:
            var col = column
            parser[].rowEntry(col)
            result.add {column: col}
    parser[].close()
    if parser != nil: dealloc parser

proc csv2json*(csv_string: string; separator: char = ','; nl: char = '\n'): string {.exportpy.} =
    ## CSV string to JSON string Pretty-printed.
    assert csv_string.len > 0, "Argument must not be empty string"
    assert separator != nl and separator notin {' ', '\n'} and nl notin {' ', ',', ';', '\t'}
    var temp = newJObject()
    let csvlines = csv_string.split(nl)
    let firstLine = csvlines[0].split(separator)
    var headers = newSeqOfCap[string](firstLine.len)
    for item in firstLine: headers.add item
    for y in headers: temp[y] = newJArray()
    for z in csvlines[1 .. ^1]:
        var i = 0
        for x in z.split(separator): # horizontal
            if x == "true": temp[headers[i]].add newJBool(true)
            elif x == "false": temp[headers[i]].add newJBool(false)
            elif x == "null": temp[headers[i]].add newJNull()
            elif unlikely(x == "NaN"): temp[headers[i]].add newJFloat(NaN)
            elif unlikely(x == "undefined"): temp[headers[i]].add newJNull()
            elif unlikely(x == "+inf"): temp[headers[i]].add newJFloat(+1e9999)
            elif unlikely(x == "-inf"): temp[headers[i]].add newJFloat(-1e9999)
            elif (func (ss: string): bool = (for v in ss: result = v in {'.', '-', '+', 'e', '0'..'9'}))(x):
                if '.' in x and 'e' notin x:
                    temp[headers[i]].add newJFloat(parseFloat(x))
                elif 'e' in x and '.' notin x:
                    temp[headers[i]].add newJFloat(parseFloat(x))
                else:
                    temp[headers[i]].add newJInt(parseInt(x))
            else:
                temp[headers[i]].add newJString(x)
            inc i
    result = temp.pretty

proc csv2ndjson*(csv_file_path, ndjson_file_path: string, separator: char = ',',
  quote: char = '"', escape: char = '\x00') {.discardable, exportpy.} =
    ## Stream Read CSV to NDJSON https://github.com/ndjson/ndjson-spec
    let parser {.noalias.} = create(CsvParser)
    var ndjson: string
    parser[].open(csv_file_path, separator, quote, escape)
    parser[].readHeaderRow()
    while parser[].readRow():
        for column in parser[].headers.items:
            var col = column
            parser[].rowEntry(col)
            ndjson.add $(%*{column: col})
            ndjson.add '\n'
    writeFile(ndjson_file_path, $ndjson)
    parser[].close()
    if parser != nil: dealloc parser

proc csv2htmltable*(csv_file_path, html_file_path: string = "",
    separator: char = ',', quote: char = '"', escape: char = '\x00';
    header_html: string = html_table_header): string {.exportpy.} =
    ## Stream Read CSV to HTML Table file and string.
    result.add header_html
    result.add "<thead class='thead'>\n<tr>\n"
    let parser {.noalias.} = create(CsvParser)
    parser[].open(csv_file_path, separator, quote, escape)
    parser[].readHeaderRow()
    for column in parser[].headers.items:
        result.add "<th class='has-background-grey-light'>"
        result.add $column
        result.add "</th>"
    result.add "</tr>\n</thead>\n<tfoot class='tfoot has-text-primary'>\n<tr>\n"
    for column in parser[].headers.items:
        result.add "<th class='has-background-grey-light'>"
        result.add $column
        result.add "</th>"
    result.add "</tr>\n</tfoot>\n<tbody class='tbody'>\n"
    while parser[].readRow():
        result.add "<tr>"
        for column in parser[].headers.mitems:
            result.add "<td>"
            parser[].rowEntry(column)
            result.add column
            result.add "</td>"
        result.add "</tr>\n"
    result.add "</tbody>\n</table>\n</div>\n</body>\n</html>\n"
    parser[].close()
    if html_file_path.len > 0: writeFile(html_file_path, result)
    if parser != nil: dealloc parser

proc csv2markdowntable*(csv_file_path, md_file_path: string = "",
    separator: char = ',', quote: char = '"', escape: char = '\x00'): string {.exportpy.} =
    ## CSV to MarkDown Table file and string.
    let parser {.noalias.} = create(CsvParser)
    parser[].open(csv_file_path, separator, quote, escape)
    parser[].readHeaderRow()
    for column in parser[].headers.items:
        result.add '|'
        result.add ' '
        result.add $column
        result.add ' '
    result.add "|\n| "
    result.add "---- | ".repeat(parser[].headers.len)
    result.add '\n'
    while parser[].readRow():
        for column in parser[].headers.mitems:
            result.add '|'
            result.add ' '
            parser[].rowEntry(column)
            result.add column
            result.add ' '
        result.add '|'
        result.add '\n'
    parser[].close()
    if md_file_path.len > 0: writeFile(md_file_path, result)
    if parser != nil: dealloc parser

proc csv2terminal*(csv_file_path: string; column_width: Natural; separator: char = ',';
    quote: char = '"'; escape: char = '\x00') {.exportpy.} =
    ## CSV to pretty-printed colored terminal table.
    let parser {.noalias.} = create(CsvParser)
    parser[].open(csv_file_path, separator, quote, escape)
    parser[].readHeaderRow()
    const colors = [fgRed, fgGreen, fgYellow, fgBlue, fgDefault, fgMagenta, fgCyan, fgWhite, fg8Bit]
    var i = 0
    for column in parser[].headers.items:
        stdout.styledWrite bgBlack, styleUnderscore, "| " & $column & " ".repeat(column_width - len($column) + 1)
    stdout.styledWrite bgBlack, styleUnderscore, "|\n"
    while parser[].readRow():
        var c = colors[i]
        for col in parser[].headers.mitems:
            parser[].rowEntry(col)
            stdout.styledWrite bgBlack, c, styleUnderscore, "| " & alignLeft(col, column_width + 1)
        stdout.styledWrite bgBlack, c, styleUnderscore, "|\n"
        if i > 7: i = 0 else: inc i
    resetAttributes()
    parser[].close()
    if parser != nil: dealloc parser

proc csv2karax*(csv_file_path: string = "", separator: char = ',',
  quote: char = '"', escape: char = '\x00'): string {.exportpy.} =
    ## CSV to Karax HTML Table.
    result.add karax_header
    let parser {.noalias.} = create(CsvParser)
    parser[].open(csv_file_path, separator, quote, escape)
    parser[].readHeaderRow()
    for column in parser[].headers.items:
        result.add "        th:\n          text(\"\"\"" & $column & "\"\"\")\n"
    result.add "    tfoot(class=\"tfoot has-text-primary\"):\n      tr(class=\"has-background-grey-light\"):\n"
    for column in parser[].headers.items:
        result.add "        th:\n         text(\"\"\"" & $column & "\"\"\")\n"
    result.add "    tbody(class=\"tbody\"):\n"
    while parser[].readRow():
        result.add "      tr:\n"
        for column in parser[].headers.mitems:
            parser[].rowEntry(column)
            result.add "        td:\n          text(\"\"\"" & column & "\"\"\")\n"
    result.add "\n\nsetRenderer(createDom)\n"
    parser[].close()
    if parser != nil: dealloc parser

proc csv2xml*(csv_file_path: string, columns: Natural = 32767; separator: char = ',',
    quote: char = '"', escape: char = '\x00'; header_xml: string = xmlHeader): string {.exportpy.} =
    ## Stream Read CSV to XML.
    result.add header_xml
    let parser {.noalias.} = create(CsvParser)
    let temp = create(seq[XmlNode])
    let e = create(XmlNode)
    parser[].open(csv_file_path, separator, quote, escape)
    parser[].readHeaderRow()
    while parser[].readRow():
        for column in parser[].headers.mitems:
            e[] = newElement(column)
            parser[].rowEntry(column)
            e[].add newText(column)
            temp[].add e[]
    parser[].close()
    result.add $newXmlTree("csv", temp[])
    if parser != nil: dealloc parser
    if temp != nil: dealloc temp
    if e != nil: dealloc e

proc tsv2csv*(csv_file_path: string; separator1: char = '\t'; separator2: char = ','): string {.exportpy.} =
    ## Stream Read TSV to CSV, or any custom format to CSV, simple replace of "," to "\t".
    result = newStringOfCap(32767)
    for line in csv_file_path.lines: result.add line.replace(separator1, separator2)

proc diff_csvs*(csv_file_path0, csv_file_path1: string): seq[Item] {.exportpy.} =
    ## Diff 2 CSV files.
    doAssert fileExists(csv_file_path0), "File not found: " & csv_file_path0
    doAssert fileExists(csv_file_path1), "File not found: " & csv_file_path1
    diffText(readFile(csv_file_path0), readFile(csv_file_path1))

proc headers(self: var CsvParser): var seq[string] =
    return self.headers

proc get_row_entry(self: var CsvParser; entry: var string): string =
    let index = create int
    index[] = self.headers.find(entry)
    if likely(index[] >= 0): entry = self.row[index[]] else: echo "ERROR: Key not found: " & entry
    if likely(index != nil): dealloc index

    return entry

proc row(self: var CsvParser): var seq[string] =
    return self.row

export CsvParser
export open, close
export readRow, readHeaderRow
export headers, row, get_row_entry