import re
from pathlib import Path
import pyexcel
import chardet


def split_by_sequence(text, sequence):
    """ helper to split text according to a split sequence. """
    chunks = tuple()
    for element in sequence:
        idx = text.find(element)
        if idx < 0:
            raise ValueError(f"'{element}' not in row")
        chunk, text = text[:idx], text[len(element) + idx:]
        chunks += (chunk,)
    chunks += (text,)  # the remaining text.
    return chunks


class TextEscape(object):
    """
    enables parsing of CSV with respecting brackets and text marks.

    Example:
    text_escape = TextEscape()  # set up the instance.
    for line in somefile.readlines():
        list_of_words = text_escape(line)  # use the instance.
        ...
    """
    def __init__(self, openings='({[', closures=']})', qoute='"', delimiter=',', strip_leading_and_tailing_whitespace=False):
        """
        As an example, the Danes and Germans use " for inches and ' for feet, 
        so we will see data that contains nail (75 x 4 mm, 3" x 3/12"), so 
        for this case ( and ) are valid escapes, but " and ' aren't.

        """
        if openings is None:
            openings = [None]
        elif isinstance(openings, str):
            self.openings = {c for c in openings}
        else:
            raise TypeError(f"expected str, got {type(openings)}")           

        if closures is None:
            closures = [None]
        elif isinstance(closures, str):
            self.closures = {c for c in closures}
        else:
            raise TypeError(f"expected str, got {type(closures)}")

        if not isinstance(delimiter, str):
            raise TypeError(f"expected str, got {type(delimiter)}")
        self.delimiter = delimiter
        self._delimiter_length = len(delimiter)
        self.strip_leading_and_tailing_whitespace= strip_leading_and_tailing_whitespace
        
        if qoute is None:
            pass
        elif qoute in openings + closures:
            raise ValueError("It's a bad idea to have qoute character appears in openings or closures.")
        else:
            self.qoute = qoute
        
        if not qoute:
            self.c = self._call1
        elif not any(openings + closures):
            self.c = self._call2
        else:
            try:
                # TODO: The regex below needs to be constructed dynamically depending on the inputs.
                self.re = re.compile("([\d\w\s\u4e00-\u9fff]+)(?=,|$)|((?<=\A)|(?<=,))(?=,|$)|(\(.+\)|\".+\")", "gmu") # <-- Disclaimer: Audrius wrote this.
                self.c = self._call3
            except TypeError:
                self.c = self._call3_slow
            
    def __call__(self,s):
        s2 = self.c(s)
        if self.strip_leading_and_tailing_whitespace:
            s2 = [w.rstrip(" ").lstrip(" ") for w in s2]
        return s2
       
    def _call1(self,s):  # just looks for delimiter.
        return s.split(self.delimiter)

    def _call2(self,s): # looks for qoutes.
        words = []
        qoute= False
        ix = 0
        while ix < len(s):  
            c = s[ix]
            if c == self.qoute:
                qoute = not qoute
            if qoute:
                ix += 1
                continue
            if c == self.delimiter:
                word, s = s[:ix], s[ix+self._delimiter_length:]
                word = word.lstrip(self.qoute).rstrip(self.qoute)
                words.append(word)
                ix = -1
            ix+=1
        if s:
            s=s.lstrip(self.qoute).rstrip(self.qoute)
            words.append(s)
        return words

    def _call3(self, s):  # looks for qoutes, openings and closures.
        return self.re.match(s)  # TODO - TEST!
    
    def _call3_slow(self, s):
        words = []
        qoute = False
        ix,depth = 0,0
        while ix < len(s):  
            c = s[ix]

            if c == self.qoute:
                qoute = not qoute

            if qoute:
                ix+=1
                continue

            if depth == 0 and c == self.delimiter:
                word, s = s[:ix], s[ix+self._delimiter_length:]
                words.append(word.rstrip(self.qoute).lstrip(self.qoute))
                ix = -1
            elif c in self.openings:
                depth += 1
            elif c in self.closures:
                depth -= 1
            else:
                pass
            ix += 1

        if s:
            words.append(s.rstrip(self.qoute).lstrip(self.qoute))
        return words



def detect_seperator(text):
    """
    :param path: pathlib.Path objects
    :param encoding: file encoding.
    :return: 1 character.
    """
    # After reviewing the logic in the CSV sniffer, I concluded that all it
    # really does is to look for a non-text character. As the separator is
    # determined by the first line, which almost always is a line of headers,
    # the text characters will be utf-8,16 or ascii letters plus white space.
    # This leaves the characters ,;:| and \t as potential separators, with one
    # exception: files that use whitespace as separator. My logic is therefore
    # to (1) find the set of characters that intersect with ',;:|\t' which in
    # practice is a single character, unless (2) it is empty whereby it must
    # be whitespace.
    seps = {',', '\t', ';', ':', '|'}.intersection(text)
    if not seps:
        if " " in text:
            return " "
        else:
            raise ValueError("separator not detected")
    if len(seps) == 1:
        return seps.pop()
    else:
        frq = [(text.count(i), i) for i in seps]
        frq.sort(reverse=True)  # most frequent first.
        return frq[0][-1]


def get_headers(path, linecount=10): 
    """
    file format	definition
    csv	    comma separated values
    tsv	    tab separated values
    csvz	a zip file that contains one or many csv files
    tsvz	a zip file that contains one or many tsv files
    xls	    a spreadsheet file format created by MS-Excel 97-2003
    xlsx	MS-Excel Extensions to the Office Open XML SpreadsheetML File Format.
    xlsm	an MS-Excel Macro-Enabled Workbook file
    ods	    open document spreadsheet
    fods	flat open document spreadsheet
    json	java script object notation
    html	html table of the data structure
    simple	simple presentation
    rst	    rStructured Text presentation of the data
    mediawiki	media wiki table
    """
    if isinstance(path, str):
        path = Path(path)
    if not isinstance(path, Path):
        raise TypeError("expected pathlib path.")
    if not path.exists():
        raise FileNotFoundError(str(path))

    delimiters = {
        '.csv': ',',
        '.tsv': '\t',
        '.txt': None,
    }

    d = {}
    if path.suffix not in delimiters:
        try:
            book = pyexcel.get_book(file_name=str(path))
            for sheet_name in book.sheet_names():
                sheet = book[sheet_name]
                stop = sheet.number_of_rows()
                d[sheet_name] = [sheet.row[i] for i in range(0, min(linecount,stop))]
                d['delimiter'] = None
            return d
        except Exception:
            pass  # it must be a raw text format.

    try:
        with path.open('rb') as fi:
            rawdata = fi.read(10000)
            encoding = chardet.detect(rawdata)['encoding']
        with path.open('r', encoding=encoding) as fi:
            lines = []
            for n, line in enumerate(fi):
                line = line.rstrip('\n')
                lines.append(line)
                if n > linecount:
                    break  # break on first
            d['delimiter'] = delimiter = detect_seperator('\n'.join(lines))
            d[path.name] = [line.split(delimiter) for line in lines]
        return d
    except Exception:
        raise ValueError(f"can't read {path.suffix}")

