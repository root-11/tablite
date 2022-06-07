import re
from pathlib import Path


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


encodings = [
    'utf-32',
    'utf-16',
    'ascii',
    'utf-8',
    'windows-1252',
    'utf-7',
]


def detect_encoding(path):
    """ helper that automatically detects encoding from files. """
    assert isinstance(path, Path)
    for encoding in encodings:
        try:
            snippet = path.open('r', encoding=encoding).read(100)
            if snippet.startswith('ï»¿'):
                return 'utf-8-sig'
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            pass
    raise UnicodeDecodeError


def detect_seperator(path, encoding):
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
    text = ""
    for line in path.open('r', encoding=encoding):  # pick the first line only.
        text = line
        break
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
