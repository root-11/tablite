import time  # python builtin first.
import random
import re
import hashlib
import pathlib

import h5py  # required packages
from tqdm import trange
from numpy import np

from tablite2.settings import HDF5_IMPORT_ROOT  # local modules.


class TextEscape(object):
    """
    enables parsing of CSV with respecting brackets and text marks.

    Example:
    text_escape = TextEscape()  # set up the instance.
    for line in somefile.readlines():
        list_of_words = text_escape(line)  # use the instance.
        ...
    """
    def __init__(self, openings='({[', closures=']})', qoute='"', delimiter=','):
        """
        As an example, the Danes and Germans use " for inches and ' for feet, 
        so we will see data that contains nail (75 x 4 mm, 3" x 3/12"), so 
        for this case ( and ) are valid escapes, but " and ' aren't.

        """
        if openings is None:
            pass
        elif isinstance(openings, str):
            self.openings = {c for c in openings}
        else:
            raise TypeError(f"expected str, got {type(openings)}")           

        if closures is None:
            pass
        elif isinstance(closures, str):
            self.closures = {c for c in closures}
        else:
            raise TypeError(f"expected str, got {type(closures)}")
    
        if not isinstance(delimiter, str):
            raise TypeError(f"expected str, got {type(delimiter)}")
        self.delimiter = delimiter
        self._delimiter_length = len(delimiter)
        
        if qoute is None:
            pass
        elif qoute in openings or qoute in closures:
            raise ValueError("It's a bad idea to have qoute character appears in openings or closures.")
        else:
            self.qoute = qoute
        
        if not qoute:
            self.c = self._call1
        elif not openings + closures:
            self.c = self._call2
        else:
            # TODO: The regex below needs to be constructed dynamically depending on the inputs.
            self.re = re.compile("([\d\w\s\u4e00-\u9fff]+)(?=,|$)|((?<=\A)|(?<=,))(?=,|$)|(\(.+\)|\".+\")", "gmu") # <-- Disclaimer: Audrius wrote this.
            self.c = self._call3

    def __call__(self,s):
        return self.c(s)
       
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
                words.append(word)
                ix = -1
            ix+=1
        if s:
            words.append(s)
        return words

    def _call3(self, s):  # looks for qoutes, openings and closures.
        return self.re.match(s)  # TODO - TEST!
        # words = []
        # qoute = False
        # ix,depth = 0,0
        # while ix < len(s):  
        #     c = s[ix]

        #     if c == self.qoute:
        #         qoute = not qoute

        #     if qoute:
        #         ix+=1
        #         continue

        #     if depth == 0 and c == self.delimiter:
        #         word, s = s[:ix], s[ix+self._delimiter_length:]
        #         words.append(word)
        #         ix = -1
        #     elif c in self.openings:
        #         depth += 1
        #     elif c in self.closures:
        #         depth -= 1
        #     else:
        #         pass
        #     ix += 1

        # if s:
        #     words.append(s)
        # return words



def detect_seperator(text):
    """
    After reviewing the logic in the CSV sniffer, I concluded that all it
    really does is to look for a non-text character. As the separator is
    determined by the first line, which almost always is a line of headers,
    the text characters will be utf-8,16 or ascii letters plus white space.
    This leaves the characters ,;:| and \t as potential separators, with one
    exception: files that use whitespace as separator. My logic is therefore
    to (1) find the set of characters that intersect with ',;:|\t' which in
    practice is a single character, unless (2) it is empty whereby it must
    be whitespace.
    """
    seps = {',', '\t', ';', ':', '|'}.intersection(text)
    if not seps:
        if " " in text:
            return " "
    else:
        frq = [(text.count(i), i) for i in seps]
        frq.sort(reverse=True)  # most frequent first.
        return {k:v for k,v in frq}


def text_reader(source, destination, columns, 
                newline, delimiter=',', first_row_has_headers=True, qoute='"',
                text_escape_openings='', text_escape_closures='',
                start=None, limit=None,
                encoding='utf-8'):
    """ PARALLEL TASK FUNCTION
    reads columnsname + path[start:limit] into hdf5.

    source: csv or txt file
    destination: available filename
    
    columns: column names or indices to import

    newline: '\r\n' or '\n'
    delimiter: ',' ';' or '|'
    first_row_has_headers: boolean
    text_escape_openings: str: default: "({[ 
    text_escape_closures: str: default: ]})" 

    start: integer: The first newline after the start will be start of blob.
    limit: integer: appx size of blob. The first newline after start of 
                    blob + limit will be the real end.

    encoding: chardet encoding ('utf-8, 'ascii', ..., 'ISO-22022-CN')
    root: hdf5 root, cannot be the same as a column name.
    """
    if isinstance(source, str):
        source = pathlib.Path(source)
    if not isinstance(source, pathlib.Path):
        raise TypeError
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    if isinstance(destination, str):
        destination = pathlib.Path(destination)
    if not isinstance(destination, pathlib.Path):
        raise TypeError

    if not isinstance(columns, dict):
        raise TypeError
    if not all(isinstance(name,str) for name in columns):
        raise ValueError

    root=HDF5_IMPORT_ROOT
    
    # declare CSV dialect.
    text_escape = TextEscape(text_escape_openings, text_escape_closures, qoute=qoute, delimiter=delimiter)

    if first_row_has_headers:
        with source.open('r', encoding=encoding) as fi:
            for line in fi:
                line = line.rstrip('\n')
                break  # break on first
        headers = text_escape(line)  
        indices = {name: headers.index(name) for name in columns}
    else:
        indices = {name: int(name) for name in columns}

    # find chunk:
    # Here is the problem in a nutshell:
    # --------------------------------------------------------
    # bs = "this is my \n text".encode('utf-16')
    # >>> bs
    # b'\xff\xfet\x00h\x00i\x00s\x00 \x00i\x00s\x00 \x00m\x00y\x00 \x00\n\x00 \x00t\x00e\x00x\x00t\x00'
    # >>> nl = "\n".encode('utf-16')
    # >>> nl in bs
    # False
    # >>> nl.decode('utf-16') in bs.decode('utf-16')
    # True
    # --------------------------------------------------------
    # This means we can't read the encoded stream to check if in contains a particular character.

    # Fetch the decoded text:
    with source.open('r', encoding=encoding) as fi:
        fi.seek(0, 2)
        filesize = fi.tell()
        fi.seek(start)
        text = fi.read(limit)
        begin = text.index(newline)
        text = text[begin+len(newline):]

        snipsize = min(1000,limit)
        while fi.tell() < filesize:
            remainder = fi.read(snipsize)  # read with decoding
            
            if newline not in remainder:  # decoded newline is in remainder
                text += remainder
                continue
            ix = remainder.index(newline)
            text += remainder[:ix]
            break

    # read rows with CSV reader.
    data = {h: [] for h in indices}
    for row in text.split(newline):
        fields = text_escape(row)
        if fields == [""] or fields == []:
            break
        for header,index in indices.items():
            data[header].append(fields[index])

    # turn rows into columns.    
    for name, dtype in columns.items():
        arr = np.array(data[name], dtype=dtype)
        if arr.dtype == 'O':  # hdf5 doesn't like 'O' type
            data[name] = np.array(arr[:], dtype='S')  
        else:
            data[name] = arr

    # store as HDF5
    for _ in range(100):  # overcome any IO blockings.
        try:
            with h5py.File(destination, 'a') as f:
                for name, arr in data.items():
                    f.create_dataset(f"/{root}/{name}/{start}", data=arr)  # `start` declares the slice id which order will be used for sorting
            return
        except OSError as e:
            time.sleep(random.randint(10,200)/1000)
    raise TimeoutError("Couldn't connect to OS.")



def sha256sum(path, column_name):
    """ PARALLEL TASK FUNCTION
    calculates the sha256sum for a HDF5 column when given a path.
    """
    with h5py.File(path,'r') as f:  # 'r+' in case the sha256sum is missing.
        m = hashlib.sha256()  # let's check if it really is new data...
        dset = f[f"/{column_name}"]
        step = 100_000
        desc = f"Calculating missing sha256sum for {column_name}: "
        for i in trange(0, len(dset), step, desc=desc):
            chunk = dset[i:i+step]
            m.update(chunk.tobytes())
        sha256sum = m.hexdigest()
        # f[f"/{column_name}"].attrs['sha256sum'] = sha256sum

    for _ in range(100):  # overcome any IO blockings.
        try:
            with h5py.File(path, 'a') as f:
                f[f"/{column_name}"].attrs['sha256sum'] = sha256sum
            return
        except OSError as e:
            time.sleep(random.randint(2,100)/1000)
    raise TimeoutError("Couldn't connect to OS.")
