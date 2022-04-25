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


def text_escape(s, escape='"', sep=';'):
    """ escapes text marks using a depth measure. """
    assert isinstance(s, str)
    word, words = [], tuple()
    in_esc_seq = False
    for ix, c in enumerate(s):
        if c == escape:
            if in_esc_seq:
                if ix+1 != len(s) and s[ix + 1] != sep:
                    word.append(c)
                    continue  # it's a fake escape.
                in_esc_seq = False
            else:
                in_esc_seq = True
            if word:
                words += ("".join(word),)
                word.clear()
        elif c == sep and not in_esc_seq:
            if word:
                words += ("".join(word),)
                word.clear()
        else:
            word.append(c)

    if word:
        if word:
            words += ("".join(word),)
            word.clear()
    return words


