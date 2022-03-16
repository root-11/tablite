

class TextEscape(object):
    def __init__(self, openings='"({[', closures=']})"', delimiter=','):
        self.delimiter = ord(delimiter)
        self.openings = {ord(c) for c in openings}
        self.closures = {ord(c) for c in closures}

    def __call__(self, s):
        
        words = []
        L = list(s)
        
        ix,depth = 0,0
        while ix < len(L):  # TODO: Compile some REGEX for this instead.
            c = L[ix]
            if depth == 0 and c == self.delimiter:
                word, L = L[:ix], L[ix+1:]
                words.append("".join(chr(c) for c in word).encode('utf-8'))
                ix = -1
            elif c in self.openings:
                depth += 1
            elif c in self.closures:
                depth -= 1
            else:
                pass
            ix += 1

        if L:
            words.append("".join(chr(c) for c in L).encode('utf-8'))
        return words


text_escape = TextEscape(openings='"({[', closures=']})"', delimiter=',')

s = b"this,is,a,,b,(comma,sep'd),text"
L = text_escape(s)
assert L == [b"this", b"is", b"a", b"",b"b", b"(comma,sep'd)", b"text"]
