
from test_multi_proc import TextEscape

text_escape = TextEscape(openings='"({[', closures=']})"', delimiter=',')

s = "this,is,a,,嗨,(comma,sep'd),text"
L = text_escape(s)
assert L == ["this", "is", "a", "","嗨", "(comma,sep'd)", "text"]
