from tablite.file_reader_utils import TextEscape


def test_text_escape():
    # set up
    text_escape = TextEscape(openings='({[', closures=']})', qoute='"', delimiter=',')
    s = "this,is,a,,嗨,(comma,sep'd),\"text\""
    L = text_escape(s)
    assert L == ["this", "is", "a", "","嗨", "(comma,sep'd)", "text"]

def test2():
    text_escape = TextEscape(openings='({[', closures=']})', delimiter=',')

    s = "this,is,a,,嗨,(comma,sep'd),text"
    L = text_escape(s)
    assert L == ["this", "is", "a", "","嗨", "(comma,sep'd)", "text"]