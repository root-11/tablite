from tablite.file_reader_utils import TextEscape, get_headers


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

def test_pyexcel():
    import pathlib
    folder = pathlib.Path(__file__).parent / 'data'
    for fname in folder.iterdir():
        d = get_headers(fname)
        assert isinstance(d, dict)
        print(fname)
        print(d)
