from tablite2.tasks.text_reader import TextEscape


def test_text_escape():
    # set up
    text_escape = TextEscape(openings='({[', closures=']})', qoute='"', delimiter=',')
    s = "this,is,a,,嗨,(comma,sep'd),\"text\""
    # use
    L = text_escape(s)
    assert L == ["this", "is", "a", "","嗨", "(comma,sep'd)", "\"text\""]

if __name__ == "__main__":
    test_text_escape()