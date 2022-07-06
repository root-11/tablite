from tablite.config import TEMPDIR


def test01():
    assert TEMPDIR.exists()