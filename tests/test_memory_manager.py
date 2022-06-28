
from tablite.memory_manager import timeout


@timeout
def sleepy(n):
    time.sleep(n/10)
    return n

@timeout
def blowup():
    raise OSError()


def test_timeout_result():
    for i in [1,3]:
        x = sleepy(i)
        assert x == i

