
from tablite.memory_manager import timeout
import time


@timeout
def sleepy(n):
    time.sleep(n/10)
    return n

@timeout
def blowup():
    raise OSError()


def test_timeout_result():  
    for i in [1,3]:
        x = sleepy(i)  # tests that the timeout decorator works on sleepy
        assert x == i

