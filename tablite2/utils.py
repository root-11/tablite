import math


def isiterable(item):
    """
    Determines if an item is iterable.
    """
    # only valid way to check that a variable is iterable.
    # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    try:   
        iter(item)
        return True
    except TypeError:
        return False


def intercept(A,B):
    """
    enables calculation of the intercept of two range objects.
    Used to determine if a datablock contains a slice.
    
    A: range
    B: range
    
    returns: range as intercept of ranges A and B.
    """
    if not isinstance(A, range):
        raise TypeError
    if A.step < 0: # turn the range around
        A = range(A.stop, A.start, abs(A.step))

    if not isinstance(B, range):
        raise TypeError
    if B.step < 0:  # turn the range around
        B = range(B.stop, B.start, abs(B.step))
    
    boundaries = [A.start, A.stop, B.start, B.stop]
    boundaries.sort()
    a,b,c,d = boundaries
    if [A.start, A.stop] in [[a,b],[c,d]]:
        return range(0) # then there is no intercept
    # else: The inner range (subset) is b,c, limited by the first shared step.
    A_start_steps = math.ceil((b - A.start) / A.step)
    A_start = A_start_steps * A.step + A.start

    B_start_steps = math.ceil((b - B.start) / B.step)
    B_start = B_start_steps * B.step + B.start

    if A.step == 1 or B.step == 1:
        start = max(A_start,B_start)
        step = B.step if A.step==1 else A.step
        end = c
    else:
        intersection = set(range(A_start, c, A.step)).intersection(set(range(B_start, c, B.step)))
        if not intersection:
            return range(0)
        start = min(intersection)
        end = max(c, max(intersection))
        intersection.remove(start)
        step = min(intersection) - start
    
    return range(start, end, step)


def normalize_slice(length, item=None):  # There's an outdated version sitting in utils.py
    """
    helper: transforms slice into range inputs
    returns start,stop,step
    """
    if item is None:
        item = slice(0, length, 1)
    assert isinstance(item, slice)
    
    stop = length if item.stop is None else item.stop
    start = 0 if item.start is None else length + item.start if item.start < 0 else item.start
    start, stop = min(start,stop), max(start,stop)
    step = 1 if item.step is None else item.step

    return start, stop, step