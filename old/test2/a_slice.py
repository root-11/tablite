import math


def intercept(A,B):
    assert isinstance(A, range)
    if A.step < 0: # turn the range around
        A = range(A.stop, A.start, abs(A.step))
    assert isinstance(B, range)
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

    intersection = set(range(A_start, c, A.step)).intersection(set(range(B_start, c, B.step)))
    if not intersection:
        return range(0)
    start = min(intersection)
    end = max(intersection)
    intersection.remove(start)
    step = min(intersection) - start
    
    return range(start, end+1, step)


A = range(500,700,3)
B = range(520,700,3)
C = range(10,1000,30)

assert intercept(A,C) == range(0)
assert intercept(B,C) == range(520,671,30)


A = range(500_000, 700_000, 1)
B = range(10, 10_000_000, 1000)

print(len(intercept(A,B)))
