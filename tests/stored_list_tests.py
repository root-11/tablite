from time import process_time
from tablite import StoredList
from random import shuffle, randint


def test_basic_stored_list():
    A = list(range(100))
    B = StoredList()
    B.extend(A)
    assert len(B) == len(A)
    # assert len(B) == len(A)
    assert B.index(0) == A.index(0)
    assert B.index(99) == A.index(99)
    assert B[0] == A[0]
    assert B[-1] == A[-1]
    assert B[:] == A[:]
    assert B[:5] == A[:5]
    assert B[5:] == A[5:]
    assert B[3:11] == A[3:11]
    # assert B[::-1] == A[::-1], (B[::-1], A[::-1])
    # assert B[15:5:-2] == A[15:5:-2]
    assert [a == b for a, b in zip(A, B)]
    # assert [a == b for a, b in zip(reversed(A), reversed(B))]
    A.remove(44)
    assert A != B
    B.remove(44)
    assert A == B
    A.append(44)
    assert B != A
    B.append(44)
    assert B == A
    assert 44 in A
    assert 44 in B

    assert 4444 not in A
    assert 4444 not in B

    A.reverse()
    assert A != B
    B.reverse()
    assert A == B
    A.sort()
    assert A != B
    B.sort()
    assert A == B

    # copy
    # count
    A2 = A * 2
    B2 = B * 2
    assert A2 == B2
    A2 *= 2
    B2 *= 2

    del A2[44]
    assert A2 != B2
    del B2[44]
    assert A2 == B2

    assert A2.count(44) == B2.count(44)
    for ix in [-1, 0, 1]:
        A.insert(ix, 101)
        B.insert(ix, 101)
        assert A.count(101) == B.count(101)
    for i in range(A.count(101)):
        A.remove(101)
        B.remove(101)
        assert A.count(101) == B.count(101)

    last_A_value = A.pop()
    last_B_value = B.pop()
    assert last_A_value == last_B_value


def test_memory_footprint():
    A = StoredList(buffer_limit=200)
    t = 0
    for i in range(int(1e3)):
        A.append(i)
        t += i
    t2 = 0
    for i in A:
        t2 += i
    assert t == t2


def test_index():
    n = int(1e6)
    A = StoredList()
    A.extend(list(range(n)))
    start = process_time()
    m = 1000
    for i in range(m):
        ix = randint(0, n)
        v = A[ix]
    end = process_time()
    print(end - start, "for ", m, "lookups is", 1000 * (end - start) / m , "msec per lookup")


def test_sort_performance1():
    A = StoredList()
    A.buffer_limit = 5 * 3
    A.extend([1] * 5 + [2] * 5 + [3] * 5 + [4] * 5)
    A.extend([1] * 5 + [3] * 5 + [5] * 5 + [7] * 5)
    A.extend([4] * 5 + [5] * 5 + [6] * 5 + [7] * 5)
    data = [v for v in A]
    data.sort()
    A.sort()
    assert data == A
    assert A == data
    A.sort(reverse=True)
    data.sort(reverse=True)
    assert A == data


def test_sort_performance2():
    A = StoredList()
    n = A.buffer_limit * 10
    data = list(range(n))
    shuffle(data)
    A.extend(data)

    start = process_time()
    A.sort()
    print("sort on disk allows {:,.0f} items/sec".format(n/(process_time() - start)))

    start = process_time()
    data.sort()
    print("sort in memory allows {:,.0f} items/sec".format(n/(process_time() - start)))
    assert A == data


