import sys
from time import process_time
from random import shuffle, randint
from tablite.stored_list import StoredList, Page


def test_mutability():
    p = Page()
    p.append(3)
    p.extend((2, 1))
    assert 2 in p

    p2 = p.copy()
    assert isinstance(p2, list)

    data = p.store()
    assert p.len == len(data)
    assert p.loaded is False
    assert "stored" in str(p)

    p.load(data)
    assert p.len == len(data)
    assert p.loaded is True
    assert "loaded" in str(p)


def test_attr_comparison():
    A = list(range(10))
    B = StoredList()
    a = set(dir(A))
    a.discard('__class_getitem__')  # PEP 585 thing for type hinting (python 3.9+)
    b = set(dir(B))
    assert a.issubset(b), [i for i in a if i not in b]


def test_info():
    v = sys.version_info
    if (v.major, v.minor) != (3,7):
        return
    B = StoredList(page_size=3, data=list(range(10)))
    assert B.__sizeof__() == 256  # size in ram
    assert B.disk_size() == 51, B.disk_size()  # size on disk.
    C = StoredList(data=list(range(10_000)))
    assert C.__sizeof__() == 87616, C.__sizeof__()  # contains the cache.
    C.clear_cache()  # drops the cached entries to disk
    assert C.__sizeof__() == 56, C.__sizeof__()  # The call to C.disk_size stores
    assert C.disk_size() == 29770, C.disk_size()  # drop to disk, and measures the bytes put to disk.


def test_basic_stored_list():
    A = list(range(1000))
    B = StoredList(page_size=200)
    B.extend(A)
    C = StoredList(page_size=2000)
    C.extend(A)
    assert len(B) == len(A)  # compare storedlist with list - short page size
    assert len(C) == len(A)  # compare storedlist with list - long page size.
    assert len(B) == len(C)  # compare same classes different page size.

    # construct storedlist from list at init.
    D = StoredList(page_size=randint(10, 150), data=A)
    assert len(D) == len(A)

    # construct storedlist from storedlist at init.
    E = StoredList(page_size=randint(10, 150), data=B)
    assert len(E) == len(A)

    F = StoredList(page_size=139, data=C)
    # resize page size based on odd division of former length of pages:
    F.page_size = 40
    assert len(F) == len(A)


def test_iteration():
    A = list(range(10))
    B = StoredList(page_size=3, data=A)
    assert len(A) == len(A)
    assert [a == b for a, b in zip(A, B)]  # compare values in addition to equal operator
    R = iter(A)
    S = iter(B)
    assert [a == b for a, b in zip(R, S)]
    assert [a == b for a, b in zip(reversed(A), reversed(B))]  # reversed on storedlist over multiple pages.


def test_sort():
    A = list(range(10))
    shuffle(A)

    B = StoredList(page_size=3, data=A)
    C = StoredList(page_size=len(A)*2, data=A)
    A.reverse()
    assert A != B
    assert C == B  # assure that data in C is decoupled from A.
    B.reverse()
    assert A == B
    assert C != B
    C.reverse()
    assert C == A
    assert C == B

    A.sort()
    assert A != B
    assert B == C
    B.sort()
    assert A == B
    assert B != C
    C.sort()
    assert A == C
    assert B == C


def test_eq_operator():
    A = list(range(1000))
    B = StoredList(page_size=200, data=A)
    C = StoredList(page_size=2000, data=A)
    assert A == B  # list == storedlist
    assert B == C  # storedlist == storedlist
    assert C == A  # storedlist == list


def test_indexing_and_slicing_methods():
    A = list(range(1000))
    B = StoredList(page_size=200, data=A)
    C = StoredList(page_size=2000, data=A)
    assert A.index(0) == B.index(0) == C.index(0)  # get value in slice 0
    assert A.index(991) == B.index(991) == C.index(991)  # get value in slice N
    assert A[0] == B[0] == C[0]  # get slice from integer.
    assert A[-1] == B[-1] == C[-1]  # get slice from integer.
    b, c = B[:], C[:]  # get slice with 3 None's
    assert b == c
    assert A[:] == B[:] == C[:]  # nothing declared
    assert A[:5] == B[:5] == C[:5]  # stop declared
    assert A[5:] == B[5:] == C[5:]  # start declared
    assert A[3:11] == B[3:11] == C[3:11]  # start and stop declared
    assert A[5:15:2] == B[5:15:2] == C[5:15:2]  # steps > 1
    assert A[15:5:-2] == B[15:5:-2] == C[15:5:-2]  # steps < -1
    # positive steps across multiple pages
    assert B[211:219:3] == A[211:219:3]  # data in B page 2
    assert B[211:419:11] == A[211:419:11]   # data in B pages 2 and 3
    assert B[211:419:111] == A[211:419:111]  # data in B in pages 2 and 3
    assert B[211:819:311] == A[211:819:311]   # data in B pages 2 and 4, but not in 3.
    # negative steps across multiple pages
    assert B[219:211:-3] == A[219:211:-3]
    assert B[419:211:-11] == A[419:211:-11]
    assert B[419:211:-111] == A[419:211:-111]
    assert B[819:211:-311] == A[819:211:-311]

    a, b, c = A[::-1], B[::-1], C[::-1]  # reversed copies.
    assert a == b
    assert a == c
    assert c == b


def test_remove_contains_append():
    A = list(range(1000))
    B = StoredList(page_size=200, data=A)
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


def test_copy_math_and_count():
    A = list(range(100))
    B = StoredList(page_size=31, data=A)

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
    A = StoredList(page_size=200)
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
    m = 100
    for i in range(m):
        ix = randint(0, n)
        v = A[ix]
    end = process_time()
    print(end - start, "for ", m, "lookups is", 1000 * (end - start) / m , "msec per lookup")


def test_sort_performance1():
    A = StoredList()
    A.page_size = 5 * 3
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
    n = StoredList.default_page_size * 10
    data = list(range(n))
    shuffle(data)

    A = StoredList()
    A.extend(data)

    start = process_time()
    A.sort()
    print("sort on disk allows {:,.0f} items/sec".format(n/(process_time() - start)))

    start = process_time()
    data.sort()
    print("sort in memory allows {:,.0f} items/sec".format(n/(process_time() - start)))
    assert A == data


def test_sort_performance3():
    n = 200_000
    data = list(range(n)) * 10
    shuffle(data)

    A = StoredList()
    A.extend(data)

    start = process_time()
    A.sort()
    print("sort on disk allows {:,.0f} items/sec".format(n / (process_time() - start)))

    start = process_time()
    data.sort()
    print("sort in memory allows {:,.0f} items/sec".format(n / (process_time() - start)))
    assert A == data

