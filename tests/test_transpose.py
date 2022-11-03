from tablite import Table

def test01():
    t = Table()
    t['a'] = [1]
    t['b'] = [2]
    t['c'] = [3]
    t['d'] = [4]
    t['e'] = [5]

    new = t.transpose(columns=['c','d','e'], keep=['a','b'])

    assert [r for r in new.rows] == [
        [1,2,'c',3],
        [1,2,'d',4],
        [1,2,'e',5],
    ]






