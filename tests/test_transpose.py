from tablite import Table

def test01():
    t = Table()
    t['a'] = [1]
    t['b'] = [2]
    t['c'] = [3]
    t['d'] = [4]
    t['e'] = [5]

    t.transpose(columns=['c','d','e'] keep=['a','b'])

    assert [r for r in t.rows] == [
        [1,2,3],
        [1,2,4],
        [1,2,5],
    ]

    



