from tablite import Table
import random
random.seed(5432)
from datetime import datetime
from string import ascii_uppercase

import pytest


@pytest.fixture(autouse=True) # this resets the HDF5 file for every test.
def refresh():
    Table.reset_storage()
    yield


def test_filter_all_1():
    t = Table()
    t['a'] = [1,2,3,4]
    t['b'] = [10,20,30,40]
    true,false = t.filter(
        [
            {"column1": 'a', "criteria":">=", 'value2':3},
            {"column1": 'b', "criteria":"<=", 'value2':20},
        ], filter_type='all'
    )
    assert len(true)+len(false)==len(t)
    assert len(true) == 0, true.show()
    assert len(false) == 4, false.show()

    t2 = t.all(a=3).all(b=20)  # same output but different syntax.
    assert true == t2
    
    true1, false1 = t.filter('all((a>=3, b<=20))')
    true2, false2 = t.filter('a>=3 and b<=20')

    assert t2 == true == true1 == true2
    assert false == false1 == false2


def test_filter_any_1():
    t = Table()
    t['a'] = [1,2,3,4]
    t['b'] = [10,20,30,40]
    true,false = t.filter(
        [
            {"column1": 'a', "criteria":"==", 'value2': 3},
            {"column1": 'b', "criteria":"==", 'value2': 20},
        ], filter_type='any'
    )
    assert len(true)+len(false)==len(t)
    assert len(true)==2, true.show()
    assert len(false)==2, false.show()


def test_filter_any_2():
    t = Table()
    t['a'] = [1,2,3,4]
    t['b'] = [10,20,30,40]
    true,false = t.filter(
        [
            {"column1": 'a', "criteria":"==", 'value2': 3},
            {"column1": 'b', "criteria":">", 'value2': 20},
        ], filter_type='any'
    )
    assert len(true)+len(false)==len(t)
    assert len(true)==2, true.show()
    assert len(false)==2, false.show()

def test_filter_any_3():
    t = Table()
    t['a'] = [1,2,3,4]
    t['b'] = [10,20,30,40]
    true,false = t.filter(
        [
            {"column1": 'a', "criteria":"==", 'value2': 3},
            {"column1": 'a', "criteria":"==", 'value2': 4},
        ], filter_type='any'
    )
    assert len(true)+len(false)==len(t)
    assert len(true)==2, true.show()
    assert len(false)==2, false.show()


def test_filter():
    t = Table()
    rows = 100_000 
    t["#"]= list(range(1,rows+1))
    t["1"]= [random.randint(18_778_628_504, 2277_772_117_504) for _ in range(rows)]
    t["2"] = [datetime.fromordinal(random.randint(738000, 738150)) for _ in range(rows)]
    t["3"] = [random.randint(50000, 51000) for _ in range(rows)]
    t["4"] = [int(i%2==0) for i in range(rows)]  #[random.randint(0, 1) for i in range(rows)]
    t["5"] = [f"C{random.randint(1, 5)}-{random.randint(1, 5)}" for i in range(rows)]
    t["6"] = ["".join(random.choice(ascii_uppercase) for _ in range(3)) for i in range(rows)]
    t["7"] = [random.choice(['None', '0°', '6°', '21°']) for i in range(rows)]
    t["8"] = [random.choice(['ABC', 'XYZ', ""]) for i in range(rows)]
    t["9"] = [random.uniform(0.01, 2.5) for i in range(rows)]
    t["10"] = [random.uniform(0.01, 2.5) for i in range(rows)]
    t["11"] =[f"{random.uniform(0.1, 25)}" for i in range(rows)]
    t.show()

    a,b = t.filter(
        [
            {'column1': '4', 'criteria': "==", 'value2':0},
            {'column1': '4', 'criteria': "==", 'value2':0},
            {'column1': '4', 'criteria': "==", 'value2':0}
        ],
        filter_type='all'
    )
    assert len(a) + len(b) == len(t)

    assert set(a['4'].unique()) == {0}
    assert set(b['4'].unique()) == {1}
    
    a,b = t.filter(
        [
            {'column1': '9', 'criteria': '>', 'column2': '10'},
        ]
    )
    a9 = list(a['9'])
    a10 = list(a['10'])
    assert all(i>j for i,j in zip(a9,a10))
    assert all(i<=j for i,j in b['9','10'].rows)
    assert len(a) + len(b) == len(t)

    a,b = t.filter(
        [
            {'column1': '7', 'criteria': '==', 'value2': '6°'},
            {'column1': '4', 'criteria': "==", 'value2':0}
        ],
        filter_type='any'
    )
    for row in a.rows:
        assert row[4]==0 or row[7]=='6°'
    for row in b.rows:
        assert row[4]!=0 and row[7]!='6°'
    
    assert len(a) + len(b) == len(t)


