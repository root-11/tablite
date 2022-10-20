
from tablite import Table
import random
random.seed(5432)
from random import randint
from datetime import datetime
from string import ascii_uppercase
import time

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
            {"column1": 'a', "criteria":"==", 'value2':3},
            {"column1": 'b', "criteria":"==", 'value2':20},
        ], filter_type='all'
    )
    assert len(true)+len(false)==len(t)
    assert len(true) == 0, true.show()
    assert len(false) == 4, false.show()
    

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


def test_any():
    t = Table()
    t['a'] = [1,2,3,4]
    t['b'] = [10,20,30,40]

    def f(x):
        return x == 4 
    def g(x):
        return x < 20

    t2 = t.any( **{"a":f, "b":g})
    assert [r for r in t2.rows] == [[1, 10], [4, 40]]
    
    t2 = t.any(a=f,b=g)
    assert [r for r in t2.rows] == [[1, 10], [4, 40]]

    def h(x):
        return x>=2
    
    def i(x):
        return x<=30

    t2 = t.all(a=h,b=i)
    assert [r for r in t2.rows] == [[2,20], [3, 30]]


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


def test_filter_profile():
    t = Table()
    t['1'] = list(range(1000))
    t['2'] = list(randint(1,20) for _ in range(1000))
    t['3'] = list(str(i) for i in range(1000))
    t['4'] = list(randint(1,20) for _ in range(1000))
    t['5'] = list(str(i) for i in range(1000))
    t['6'] = list(randint(1,20) for _ in range(1000))
    t['7'] = list(str(i) for i in range(1000))
    t['8'] = list(randint(1,20) for _ in range(1000))
    t['9'] = list(str(i) for i in range(1000))
    t['10'] = list(range(1000))
    t['11'] = list(range(1000))
    t['12'] = list(range(1000))
    t['13'] = list(range(1000))
    t['14'] = list(range(1000))

    start = time.process_time()
    t2 = t.all(**{'2': lambda x: x >4, '4': lambda x: x>5, '6': lambda x: x>6, '8': lambda x : x>7})
    end = time.process_time()
    assert 250 < len(t2) < 265, len(t2)
    assert end-start < 2, "this shouldn't take 2 seconds."




def test_drop_na():
    t = Table()
    t['a'] = [1,2,3,None]
    t['b'] = [1,2,None,None]
    t['c'] = [1,2,3,4]
    t['d'] = [10,20,30,40]
    t2 = t.drop(None)
    assert len(t2)==2
    t3 = t.drop(30,40)
    assert len(t3) == 2
    assert t2==t3

    try:
        t.drop()
        assert False, "this should raise as it is unknown what to drop."
    except ValueError:
        assert True
    
    t4 = t.drop(None,None,None)
    assert t4 == t2