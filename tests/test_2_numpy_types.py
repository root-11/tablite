from tablite2.datatypes import *




def test_0000_dtypes():
    from datetime import datetime, timedelta
    now = datetime.now().replace(microsecond=0)

    abool = np.array([True,False])
    assert abool.dtype.name == 'bool'
    
    alargeint = np.array([-10**23,-1,0,1,10**23])
    assert alargeint.dtype.name == 'object'
    
    asmallint = np.array([-1,0,1])
    assert asmallint.dtype.name == 'int32'
    
    apositiveint = np.array([0,1,2])
    assert apositiveint.dtype.name == 'int32'
    
    astr = np.array(['a','b',"嗨"])
    assert astr.dtype.name == 'str32'

    afloat = np.array([float('-inf'),-1.0,0.0,1.0,float('inf')])
    assert afloat.dtype.name == 'float64'
    
    adate = np.array([now.date()])
    assert adate.dtype.name == 'object'
    adatetime = np.array([now])
    assert adatetime.dtype.name == 'object'
    atime = np.array([now.time()])
    assert atime.dtype.name == 'object'
    atimedelta = np.array([timedelta(days=1)])
    assert atimedelta.dtype.name == 'object'
    anone = np.array([None])
    assert anone.dtype.name == 'object'
    

    # '?'  boolean
    # 'b'  (signed) byte
    # 'B'  unsigned byte
    # 'i'  (signed) integer
    # 'u'  unsigned integer
    # 'f'  floating-point
    # 'c'  complex-floating point
    # 'm'  timedelta
    # 'M'  datetime
    # 'O'  (Python) objects
    # 'S', 'a' zero-terminated bytes (not recommended)
    # 'U' Unicode string
    # 'V' raw data (void)

    # Examples.
    # dt = np.dtype('i4')   # 32-bit signed integer
    # dt = np.dtype('f8')   # 64-bit floating-point number
    # dt = np.dtype('c16')  # 128-bit complex floating-point number
    # dt = np.dtype('a25')  # 25-length zero-terminated bytes
    # dt = np.dtype('U25')  # 25-character string

    # from python to np types.
    assert np.dtype(int) == np.dtype('int32')
    assert np.dtype(float) == np.dtype('float64')
    assert np.dtype(str) == np.dtype('<U')
    assert np.dtype(bytes) == np.dtype('S')
    assert np.dtype(type(None)) == np.dtype('O')

    print("pass")


if __name__ == "__main__":
    test_0000_dtypes()