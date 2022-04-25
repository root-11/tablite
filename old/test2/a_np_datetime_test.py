

from multiprocessing.sharedctypes import Value
import numpy as np

a = np.datetime64(b'2014-12-12')   # Y-M-D
b = np.datetime64(b'2014-12-12T12:23:34')   # Y-M-DTh:m:s
c = np.datetime64(b'2014-12-12T12:23:34.123456')   # Y-M-DTh:m:s.u
d = np.datetime64('2014-12-12T12:23:34.123456')   # Y-M-DTh:m:s.u

def dt(s='12:34:56.654323'):
    if "." in s:
        v = sum(int(v)*m for v,m in zip(s.split(":"), [3600,60,1]))
    
    return 

dts = dt(s='12:34:56.6523')
e = np.timedelta64(dts, 'ms')  # h:m:s

# e = np.datetime64(b'0000-01-01T12:23:34')  # h:m:s
print(a,b,c,d,e)
print(a.dtype,b.dtype,c.dtype, d.dtype, e.dtype)

print(a,"+", e, "=", a+e)


# Dates, datetimes and times almost work.
# try:
#     data[name][line_no] = Value
# except:
#     data[name] = np.array(self, dtype=bytes)

