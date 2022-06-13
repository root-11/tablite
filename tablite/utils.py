from collections import defaultdict
import math
from datetime import datetime,date,time,timedelta
from itertools import compress
from statistics import StatisticsError


def unique_name(wanted_name, list_of_names):
    """
    returns a wanted_name as wanted_name_i given a list of names
    which guarantees unique naming.
    """
    name,i = wanted_name,1
    while name in list_of_names:
        name = f"{wanted_name}_{i}"
        i+=1
    return name


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


# This list is the contact:
required_keys = {
    'min','max','mean','median','stdev','mode',
    'distinct', 'iqr_low','iqr_high','iqr','sum',
    'summary type', 'histogram'}


def summary_statistics(values,counts):
    """
    values: any type
    counts: integer

    returns dict with:
    - min (int/float, length of str, date)
    - max (int/float, length of str, date)
    - mean (int/float, length of str, date)
    - median (int/float, length of str, date)
    - stdev (int/float, length of str, date)
    - mode (int/float, length of str, date)
    - distinct (number of distinct values)
    - iqr (int/float, length of str, date)
    - sum (int/float, length of str, date)
    - histogram (2 arrays: values, count of each values)
    """
    # determine the dominant datatype:
    dtypes = defaultdict(int)
    most_frequent, most_frequent_dtype = 0, int
    for v,c in zip(values, counts):
        dtype = type(v)
        total = dtypes[dtype] + c
        dtypes[dtype] = total
        if total > most_frequent:
            most_frequent_dtype = dtype
            most_frequent = total
    
    most_frequent_dtype = max(dtypes, key=dtypes.get)
    mask = [type(v)==most_frequent_dtype for v in values]
    v = list(compress(values, mask))
    c = list(compress(counts, mask))
    
    f = summary_methods.get(most_frequent_dtype, int)        
    result = f(v,c)
    result['distinct'] = len(values)
    result['summary type'] = most_frequent_dtype.__name__
    result['histogram'] = [values, counts]
    assert set(result.keys()) == required_keys, "Key missing!"
    return result


def _numeric_statistics_summary(v,c):
    VC = [[v,c] for v,c in zip(v, c)]
    VC.sort()
    
    total_val, mode, median, total_cnt =  0, None, None, sum(c)
    
    max_cnt, cnt_n = -1, 0
    mn,cstd = 0, 0.0
    iqr25 = total_cnt * 1/4
    iqr50 = total_cnt * 1/2
    iqr75 = total_cnt * 3/4
    iqr_low, iqr_high = 0, 0
    vx_0 = None
    vmin,vmax = VC[0][0], VC[-1][0]

    for vx, cx in VC:
        cnt_0 = cnt_n
        cnt_n += cx

        if cnt_0 < iqr25 < cnt_n:  # iqr 25% 
            iqr_low = vx
        elif cnt_0 == iqr25:
            _,delta = divmod(1*(total_cnt-1), 4)
            iqr_low = (vx_0 * (4-delta) + vx * delta) / 4

        # median calculations
        if cnt_n-cx < iqr50 < cnt_n:
            median = vx
        elif cnt_0 == iqr50:
            _,delta = divmod(2*(total_cnt-1), 4)
            median = (vx_0 * (4-delta) + vx * delta) / 4

        if cnt_0 < iqr75 < cnt_n:  # iqr 75%
            iqr_high = vx
        elif cnt_0 == iqr75:
            _,delta = divmod(3*(total_cnt-1), 4)
            iqr_high = (vx_0 * (4-delta) + vx * delta) / 4
        
        # stdev calulations
        # cnt = cnt_n  # self.count += 1
        dt = cx * (vx-mn) # dt = value - self.mean
        mn += dt / cnt_n  # self.mean += dt / self.count
        cstd += dt * (vx-mn)  #self.c += dt * (value - self.mean)

        # mode calculations
        if cx > max_cnt:
            mode,max_cnt = vx,cx

        total_val += vx*cx
        vx_0 = vx
    
    var = cstd / (cnt_n-1) if cnt_n > 1 else 0
    stdev = var**(1/2) if cnt_n > 1 else 0

    d = {
        'min': vmin,
        'max': vmax,
        'mean': total_val / (total_cnt if total_cnt >= 1 else None),
        'median': median,
        'stdev': stdev,
        'mode': mode,
        'iqr_low': iqr_low,
        'iqr_high': iqr_high,
        'iqr': iqr_high - iqr_low,
        'sum': total_val,
    }
    return d


def _none_type_summary(v,c):
    return {k:'n/a' for k in required_keys}


def _boolean_statistics_summary(v,c):
    v = [int(vx) for vx in v]
    d = _numeric_statistics_summary(v,c)
    for k,v in d.items():
        if k in {'mean','stdev','sum','iqr_low','iqr_high','iqr'}:
            continue
        elif v == 1:
            d[k] = True
        elif v == 0:
            d[k] = False
        else:
            pass
    return d


def _timedelta_statistics_summary(v,c):
    v= [vx.days + v.seconds/(24*60*60) for vx in v]
    d = _numeric_statistics_summary(v,c)
    for k in d.keys():
        d[k] = timedelta(d[k])
    return d


def _datetime_statistics_summary(v,c):
    v = [vx.timestamp() for vx in v]
    d = _numeric_statistics_summary(v,c)
    for k in d.keys():
        if k in {'stdev','iqr','sum'}:
            d[k] = f"{d[k]/(24*60*60)} days"
        else:
            d[k] = datetime.fromtimestamp(d[k])
    return d


def _time_statistics_summary(v,c):
    v = [ sum(t.hour * 60 * 60,t.minute * 60, t.second, t.microsecond/1e6) for t in v]
    d = _numeric_statistics_summary(v,c)
    for k in d.keys():
        if k in {'min','max','mean','median'}:
            timestamp = d[k]
            hours = timestamp // (60 * 60)
            timestamp -= hours * 60 * 60
            minutes = timestamp // 60
            timestamp -= minutes * 60
            d[k] = time.fromtimestamp(hours,minutes,timestamp)
        elif k in {'stdev','iqr','sum'}:
            d[k] = f"{d[k]} seconds"
        else:
            pass
    return d


def _date_statistics_summary(v,c):
    v = [datetime(d.year,d.month,d.day,0,0,0).timestamp() for d in v]
    d = _numeric_statistics_summary(v,c)
    for k in d.keys():
        if k in {'min','max','mean','median'}:
            d[k] = date(*datetime.fromtimestamp(d[k]).timetuple()[:3])
        elif k in {'stdev','iqr','sum'}:
            d[k] = f"{d[k]/(24*60*60)} days"
        else:
            pass
    return d


def _string_statistics_summary(v,c):
    vx = [len(x) for x in v]
    d = _numeric_statistics_summary(vx,c)
    for k in d.keys():
        d[k] = f"{d[k]} characters"
    return d


summary_methods = {
        bool: _boolean_statistics_summary,
        int: _numeric_statistics_summary,
        float: _numeric_statistics_summary,
        str: _string_statistics_summary,
        date: _date_statistics_summary,
        datetime: _datetime_statistics_summary,
        time: _time_statistics_summary,
        timedelta: _timedelta_statistics_summary,
        type(None): _none_type_summary,
    }

