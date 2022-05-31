import math
from datetime import datetime,date,time,timedelta


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
    - histogram
    """
    VC = [[v,c] for v,c in zip(values, counts)]
    VC.sort()
    vmin,vmax = VC[0][0], VC[-1][0]
    
    total_val, mode, median, total_cnt =  0, None, None, sum(counts)
    
    max_cnt, cnt_n =-1, 0
    mn,c = 0, 0.0
    iqr25 = total_cnt * 1/4
    iqr50 = total_cnt * 1/2
    iqr75 = total_cnt * 3/4
    iqr_low, iqr_high = 0, 0
    vx_0 = None
    for vx, cx in VC:
        if vx is None:
            continue
        elif isinstance(vx, (date, datetime, time, timedelta)):
            continue
        elif isinstance(vx, str):
            vx = len(str)
        elif isinstance(vx,(float,int)):
            pass
        else:
            raise TypeError(vx)

        cnt_0 = cnt_n
        cnt_n += cx

        if cnt_0 < iqr25 < cnt_n:
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

        if cnt_0 < iqr75 < cnt_n:
            iqr_high = vx
        elif cnt_0 == iqr75:
            _,delta = divmod(3*(total_cnt-1), 4)
            iqr_high = (vx_0 * (4-delta) + vx * delta) / 4
        
        # stdev calulations
        # cnt = cnt_n  # self.count += 1
        dt = cx * (vx-mn) # dt = value - self.mean
        mn += dt / cnt_n  # self.mean += dt / self.count
        c += dt * (vx-mn)  #self.c += dt * (value - self.mean)

        # mode calculations
        if cx > max_cnt:
            mode,max_cnt = vx,cx

        total_val += vx*cx
        vx_0 = vx
    
    if cnt_n - 1 > 1:
        var = c / (cnt_n-1)
        stdev = var**(1/2)
    
    d = {
        'min': vmin,
        'max': vmax,
        'mean': total_val / total_cnt,
        'median': median,
        'stdev': stdev,
        'mode': mode,
        'distinct': len(values),
        'iqr_low': iqr_low,
        'iqr_high': iqr_high,
        'iqr': iqr_high-iqr_low,
        'sum': total_val,
        'histogram': [values,counts]
    }
    return d
