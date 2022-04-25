import os
from  collections import defaultdict
import numpy as np
from pathlib import Path
import h5py
import multiprocessing

IMPORT_ROOT = "__import"  # the hdf5 base name for imports. f.x. f['/__h5_import/column A']
CACHE_DIR = os.getcwd()
CACHE_FILE = " tablite_cache.hdf5"
CACHE_PATH = Path(CACHE_DIR) / CACHE_FILE
MP_POOL = None

class Worker(multiprocessing.Process):
    def __init__(self, group: None = ..., target: Callable[..., Any] | None = ..., name: str | None = ..., args: Iterable[Any] = ..., kwargs: Mapping[str, Any] = ..., *, daemon: bool | None = ...) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)


class TaskManager(object):
    def __init__(self) -> None:
        if MP_POOL is None and __file__ == "__main__":
            MP_POOL = [Worker() for _ in range(multiprocessing.cpu_count())]

    def __enter__(self):
        pass
    def __exit__(self):
        pass
    def __del__(self):
        pass  # stop workers.


class Column(object):
    def __init__(self, data, dtype, *arg,**kwargs):
        if isinstance(data, np.ndarray):
            pass
        elif isinstance(data, Column):
            pass
        else:
            pass
        self._data = h5py.VirtualLayout
        
    def __iter__(self) -> Iterator[_T]:
        if self._index:
            return (self[i] for i in self._index)
        else:
            return (i for i in self)

    def __next__(self):
        for i in self:
            yield i

    def sort(*kwargs):
        self._index = sort_index(self, *kwargs)

    def filter(self, *kwargs):  # used by
        self._index = filter(self, *kwargs)

    def set(self):
        return set(self)

    def index(self):  # used by groupby, join, lookup, filter & sort
        d = defaultdict(list)
        for ix,v in enumerate(self):
            d[v].append(ix)
        return d  # calculated once. Stored in hdf5.

    def __getitem__(self, slice):
        pass

    def __setitem__(self, keys, values):
        pass

class Table(object):
    def __init__(self) -> None:
        self.columns = {}
    
if __name__ == "__main__":
    tbl1 = Table.import_file('d:\remove_duplicates.csv', table_key='2345eafd2faf')  # table_key is from datamap or just a counter in the script.
    tbl2 = Table.load_file('d:\remove_duplicates.csv.hdf5')  # no additional storage is needed because the key is the same.
    tbl3 = Table.from_json('d:\some.json')
    tbl3 = tbl1 + tbl2   # append
    tb3_copy = tbl3.copy()  # virtual dataset.
    tbl4 = tbl3.sort(date=True, order=False)  # date ascending, order descending.
    tbl5 = tbl1.join(tbl2, left_keys=['date'], right_key=['date'], left_columns=['date', 'order'], right_columns=['quantity'], type='left')
    tbl6 = tbl1.lookup(tbl2, left_keys=['date'], right_key=['date'], left_columns=['date', 'order'], right_columns=['quantity'], expr='==')
    dates,quantities = tbl6['date'], tbl6['quantity']
    
    def pct(A,B):  # custom ops.
        d = defaultdict(int)
        for a,b in zip(A,B):
            d[a]+=b        
        return [b/d[a] for a,b in zip(A,B)]

    tbl6['pct'] = Column(dtype=np.float, data=pct(dates,quantities))  # adding column to tbl6.

    tbl7 = tbl2.filter('date')



