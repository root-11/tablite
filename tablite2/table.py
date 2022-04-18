from itertools import count

class Table(object):
    gid = count(1)
    def __init__(self,key=None) -> None:
        if key is None: key = next(Table.gid)
        self.key = key
        self._columns = {}
    @property
    def columns(self):
        pass
    @property
    def rows(self):
        pass
    def __getitem__(self, *keys):
        """
        tbl6[int]  --> row
        tbl6[slice] --> n rows
        tbl6[text]  --> column
        tbl6[args]  --> columns if text, rows if slice or int
        """
        pass
    def __setitem__(self, keys,values):
        """
        tbl6[int]  = row
        tbl6[slice] = rows
        tbl6[text] = Column(...)
        tbl6[args] ... depends: columns if text, rows if slice or int
        """
        pass
    def __iadd__(self,other):
        pass
    def __add__(self,other):
        pass
    def sort(self, **kwargs):
        pass
    def join(self, other, left_keys, right_keys, left_columns, right_columns, join_type):
        pass
    def lookup(self, other, key, lookup_type):
        pass
    def groupby(self, keys, aggregates):
        pass
    def pivot(self, keys, aggregates, **kwargs):
        pass
    def show(self, *args,**kwargs):
        """
        sort AZ, ZA  <--- show only! the source doesn't change.
        unique values <--- applies on the column only.
        filter by condition [
            is empty, is not empty, 
            text {contains, does not contain, starts with, ends with, is exactly},
            date {is, is before, is after}
            value is {> >= < <= == != between, not between}
            formula (uses eval)
        ]
        filter by values [ unique values ]
        """  
        pass  # tbl.show( ('A','>',3), ('B','!=',None), ('A'*'B',">=",6), limit=50, sort_asc={'A':True, 'B':False})  
        # can be programmed as chained generators:
        # arg1 = (ix for ix,v in enumerate(self['A']) if v>3)
        # arg2 = (ix for ix in arg1 if self['B'][ix] != None)
        # arg3 = (ix for ix in arg2 if self['A'][ix] * self['B'] >= 6)
        # unsorted = [self[ix] for ix, _ in zip(arg3, range(limit))]  # self[ix] get's the row.
        # for key,asc in reversed(sort_asc.items():
        #   unsorted.sort(key,reversed=not asc)
        # t = self.copy(no_rows=True).extend(unsorted)
        # return t
    @classmethod
    def import_file(path):
        pass
    @classmethod
    def load_file(path):
        pass
    @classmethod
    def from_json(path):
        pass


