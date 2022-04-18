


class Page(object):  # HDF5 dataset.
    def __init__(self, path,route) -> None:
        self.source_path = ""
        self.source_route = "" 

class Column(object):   # HDF5 virtual dataset.
    def __init__(self, data) -> None:
        self.source_path = ""
        self.source_route = ""
        dd = np.array(data)
        self.pytype = dd.dtype
        self._len = None
    @property
    def address(self):
        return (self.source_path,self.source_route)
    def __len__(self):
        pass
    def __setitem__(self, key,value):
        """
        col[ix] == value
        col[slice] == ndarray
        """
        pass
    def __getitem__(self, key):
        """
        col[int] --> value
        col[slice] --> ndarray
        """
        pass
    def append(self, value):
        raise AttributeError("no point. Use extend instead.")
    def extend(self, values):
        pass
    def index(self):
        pass
    def unique(self):
        pass
    def __iter__(self):
        pass  
    def __next__(self):
        pass
    
