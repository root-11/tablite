from collections import defaultdict

import h5py  #https://stackoverflow.com/questions/27710245/is-there-an-analysis-speed-or-memory-usage-advantage-to-using-hdf5-for-large-arr?rq=1  
import numpy as np
import json


from tablite.config import HDF5_Config
from tablite.utils import intercept
from tablite.datatypes import DataTypes

READONLY = 'r'   # r  Readonly, file must exist (default)
READWRITE = 'r+' # r+ Read/write, file must exist
TRUNCATE = 'w'   # w  Create file, truncate if exists
#                x    Create file, fail if exists
#                a    Read/write if exists, create otherwise


class MemoryManager(object):
    def __init__(self) -> None:
        self.ref_counts = defaultdict(int)
        
        self.path = HDF5_Config.H5_STORAGE
        if not self.path.exists():
            self.path.touch()  

    def create_table(self,key, save):  # /table/{key}
        with h5py.File(self.path, READWRITE) as h5:
            dset = h5.create_dataset(name=key, dtype=h5py.Empty('f'))
            dset.attrs['columns'] = json.dumps({})
            assert isinstance(save, bool)
            dset.attrs['saved'] = save

    def set_saved_flag(self, group, value):
        with h5py.File(self.path, READWRITE) as h5:
            dset = h5[group]
            dset.attrs['saved'] = value

    def create_column_reference(self, table_key, column_name, column_key):  # /column/{key}
        with h5py.File(self.path, READWRITE) as h5:
            table_group = f"/table/{table_key}"
            dset = h5[table_group]  
            columns = json.loads(dset.attrs['columns'])
            columns[column_name] = column_key
            dset.attrs['columns'] = json.dumps(columns) 
            
    def delete_table(self, table_key):        
        with h5py.File(self.path, READWRITE) as h5:
            dset = h5[table_key]
            saved_flag = dset.attrs['saved']
            if saved_flag:
                return

            columns = json.loads(dset.attrs['columns'])
            for name, column_key in columns.items():
                self.delete_column_reference(table_key=table_key, column_name=name, column_key=column_key)
            del h5[table_key]

    def delete_column_reference(self, table_key, column_name, column_key):
        with h5py.File(self.path, READWRITE) as h5:
            dset = h5[table_key]
            saved_flag = dset.attrs['saved']
            if saved_flag:
                return
            
            columns = json.loads(dset.attrs['columns'])
            del columns[column_name] 
            dset.attrs['columns'] = json.dumps(columns)       

            pages = self.get_pages(f"/column/{column_key}")
            del h5[f"/column/{column_key}"]  # deletes the virtual dataset.
            
            for page in pages:
                self.ref_counts[page.group] -= 1

            self.del_pages_if_required(pages)

    def del_pages_if_required(self, pages):
        if not pages:
            return
        with h5py.File(self.path, READWRITE) as h5:
            for page in set(pages):
                if self.ref_counts[page.group]==0:
                    dset = h5[page.group]
                    if dset.attrs['datatype'] == 'typearray':
                        del h5[f"{page.group}_types"]
                    del h5[page.group]        
                    del self.ref_counts[page.group]

    def create_virtual_dataset(self, group, pages_before, pages_after):
        """ The consumer API for Columns to create, update and delete datasets."""
        if not isinstance(pages_before,list):
            raise TypeError("expected at least an empty list.")
        if not isinstance(pages_after, list):
            raise TypeError("expected at least an empty list.")
                
        with h5py.File(self.path, READWRITE) as h5:
            # 1. adjust ref count by adding first, then remove, as this prevents ref count < 1.
            all_pages = pages_before + pages_after
            assert all(isinstance(i, Page) for i in all_pages)

            for page in pages_after:
                if len(page)==0:
                    raise ValueError("page length == 0")
                self.ref_counts[page.group] += 1
            for page in pages_before:
                self.ref_counts[page.group] -= 1
                
            self.del_pages_if_required(pages_before)
            
            # 2. determine new layout.
            dtype, shape = Page.layout(pages_after)
            # 3. create the layout.
            layout = h5py.VirtualLayout(shape=(shape,), dtype=dtype, maxshape=(None,), filename=self.path)
            a, b = 0, 0
            for page in pages_after:
                dset = h5[page.group]
                b += dset.len()
                vsource = h5py.VirtualSource(dset)
                layout[a:b] = vsource
                a = b

            # 4. final write to disk.
            if group in h5:
                del h5[group]
            h5.create_virtual_dataset(group, layout=layout)
            
            return shape

    def get_pages(self, group):
        if not group.startswith('/column'):
            raise ValueError

        with h5py.File(self.path, READONLY) as h5:
            if group not in h5:
                return Pages()
            else:
                dset = h5[group]
                return Pages([ Page.load(pg_grp) for _,_,pg_grp,_ in dset.virtual_sources() ])  # https://docs.h5py.org/en/stable/high/dataset.html#h5py.Dataset.virtual_sources

    def get_ref_count(self, page):
        assert isinstance(page, Page)
        return self.ref_counts[page.group]

    def reset_storage(self):
        with h5py.File(self.path, TRUNCATE) as h5:
            assert list(h5.keys()) == []
            
    def get_data(self, group, item):
        if not group.startswith('/column'):
            raise ValueError("get data should be called by columns only.")
        if not isinstance(item, (int,slice)):
            raise TypeError(f'{type(item)} is not slice')
        
        with h5py.File(self.path, READONLY) as h5:
            if group not in h5:
                return np.array([])
            dset = h5[group]
            if dset.dtype.char != 'O':  # it's a single type dataset.
                return dset[item]
            
            # else: it's a multi-type dataset
            arrays = []
            assert isinstance(item, slice)
            item_range = range(*item.indices(dset.len()))
            
            start,end = 0,0
            pages = self.get_pages(group)
            for page in pages:  # loaded Pages
                start = end
                end += len(page)
                ro = intercept(range(start,end,1), item_range)  # check if the slice is worth converting.
                if len(ro)!=0:  # fetch the slice and filter it.
                    search_slice = slice(ro.start - start, ro.stop - start, ro.step)
                    match = page[search_slice]  # page.__getitem__ handles type conversion for Mixed and Str types.
                    arrays.append(match)
            
            dtype, _ = Page.layout(pages)
            return np.concatenate(arrays, dtype=dtype)  


class Pages(list):
    """
    behaves like a list.
    """
    def __init__(self,values=None):
        if values:
            super().__init__(values)
        else:
            super().__init__()
        
    def get_page_by_index(self, index):
        a,b = 0,0
        for ix, page in enumerate(self):
            a = b
            b += len(page)
            if a <= index < b:
                return ix, a, b, page
        raise IndexError(f"index {index} not found (lenght {b})")

    def length(self):
        """ returns sum(len(p) for p in pages) """
        return sum(len(p) for p in self)

    def getslice(self, start, stop):
        """ returns a slice as pages 
        
        logic:
         A:B page start:stop
         a:b slice start:stop
    
              A-------B                              case
        a---b |       |        stop search            (1)
          a---b       |        stop search            (2)
            a-+-b     |        create new page        (3)
              a---b   |        create new page        (4)
              | a---b |        create new page        (5)
              |   a---b        create new page        (6)
              |     a-+-b      create new page        (7)
              |       a---b    continue to next page  (8)
              |       | a---b  continue to next page  (9)
              a-------b        include page          (10)
            a-+-------b        include page          (11)
              a-------+-b      include page          (12)
            a-+-------+-b      include page          (13)
        
        """
        if start==stop:
            return []
        L = []
        a, b = 0, 0
        for page in self:
            a = b
            b += len(page)
            if stop < a:  # cases (1,2)
                break
            elif b < start:  # cases (8,9)
                continue
            elif start <= a and b <= stop:  # cases (10,11,12,13)
                L.append(page)
            else:  # cases (3,4,5,6,7)
                p_start = a if start < a else start
                p_stop = b if stop > b else stop                   
                data = page[p_start-a:p_stop-a]
                if len(data):
                    new = Page(data)
                    L.append(new)
        return L




class GenericPage(object):
    _page_ids = 0  # when loading from disk increment this to max id.
    _type_array = 'typearray'
    _type_array_postfix = "_types"
    _str = str.__name__
    _datatype = 'datatype'
    _encoding = 'encoding'
    _MixedTypes = "O" # + _type_array
    _StringTypes = "U" # + _str
    _SimpleTypes = "ld?"

    @classmethod
    def page_class_type_from_np(cls, np_arr):
        """ returns Page class from np.ndarray """
        if not isinstance(np_arr, np.ndarray):
            raise TypeError
        
        if np_arr.dtype.char in cls._MixedTypes:
            clss = MixedType
        elif np_arr.dtype.char in cls._StringTypes:
            clss = StringType
        elif np_arr.dtype.char in cls._SimpleTypes:  # check if a new int8 is not included in an int32.
            clss = SimpleType
        else:
            raise NotImplementedError(f"method missing for {np_arr.dtype.char}")
        return clss

    @classmethod
    def page_class_type(cls, h5attr_dtype):
        if not isinstance(h5attr_dtype,str):
            raise TypeError
        
        class_types = {
            cls._type_array: MixedType,
            cls._str: StringType
        }
        class_type = class_types.get(h5attr_dtype, SimpleType)
        return class_type
   
    @classmethod
    def new_id(cls):
        cls._page_ids += 1
        return cls._page_ids

    def __init__(self, group):        
        if not group.startswith('/page'):
            raise ValueError
        
        self.encoding = HDF5_Config.H5_ENCODING
        self.path = HDF5_Config.H5_STORAGE
        self.group = group

        self.stored_datatype = None  # stored type
        self.original_datatype = None  # original type
        self._len = 0
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__} | {self.group} | {self.original_datatype} | {self.stored_datatype} | {self._len}"

    def __hash__(self) -> int:
        return hash(self.group)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __o: object) -> bool:
        return self.group == __o.group

    @classmethod
    def layout(cls, pages):
        """ 
        finds the common datatype 

        int + int --> higher order int wins
        float + float --> higher order float wins
        str + str --> longest str wins.
        str + float --> Mixed
        int + float --> Mixed
        str + int --> Mixed
        Python object + {str,int,float} --> Mixed
        Python object + python object --> Mixed

        return dtype, shape
        """
        assert all(isinstance(p, Page) for p in pages)
        dtypes = {page.original_datatype for page in pages}
        shape = sum(len(page) for page in pages)

        if len(dtypes) == 1:
            dtype = dtypes.pop()
            if dtype in {cls._type_array, cls._str}:
                dtype = h5py.string_dtype(encoding=HDF5_Config.H5_ENCODING)
            else:
                dtype = np.dtype(dtype)
        else:  # there are multiple datatypes.
            dtype = h5py.string_dtype(encoding=HDF5_Config.H5_ENCODING)
        
        return dtype, shape

    @classmethod
    def load(cls, group):
        """
        loads an existing group.
        """
        with h5py.File(HDF5_Config.H5_STORAGE, READONLY) as h5:
            dset = h5[group]
            datatype = dset.attrs[cls._datatype]
            
            pg_class = cls.page_class_type(h5attr_dtype=datatype)
            page = pg_class(group)
            page.stored_datatype = dset.dtype
            page.original_datatype = datatype
            page._len = dset.len()

            return page

    @classmethod
    def create(cls, data):
        if not isinstance(data, np.ndarray):
            types = {type(v) for v in data}
            if len(types)>1:
                data = np.array(data, dtype='O')
            else:
                data = np.array(data)  # str, int, float

        pg_cls = cls.page_class_type_from_np(data)
        group = f"/page/{cls.new_id()}"
        pg = pg_cls(group,data)
        return pg

    def write(self, data, dtype):
        with h5py.File(READWRITE) as h5:
            if self.group in h5:
                raise ValueError("page already exists")
            dset = h5.create_dataset(name=self.group, 
                                     data=data,  # data is now HDF5 compatible.
                                     dtype=dtype,  # the HDF5 stored dtype may require unpacking using dtypes if they are different.
                                     maxshape=(None,),  # the stored data is now extendible / resizeable.
                                     chunks=HDF5_Config.H5_PAGE_SIZE)  # pages are chunked, so that IO block will be limited.
            dset.attrs['datatype'] = self.original_datatype

    def __len__(self):
        return self._len

    def read(self):
        with h5py.File(self.path, READONLY) as h5:
            dset = h5[self.group]
            self.stored_datatype = dset.dtype
            self.original_datatype = dset.attrs['datatype']
            self._len = dset.len()
    
    def __getitem__(self, item):
        raise NotImplementedError("subclasses must implement this method.")

    def __setitem__(self, index, value):
        raise NotImplementedError("subclasses must implement this method.")

    def append(self, value):
        raise NotImplementedError("subclasses must implement this method.")

    def insert(self, index, value):
        raise NotImplementedError("subclasses must implement this method.")

    def extend(self, values):
        raise NotImplementedError("subclasses must implement this method.")


class SimpleType(GenericPage):
    def __init__(self, group, data=None):
        super().__init__(group)
        if data is not None:
            self.create(data)
    
    def create(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError
        assert data.dtype.char not in 'OU'
     
        self._len = len(data)
        self.stored_datatype = data.dtype
        self.original_datatype = data.dtype.name
             
        with h5py.File(self.path, READWRITE) as h5:
            if self.group in h5:
                raise ValueError("page already exists")
            dset = h5.create_dataset(name=self.group, 
                                     data=data,  
                                     dtype=data.dtype,  
                                     maxshape=(None,),  # the stored data is now extendible / resizeable.
                                     chunks=HDF5_Config.H5_PAGE_SIZE)  # pages are chunked, so that IO block will be limited.
            dset.attrs[self._datatype] = self.original_datatype
            dset.attrs[self._encoding] = self.encoding

    def __getitem__(self, item):
        if item is None:
            item = slice(0,None,1)
        if not isinstance(item, slice):
            raise TypeError

        with h5py.File(self.path, READONLY) as h5:
            dset = h5[self.group]
            return dset[item]

    def __setitem__(self, keys, values):
        with h5py.File(self.path, READWRITE) as h5:
            if isinstance(keys, int):
                dset = h5[self.group]
                if isinstance(values, (list,tuple,np.ndarray)):
                    dset[keys] = values[0]
                else:    # it's a boolean, int or float or similar. just try and let the exception handler deal with it...
                    dset[keys] = values  

            elif isinstance(keys, slice):
                if len(values) != self._len:
                    self._len = len(values)
                    dset = h5[self.group]
                    dset.resize((len(values)), axis=0)
                dset[slice] = values
            else:
                raise TypeError()

    def append(self, value):
        # value must be np.ndarray and of same type as self.
        # this has been checked before this point, so I let h5py complain if there is something wrong.
        with h5py.File(self.path, READWRITE) as h5:
            dset = h5[self.group]
            dset.resize(dset.len() + len(value),axis=0)
            dset[-len(value):] = value
            self._len += len(value)

    def insert(self, index, value):
        with h5py.File(self.path, READWRITE) as h5:
            dset = h5[self.group]
            value = np.array([value], dtype=dset.dtype)
            data = dset[:]
            dset.resize(dset.len() + len(value), axis=0)

            a, b = index, index + len(value)
            dset[:a] = data[:a]
            dset[a:b] = value
            dset[b:] = data[a:]
            self._len += len(value)

    def extend(self, values):
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError(f".extend requires an iterable")
        with h5py.File(self.path, READWRITE) as h5:
            dset = h5[self.group]

            if not isinstance(values, np.ndarray):
                values = np.array(values, dtype=dset.dtype)
            dset.resize(dset.len() + len(values), axis=0)
            dset[-len(values):] = values
            self._len += len(values)

    def remove(self, value):
        raise NotImplementedError("subclasses must implement this method.")
    
    def remove_all(self, value):
        raise NotImplementedError("subclasses must implement this method.")

    def pop(self, index):
        raise NotImplementedError("subclasses must implement this method.")


class StringType(GenericPage):
    def __init__(self, group, data=None):
        super().__init__(group)
        if data is not None:
            self.create(data)

    def create(self, data):
        self.original_datatype = self._str
        self.stored_datatype = h5py.string_dtype(encoding=HDF5_Config.H5_ENCODING)  # 'U'
        data = data.astype(bytes)
        self._len = len(data)

        with h5py.File(self.path, READWRITE) as h5:
            if self.group in h5:
                raise ValueError("page already exists")
                
            dset = h5.create_dataset(name=self.group, 
                                     data=data,  # data is now HDF5 compatible.
                                     dtype=self.stored_datatype,  # the HDF5 stored dtype may require unpacking using dtypes if they are different.
                                     maxshape=(None,),  # the stored data is now extendible / resizeable.
                                     chunks=HDF5_Config.H5_PAGE_SIZE)  # pages are chunked, so that IO block will be limited.
            dset.attrs[self._datatype] = self.original_datatype
            dset.attrs[self._encoding] = self.encoding

    def __getitem__(self, item):
        if item is None:
            item = slice(0,None,1)
        if not isinstance(item, slice):
            raise TypeError
        
        with h5py.File(self.path, READONLY) as h5:
            dset = h5[self.group]
            match = dset[item]
            encoding = dset.attrs['encoding']
            match = np.array( [v.decode(encoding) for v in match] )
            return match            

    def __setitem__(self, index, value):
        raise NotImplementedError("subclasses must implement this method.")

    def append(self, value):
        # value must be np.ndarray and of same type as self.
        # this has been checked before this point, so I let h5py complain if there is something wrong.
        with h5py.File(self.path, READWRITE) as h5:
            dset = h5[self.group]
            dset.resize(dset.len() + len(value),axis=0)
            dset[-len(value):] = value.astype(bytes) # encoding to bytes is required.
            self._len += len(value)
    
    def insert(self, index, value):
        raise NotImplementedError("subclasses must implement this method.")

    def extend(self, values):
        with h5py.File(self.path, READWRITE) as h5:
            dset = h5[self.group]
            dset.resize(dset.len() + len(values),axis=0)
            dset[-len(values):] = np.array(values, dtype=str).astype(bytes) # encoding to bytes is required.
            self._len += len(values)
    
    def remove(self, value):
        raise NotImplementedError("subclasses must implement this method.")
    
    def remove_all(self, value):
        raise NotImplementedError("subclasses must implement this method.")

    def pop(self, index):
        raise NotImplementedError("subclasses must implement this method.")


class MixedType(GenericPage):
    def __init__(self, group, data=None):
        super().__init__(group)
        self.type_group = None
        self.type_array = None
        if data is not None:
            self.create(data)
    
    def create(self, data):
        self._len = len(data)

        with h5py.File(self.path, READWRITE) as h5:
            if self.group in h5:
                raise ValueError("page already exists")
            
            # get typecode for encoding.
            type_code = DataTypes.type_code
            self.type_group = f"{self.group}{self._type_array_postfix}"
            self.type_array = np.array( [ type_code(v) for v in data.tolist() ] )
            self.original_datatype = self._type_array
            
            byte_function = DataTypes.to_bytes
            data = np.array( [byte_function(v) for v in data.tolist()] )
            
            self.stored_datatype = h5py.string_dtype(encoding=HDF5_Config.H5_ENCODING)  # type 'O'

            # dataset
            dset = h5.create_dataset(name=self.group, 
                                     data=data,  # data is now HDF5 compatible.
                                     dtype=self.stored_datatype,  # the HDF5 stored dtype may require unpacking using dtypes if they are different.
                                     maxshape=(None,),  # the stored data is now extendible / resizeable.
                                     chunks=HDF5_Config.H5_PAGE_SIZE)  # pages are chunked, so that IO block will be limited.
            dset.attrs[self._datatype] = self.original_datatype
            dset.attrs[self._encoding] = self.encoding
            
            # typearray 
            h5.create_dataset(name=self.type_group, 
                              data=self.type_array, 
                              dtype=self.type_array.dtype, 
                              maxshape=(None,), 
                              chunks=HDF5_Config.H5_PAGE_SIZE)

    def __getitem__(self, item):
        if item is None:
            item = slice(0,None,1)
        if not isinstance(item, slice):
            raise TypeError

        with h5py.File(self.path, READONLY) as h5:
            # check if the slice is worth converting.
            dset = h5[self.group]
            match = dset[item]            
            
            type_group = f"{self.group}{self._type_array_postfix}"
            type_array = h5[type_group][item]  # includes the page id
            type_functions = DataTypes.from_type_code

            match = np.array( [type_functions(v,type_code) for v,type_code in zip(match, type_array)], dtype='O' )
            return match
            
    def __setitem__(self, keys, values):
        dtypes = DataTypes
        with h5py.File(self.path, READWRITE) as h5:
            if isinstance(keys, int):
                dset = h5[self.group]
                dset[keys] = dtypes.to_bytes(values)

                type_group = f"{self.group}{self._type_array_postfix}"
                dset = h5[type_group]
                dset[keys] = dtypes.type_code(values)

            elif isinstance(keys, slice):
                raise NotImplementedError("subclasses must implement this method.")
            else:
                raise TypeError(f"bad key: {type(keys)}")

    def append(self, value):
        # value must be np.ndarray and of same type as self.
        # this has been checked before this point, so I let h5py complain if there is something wrong.
        type_code = DataTypes.type_code
        type_array = np.array( [ type_code(v) for v in value.tolist() ] )

        byte_function = DataTypes.to_bytes
        data = np.array( [byte_function(v) for v in data.tolist()] )
        
        with h5py.File(self.path, READWRITE) as h5:
            # update dataset
            dset = h5[self.group]
            dset.resize(dset.len() + len(value),axis=0)
            dset[-len(value):] = data
            
            # update type array
            dset = h5[self.type_group]
            dset.resize(dset.len() + len(value),axis=0)
            dset[-len(value):] = type_array

            # update self
            self._len += len(value)

    def insert(self, index, value):
        raise NotImplementedError("subclasses must implement this method.")

    def extend(self, values):
        raise NotImplementedError("subclasses must implement this method.")

    def remove(self, value):
        raise NotImplementedError("subclasses must implement this method.")
    
    def remove_all(self, value):
        raise NotImplementedError("subclasses must implement this method.")

    def pop(self, index):
        raise NotImplementedError("subclasses must implement this method.")
   

class Page(object):
    """
    The Page class is the consumer API for the Column. It hides the underlying type
    handling of numpy and hdf5 so that the user can use the Column as a paginated
    list.

    The class inheritance uses a map-reduce type layout:

                    GenericPage  -- common skeleton and helper methods for all 3 page types.
                        |
                        V
        +---------------+-----------+
        |               |           |
    SimpleType     StringType    MixedType  -- type specific implementation.
        |               |           |
        +---------------+-----------+
                        |
                        V
                       Page   -- consumer api.
    """

    def __init__(self, data=None):
        if isinstance(data, GenericPage):  # used only during cls.load
            self._page = data
        else:
            self._page = GenericPage.create(data) 
    
    
    def __str__(self) -> str:
        return f"Page({self._page.__str__()})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Page):
            raise TypeError
        return self._page == __o._page

    def __hash__(self):
        return self._page.__hash__()

    @property
    def group(self):
        return self._page.group

    @property
    def original_datatype(self):
        return self._page.original_datatype

    @classmethod
    def load(cls, group):
        return Page(GenericPage.load(group))

    @classmethod
    def layout(cls, pages):
        return GenericPage.layout(pages)

    def __len__(self):
        return len(self._page)

    def extend(self, values):
        try:
            self._page.extend(values)
        except (TypeError, ValueError):
            data = self._page[:].tolist()
            data.extend(values)
            self._page = GenericPage.create(self._page.group, data)

    def append(self, value):
        try:
            self._page.append(value)
        except (TypeError, ValueError):
            data = self._page[:].tolist()
            data.append(value)
            self._page = GenericPage.create(self._page.group, data)

    def insert(self, index, values):
        try:
            self._page.insert(index, values)
        except (TypeError, ValueError):
            data = self._page[:].tolist()
            data.insert(index, values)
            self._page = GenericPage.create(self._page.group, data)

    def remove(self, value):
        self._page.remove(value)
    
    def remove_all(self, value):
        self._page.remove_all(value)

    def pop(self, index):
        self._page.pop(index)

    def __getitem__(self, key):
        return self._page[key]

    def __setitem__(self, key, value):
        try:
            self._page[key] = value
        except (TypeError, ValueError):
            data = self._page[:]
            data[key]=value
            self._page = GenericPage.create(self._page.group, data)
        
    def __delitem__(self, key):
        del self._page[key]

    