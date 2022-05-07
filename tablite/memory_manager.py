from collections import defaultdict

import h5py
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
                        del h5[f"{page}_types"]
                    del h5[page.group]        
                    del self.ref_counts[page.group]

    # def create_column(self, key):
    #     pass  # nothing to do.
    #     raise AttributeError("create column does nothing. Column creation is handled by the table. See create_column_reference")
        
    # def delete_column(self, key):
    #     pass  # nothing to do.
    #     raise AttributeError("delete column does nothing. Column delete is handled by the table. See delete_column_reference")   

    # def create_virtual_dataset_from_groups(self, *groups):
    #     new_pages = []
    #     for group in groups:
    #         new_pages.extend(self.get_pages(group))

    def create_virtual_dataset(self, group, old_pages, new_pages):
        """ The consumer API for Columns to create, update and delete datasets."""
        if not isinstance(old_pages,list):
            raise TypeError
        if not isinstance(new_pages, list):
            raise TypeError
        
        with h5py.File(self.path, READWRITE) as h5:
            # 2. adjust ref count by adding first, then remove, as this prevents ref count < 1.
            all_pages = old_pages + new_pages
            assert all(isinstance(i, Page) for i in all_pages)
           
            if new_pages:
                for page in all_pages:  # add ref count for new connection.
                    self.ref_counts[page.group] += 1
                for page in old_pages:  # remove duplicate ref count.
                    self.ref_counts[page.group] -= 1
                
                self.del_pages_if_required(old_pages)
            
            # 3. determine new layout.
            dtype, shape = Page.layout(all_pages)
            # 4. create the layout.
            layout = h5py.VirtualLayout(shape=(shape,), dtype=dtype, maxshape=(None,), filename=self.path)
            a, b = 0, 0
            for page in all_pages:
                dset = h5[page.group]
                b += dset.len()
                vsource = h5py.VirtualSource(dset)
                layout[a:b] = vsource
                a = b

            # 5. final write to disk.
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
                return Pages([ Page.load(pg_grp) for _,_,pg_grp,_ in dset.virtual_sources() ])

    # def get_page_by_index(self, group, index):
    #     """ consumer API for column index """
    #     if not group.startswith('/column'):
    #         raise ValueError

    #     with h5py.File(self.path, READONLY) as h5:
    #         if group not in h5:
    #             raise KeyError(group)
    #         a,b = 0,0
    #         for page in self.get_pages(group):
    #             a = b
    #             b += len(page)
    #             if a <= index < b:
    #                 return page
    #         raise IndexError(f"index {index} not found (lenght {b})")

    def get_ref_count(self, page):
        assert isinstance(page, Page)
        return self.ref_counts[page.group]

    # def setitem(self, group, key, value):
    #     if not group.startswith('/column'):
    #         raise ValueError

    #     if isinstance(key, int):
    #         key = slice(key,key+1)
        
        
    #     with h5py.File(self.path, READONLY) as h5:
    #         if group not in h5:
    #             raise KeyError(group)

    #         dset = h5[group]
    #         pages = [ Page.load(pg_grp) for _,_,pg_grp,_ in dset.virtual_sources() ]
            
    #         ros, last_page = [], None
    #         a,b = 0,0
    #         for ix, page in enumerate(pages):
    #             a, b = b, len(page)
    #             ro = intercept( range(a,b), range(*key.indices(b)) )
    #             if len(ro)!=0:
    #                 ros.append((ix, a, b, page, ro))
    #                 last_page = page
                
    #         # check for special cases:
    #         if len(range(*key.indices(b))) == 0:
    #             if key.start == b:  # it's an append/extend: L[10:] = [values]
    #                 pass
    #             elif key.stop == 0:  # it's an upfront insert  L[:0] = [values]
    #                 pass
    #             else:  # it's a regular insert L[3:3] = [values]
    #                 pass
    #         for ix,a,b,page,ro in ros:
    #             if page == last_page:
    #                 pass
    #             if self.ref_counts[page.group] == 1:
    #                 new_key = slice(ro.start-a, ro.stop-a, ro.step)
    #                 page[new_key] = value[:len(ro)]
    #             else:
    #                 data = page[:]
    #                 new_page = pass

    def reset_storage(self):
        with h5py.File(self.path, TRUNCATE) as h5:
            assert list(h5.keys()) == []

    # def append_to_virtual_dataset(self, group, new_data):  
    #     if not group.startswith("/column"):
    #         raise ValueError("only columns have virtual datasets.")
    #     if not isinstance(new_data, np.ndarray):
    #         raise TypeError
        
    #     pages = self.get_pages(group)
    #     last_page = pages[-1]
    #     if self.ref_counts[last_page.group] == 1:  # resize page
    #         target_cls = last_page.page_class_type_from_np(new_data)
    #         if isinstance(last_page, target_cls):
    #             last_page.append(new_data)
    #             shape = self.create_virtual_dataset(group, new_pages=None)
    #             # Note above: check if it's better to resize the virtual layout using
    #             #             info from: https://github.com/h5py/h5py/issues/1202
    #             return shape  # EXIT 1 ---
    #         else:  # make new page.
    #             pass 
    #     else:  # make new page.
    #         pass
    #     # make new page:
    #     page = Page.create(new_data)
    #     shape = self.create_virtual_dataset(group, [page])
    #     return shape  # EXIT 2 ---
            
    def get_data(self, group, item):
        if not group.startswith('/column'):
            raise ValueError("get data should be called by columns only.")
        
        with h5py.File(self.path, READONLY) as h5:
            dset = h5[group]
            if dset.dtype.char != 'O':  # it's a single type dataset.
                return dset[item]
            
            arrays = []
            assert isinstance(item, slice)
            item_range = range(*item.indices(dset.len()))
            
            start,end = 0,0
            pages = self.get_pages(group)
            for page in pages:  # loaded Pages
                end = len(page)
                ro = intercept(range(start,end,1), item_range)  # check if the slice is worth converting.
                if len(ro)!=0:  # fetch the slice and filter it.
                    search_slice = slice(ro.start - start, ro.stop - start, ro.step)
                    match = page[search_slice]  # page.__getitem__ handles type conversion for Mixed and Str types.
                    arrays.append(match)
                start = end
            
            dtype, _ = Page.layout(pages)
            return np.concatenate(arrays, dtype=dtype)

    # def update_data(self, group, keys, other_group):
    #     if not group.startswith('/column'):
    #         raise ValueError("get data should be called by columns only.")
    #     if not other_group.startswith('/column'):
    #         raise ValueError("get data should be called by columns only.")
        
        
    #     if keys == slice(0,None,None):  # then it's simple extension
    #         new_pages = self.get_pages(other_group)
    #         shape = self.create_virtual_dataset(group, new_pages=new_pages)
    #         return shape
        
    #     if keys == slice(None,0,None):  # then it's an up front insert
    #         new_pages = self.get_pages(other_group)
    #         shape = 

    #     old_pages = self.get_pages(group)
    #     old_length = sum(len(p) for p in old_pages)

    #     ri = intercept(range(0,old_length), range(*keys.indices()))
    #     if len(ri)==0:
    #         pass

    #     if 0 <= keys.start <= keys.stop <= old_length:  # it's probably an insert      

    def update_data(self, group, keys, values):
        """ consumer API for core.column - mainly delegates to Pages. """
        if not group.startswith('/column'):
            raise ValueError("get data should be called by columns only.")

        if not isinstance(values, np.ndarray):
            if isinstance(values, (list,tuple)):
                values = np.array(values)
            else:
                values = np.array([values])
        
        pages = self.get_pages(group)

        if not pages:
            new_page = Page.create(values)
            new_pages = [new_page]
            shape = self.create_virtual_dataset(group, new_pages=new_pages)
            return shape  # EXIT 1

        new_pages = pages[:]  # we keep a copy for the virtual dataset.
        shape = sum(len(p) for p in pages)

        if isinstance(keys, int):
            a,b = 0,0
            for ix, page in enumerate(pages):
                b += len(page)
                if a <= keys < b:
                    vclass = Page.page_class_type_from_np(values) 
                    if self.ref_counts[page.group] == 1 and isinstance(page,vclass):  # update.
                        page[keys-a] = values
                    else:
                        data = page[:]
                        data[keys-a] = values
                        new_page = Page.create(data)
                        new_pages = pages[:]
                        new_pages[ix] = new_page
                    break  #  ...
                a = b

        elif isinstance(keys, slice):
            max_length = shape + len(values)
            start,stop,_ = slice.indices(max_length)  # slice as forward loop.

            keys_range = range(*keys.indices(max_length))
            
            a,b = 0,0
            for ix, page in enumerate(pages):
                b += len(page)
                if start > b:  # we're before the start of the key.
                    a = b
                    continue
                if stop < a:  # we've passed the end of the key
                    break
                             
                ro = intercept(range(a,b,1), keys_range)  # ro = range object sliced according to the keys
                size = len(ro)
                if size != 0:
                    slc = slice(ro.start-a, ro.stop-a, ro.step)
                    data = page[:].tolist()
                    data[slc], values = values[:size], values[size:]  # perform the replacement and reduce the value list
                
                    if (stop < b or b == shape) and values:  # insert as we've reached 
                        # the end of the key slice or the end of the pages.
                        data += values
                        values = []

                    arr = np.array(data)
                    vclass = Page.page_class_type_from_np(arr)
                    if self.ref_counts[page.group] == 1 and isinstance(page, arr):  # update.
                        page[:] = data
                    else:  # create new page
                        new_page = Page.create(data)
                        new_pages[ix] = new_page
                a = b 
                if not values:
                    break
        else:
            raise NotImplementedError()

        if new_pages != pages:
            shape = self.create_virtual_dataset(group, new_pages=new_pages)
        return shape  # EXIT 2


class Pages(list):
    def __init__(self,values=None):
        if values:
            super().__init__(values)
        else:
            super().__init__()

    def get_page_by_index(self, index):
        a,b = 0,0
        for page in self:
            a = b
            b += len(page)
            if a <= index < b:
                return page
        raise IndexError(f"index {index} not found (lenght {b})")

    # def start(self, page):
    #     a,b = 0,0
    #     for p in self:
    #         a = b 
    #         b += len(p)
    #         if p == page:
    #             return a

class Page(object):
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
        elif np_arr.dtype.char in cls._SimpleTypes:
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
        return f"Page({self.group} | {self.original_datatype} | {self.stored_datatype} | {self._len})"

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
        """
        creates a new group.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError
        
        pg_cls = cls.page_class_type_from_np(data)
        group = f"/page/{Page.new_id()}"
        pg = pg_cls(group)
        pg.create(data)
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

    def update(self):
        raise NotImplementedError("subclasses must implement this method.")
    
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


class SimpleType(Page):
    def __init__(self, group, data=None):
        super().__init__(group)
        if data:
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
        elif not isinstance(item, slice):
            raise TypeError

        with h5py.File(self.path, READONLY) as h5:
            dset = h5[self.group]
            return dset[item]

    def __setitem__(self, keys, values):
        if not isinstance(keys, (slice,int)) or not isinstance(values, np.ndarray):
            raise TypeError("pages should only see numpy arrays.")
        if not isinstance(self, self.page_class_type_from_np(values)):
            raise TypeError()

        with h5py.File(self.path, READWRITE) as h5:
            if isinstance(keys, int):
                dset = h5[self.group]
                dset[keys] = values[0]

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
        raise NotImplementedError("subclasses must implement this method.")

    def extend(self, values):
        raise NotImplementedError("subclasses must implement this method.")

    def remove(self, value):
        raise NotImplementedError("subclasses must implement this method.")
    
    def pop(self, index):
        raise NotImplementedError("subclasses must implement this method.")


class StringType(Page):
    def __init__(self, group, data=None):
        super().__init__(group)
        if data:
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
        elif not isinstance(item, slice):
            raise TypeError
        
        with h5py.File(self.path, READONLY) as h5:
            dset = h5[self.group]
            encoding = dset.attrs['encoding']
            match = np.array( [v.decode(encoding) for v in dset] )
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
        raise NotImplementedError("subclasses must implement this method.")

    def remove(self, value):
        raise NotImplementedError("subclasses must implement this method.")
    
    def pop(self, index):
        raise NotImplementedError("subclasses must implement this method.")


class MixedType(Page):
    def __init__(self, group, data=None):
        super().__init__(group)
        self.type_group = None
        self.type_array = None
        if data:
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
        elif not isinstance(item, slice):
            raise TypeError

        with h5py.File(self.path, READONLY) as h5:
            # check if the slice is worth converting.
            dset = h5[self.group]
            match = dset[item]            
            
            type_group = f"{self.group}{self._type_array_postfix}"
            type_array = h5[type_group][item]  # includes the page id
            type_functions = DataTypes.from_type_code

            match = np.array( [type_functions(v,type_code) for v,type_code in zip(match, type_array)] )
            return match
            
    def __setitem__(self, index, value):
        raise NotImplementedError("subclasses must implement this method.")

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
    
    def pop(self, index):
        raise NotImplementedError("subclasses must implement this method.")
   
