from collections import defaultdict
import atexit

import h5py
import numpy as np
import json


from tablite.config import HDF5_Config
from tablite.utils import normalize_slice, intercept
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
                self.ref_counts[page] -= 1

            for page in set(pages):
                if self.ref_counts[page]==0:
                    dset = h5[page]
                    if dset.attrs['datatype'] == 'typearray':
                        del h5[f"{page}_types"]
                    del h5[page]        
                    del self.ref_counts[page]

    def create_column(self, key):
        pass  # nothing to do.
        raise AttributeError("create column does nothing. Column creation is handled by the table. See create_column_reference")
        
    def delete_column(self, key):
        pass  # nothing to do.
        raise AttributeError("delete column does nothing. Column delete is handled by the table. See delete_column_reference")   

    def create_virtual_dataset(self, group, new_pages):
        old_pages = self.get_pages(group)
        
        with h5py.File(self.path, READWRITE) as h5:
            # 2. adjust ref count by adding first, then remove, as this prevents ref count < 1.
            all_pages = old_pages + new_pages
            assert all(isinstance(i, Page) for i in all_pages)
           
            for page in all_pages:  # add ref count for new connection.
                self.ref_counts[page.group] += 1
            for page in old_pages:  # remove duplicate ref count.
                self.ref_counts[page.group] -= 1
            
            # 3. determine new layout.
            shape, dtype = Page.layout(all_pages)
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
            if group in h5:
                dset = h5[group]
                return [ Page.load(pg_grp) for _,_,pg_grp,_ in dset.virtual_sources() ]
        return []

    def reset_storage(self):
        with h5py.File(self.path, TRUNCATE) as h5:
            assert list(h5.keys()) == []

    def append_to_virtual_dataset(self, group, new_data):  # REWORK.
        if not group.startswith("/column"):
            raise ValueError("only columns have virtual datasets.")

        with h5py.File(self.path, READWRITE) as h5:
            pages = self.get_pages(group)
            last_page = pages[-1]
            if self.ref_counts[last_page] > 1:  # make new page.
                page = self.create_page(new_data)
                shape = self.create_virtual_dataset(group, page)
                return shape
            else:  # resize page
                dset = h5[last_page]

    # def append_to_page(self, page, new_data):  # REWORK.
    #     if not page.startswith("/page"):
    #         raise ValueError
    #     if self.ref_counts[page] > 1: 
    #         raise AttributeError("ref count > 1. Bad logic?")

    #     if not isinstance(new_data,np.array):
    #         new_data = np.array(new_data)

    #     with h5py.File(self.path, READWRITE) as h5:
    #         dset = h5[page]
    #         source_type = dset.attrs['datatype']
            
    #         if new_data.dtype.name == source_type:
    #             pass # they're the same. Resize page

    #         elif source_type == 'typearray':
    #             pass  # convert input to match mixed types.

    #         elif source_type == 'str':  # make new data strings.
    #             if new_data.dtype.char == 'U':
    #                 new_data = new_data.astype(bytes)
    #                 data = np.concatenate((dset,new_data))
                    
    #                 pass  # create new page.
    #             else:
    #                 pass  # resize page
    #         # ... else:  # source is a native numpy type and they're different.
    #         elif new_data.dtype.char == 'O':
    #             pass  # make everything 'typearray'
    #         elif new_data.dtype.char == 'U':
    #             pass  # create new page where everything is 'str'
    #         else:  # uniform HDF5 & numpy compatible datatype.
    #             pass # both float? int? You will never know. create new page.
    #             data = np.array( list(dset) + list(new_data) )
                
    #         dset.resize( dset.shape[0] + len(new_data ), axis=0)
    #         dset[-len(new_data):] = new_data  # this will not work for composite data.

    def create_page(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        page = Page.create(data)
        self.ref_counts[page.group] += 1       
        return page.group

    def get_data(self, group, item):
        if not group.startswith('/column'):
            raise ValueError("get data should be called by columns only.")
        with h5py.File(self.path, READONLY) as h5:
            dset = h5[group]
            return dset[item]


class Page(object):
    _page_ids = 0  # when loading from disk increment this to max id.

    @classmethod
    def new_id(cls):
        cls._page_ids += 1
        return cls._page_ids

    def __init__(self, group=None):        
        if not group.startswith('/page'):
            raise ValueError
        
        self.path = HDF5_Config.H5_STORAGE
        self.group = group
        self.dtype = None
        self.source_type = None
        self.encoding = HDF5_Config.H5_ENCODING
            
    @classmethod
    def layout(pages):
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
        dtypes = {page.dtype.char for page in pages}
        shape = sum(len(page) for page in pages)
        if 'O' in dtypes:
            dtype = 'O'
        elif len(dtypes) == 1:
            dtype = dtypes.pop()
        else:  # there are multiple datatypes.
            dtype = 'O'
        
        return dtype, shape
    
    @classmethod
    def load(cls, group):
        """
        loads an existing group.
        """
        with h5py.File(HDF5_Config.H5_STORAGE, READONLY) as h5:
            dset = h5[group]
            
            if dset.dtype.char == 'O':
                page =  MixedType(group)
            elif dset.dtype.char == 'U':
                page = StringType(group)
            else:
                page = SimpleType(group)

            page.dtype = dset.dtype
            page.source_type = dset.attrs['datatype']
            page._len = dset.len()
            return page

    @classmethod
    def create(self, data):
        """
        creates a new group.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError
        
        if group is None:
            group = f"/page/{Page.new_id()}"

        if data.dtype.char == 'O':
            return MixedType(group, data)
        elif data.dtype.char == 'U':
            return StringType(group, data)
        elif data.dtype.char in {'ld'}:
            return SimpleType(group, data)
        else:
            raise NotImplementedError(f"method missing for {data.dtype.char}")

    def __len__(self):
        return self._len

    def read(self):
        with h5py.File(self.path, READONLY) as h5:
            dset = h5[self.group]
            self.dtype = dset.dtype
            self.source_type = dset.attrs['datatype']
            self._len = dset.len()

    def update(self):
        raise NotImplementedError("subclasses must implement this method.")
    
    def delete(self):
        raise NotImplementedError("subclasses must implement this method.")

    def __getitem__(self, item):
        raise NotImplementedError("subclasses must implement this method.")

    def __setitem__(self, index, value):
        raise NotImplementedError("subclasses must implement this method.")

    def append(self, value):
        raise NotImplementedError("subclasses must implement this method.")

    def extend(self, values):
        raise NotImplementedError("subclasses must implement this method.")


class SimpleType(Page):
    def __init__(self, group, data=None):
        super().__init__(group, data)
        if data:
            self.create(data)
    
    def create(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError

        with h5py.File(self.path, READWRITE) as h5:
            if self.group in h5:
                raise ValueError("page already exists")
            dset = h5.create_dataset(name=self.group, 
                                     data=self.data,  # data is now HDF5 compatible.
                                     dtype=self.stored_type,  # the HDF5 stored dtype may require unpacking using dtypes if they are different.
                                     maxshape=(None,),  # the stored data is now extendible / resizeable.
                                     chunks=HDF5_Config.H5_PAGE_SIZE)  # pages are chunked, so that IO block will be limited.
            dset.attrs['datatype'] = self.source_type
            dset.attrs['encoding'] = self.encoding

    def __getitem__(self, item):
        if item is None:
            item = slice(0,None,1)
        elif not isinstance(item, slice):
            raise TypeError

        with h5py.File(self.path, READONLY) as h5:
            dset = h5[self.group]
            return dset[item]

    def __setitem__(self, index, value):
        raise NotImplementedError("subclasses must implement this method.")

    def append(self, value):
        raise NotImplementedError("subclasses must implement this method.")

    def extend(self, values):
        raise NotImplementedError("subclasses must implement this method.")


class StringType(Page):
    def __init__(self, group, data=None, encoding=HDF5_Config.H5_ENCODING):
        super().__init__(group, data, encoding)
        if data:
            self.create(data)

    def create(self, data):
        source_type = str.__name__
        data = data.astype(bytes)
        stored_type = h5py.string_dtype(encoding=HDF5_Config.H5_ENCODING)

        with h5py.File(self.path, READWRITE) as h5:
            if self.group in h5:
                raise ValueError("page already exists")
            dset = h5.create_dataset(name=self.group, 
                                     data=data,  # data is now HDF5 compatible.
                                     dtype=stored_type,  # the HDF5 stored dtype may require unpacking using dtypes if they are different.
                                     maxshape=(None,),  # the stored data is now extendible / resizeable.
                                     chunks=HDF5_Config.H5_PAGE_SIZE)  # pages are chunked, so that IO block will be limited.
            dset.attrs['datatype'] = source_type
            dset.attrs['encoding'] = HDF5_Config.H5_ENCODING

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
        raise NotImplementedError("subclasses must implement this method.")

    def extend(self, values):
        raise NotImplementedError("subclasses must implement this method.")


class MixedType(Page):
    def __init__(self, group, data=None):
        super().__init__(group, data)
        self.type_group = None
        self.type_array = None
        if data:
            self.create(data)
    
    def create(self, data):
        with h5py.File(self.path, READWRITE) as h5:
            if self.group in h5:
                raise ValueError("page already exists")
            if not isinstance(data, np.ndarray):
                raise TypeError

            type_code = DataTypes.type_code
            self.type_group = f"{self.group}_types"
            self.type_array = np.array( [ type_code(v) for v in data.tolist() ] )
            
            self.source_type = 'typearray'
            byte_function = DataTypes.to_bytes
            data = np.array( [byte_function(v) for v in data.tolist()] )
            self.stored_type = h5py.string_dtype(encoding=HDF5_Config.H5_ENCODING)

            dset = h5.create_dataset(name=self.group, 
                                     data=data,  # data is now HDF5 compatible.
                                     dtype=self.stored_type,  # the HDF5 stored dtype may require unpacking using dtypes if they are different.
                                     maxshape=(None,),  # the stored data is now extendible / resizeable.
                                     chunks=HDF5_Config.H5_PAGE_SIZE)  # pages are chunked, so that IO block will be limited.
            dset.attrs['datatype'] = self.source_type
            dset.attrs['encoding'] = self.encoding
            
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
            
            type_group = f"{self.group}_types"
            type_array = h5[type_group][item]  # includes the page id
            type_functions = DataTypes.from_type_code

            match = np.array( [type_functions(v,type_code) for v,type_code in zip(match, type_array)] )
            return match
            
    def __setitem__(self, index, value):
        raise NotImplementedError("subclasses must implement this method.")

    def append(self, value):
        raise NotImplementedError("subclasses must implement this method.")

    def extend(self, values):
        raise NotImplementedError("subclasses must implement this method.")
   
