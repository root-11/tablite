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
        self._page_id = 0  # when loading from disk, increment this to max page.
    
    def next_page_id(self) -> int:
        self._page_id += 1
        return self._page_id

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
           
            for name in all_pages:  # add ref count for new connection.
                self.ref_counts[name] += 1
            for name in old_pages:  # remove duplicate ref count.
                self.ref_counts[name] -= 1
            
            # 3. determine new layout.
            dtypes = defaultdict(int)
            for dset_name in all_pages:
                dset = h5[dset_name]
                dtypes[dset.dtype] += dset.len()
            shape = sum(dtypes.values())
            L = [x for x in dtypes]
            L.sort(key=lambda x: x.itemsize, reverse=True)
            dtype = L[0]  # np.bytes

            # 4. create the layout.
            layout = h5py.VirtualLayout(shape=(shape,), dtype=dtype, maxshape=(None,), filename=self.path)
            a, b = 0, 0
            for page in all_pages:
                dset = h5[page]
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
        with h5py.File(self.path, READONLY) as h5:
            if group in h5:
                dset = h5[group]
                return [ group for _,_,group,_ in dset.virtual_sources() ]
        return []

    def reset_storage(self):
        with h5py.File(self.path, TRUNCATE) as h5:
            assert list(h5.keys()) == []

    def create_page(self, data):
        group = f"/page/{self.next_page_id()}"  # /page/{key} 

        self.ref_counts[group] += 1

        if not isinstance(data,np.ndarray):
            data = np.array(data)
        
        # type check
        type_group = None
        if data.dtype.char == 'O':  # datatype was non-native to HDF5, so utf-8 encoded bytes must used
            type_code = DataTypes.type_code
            type_group = f"{group}_types"
            type_array = np.array( [ type_code(v) for v in data.tolist() ] )

            source_type = 'typearray'
            byte_function = DataTypes.to_bytes
            data = np.array( [byte_function(v) for v in data.tolist()] )
            stored_type = h5py.string_dtype(encoding=HDF5_Config.H5_ENCODING)
            
        elif data.dtype.char == 'U':  # looks like text.
            source_type = str.__name__
            data = data.astype(bytes)
            stored_type = h5py.string_dtype(encoding=HDF5_Config.H5_ENCODING)
            
        else:  # uniform HDF5 & numpy compatible datatype.
            source_type = data.dtype.name
            stored_type = data.dtype
        
        # store using page id.
        with h5py.File(self.path, READWRITE) as h5:
            if group in h5:
                raise ValueError("page already exists")
            dset = h5.create_dataset(name=group, 
                                     data=data,  # data is now HDF5 compatible.
                                     dtype=stored_type,  # the HDF5 stored dtype may require unpacking using dtypes if they are different.
                                     maxshape=(None,),  # the stored data is now extendible / resizeable.
                                     chunks=HDF5_Config.H5_PAGE_SIZE)  # pages are chunked, so that IO block will be limited.
            dset.attrs['datatype'] = source_type
            dset.attrs['encoding'] = HDF5_Config.H5_ENCODING
            if type_group is not None:
                _ = h5.create_dataset(name=type_group, 
                                         data=type_array, 
                                         dtype=type_array.dtype, 
                                         maxshape=(None,), 
                                         chunks=HDF5_Config.H5_PAGE_SIZE)
        
        return group

    def get_data(self, group, item):
        with h5py.File(self.path, READONLY) as h5:
            dset = h5[group]
            if dset.dtype.char != 'O':  # it's a single type dataset.
                return dset[item]
            else:
                arrays = []
                target_range = range(*normalize_slice(length=dset.len(), item=item))
                # check if the slice is worth converting.
                start,end = 0,0
                for page in self.get_pages(group):
                    dset = h5[page]
                    end = dset.len()
                    search_range = range(start,end,1)
                    intercept_range = intercept(target_range, search_range)
                    start = end
                    if intercept_range == range(0):
                        continue

                    # fetch the slice and filter it.
                    search_slice = slice(intercept_range.start, intercept_range.stop, intercept_range.step)
                    match = dset[search_slice]
                    
                    source_type = dset.attrs['datatype']
                    encoding = dset.attrs['encoding']

                    if source_type == 'typearray':
                        type_group = f"{page}_types"
                        type_array = h5[type_group][search_slice]  # includes the page id
                        type_functions = DataTypes.from_type_code

                        match = np.array( [type_functions(v,type_code) for v,type_code in zip(match, type_array)] )
                    elif source_type == 'str':
                        match = np.array( [v.decode(encoding) for v in match] )
                    else:
                        raise NotImplemented

                    arrays.append(match)
                    
                return np.concatenate(arrays)

