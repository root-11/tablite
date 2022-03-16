import h5py
import numpy as np
import json

def descend_obj(obj,sep='  ', offset=''):
    """
    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    """
    if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
        if obj.attrs.keys():
                for k,v in obj.attrs.items():
                    print(offset, k,":",v)
        for key in obj.keys():
            print(offset, key,':',obj[key])
            
            descend_obj(obj[key],sep=sep, offset=offset+sep)
    elif type(obj)==h5py._hl.dataset.Dataset:
        for key in obj.attrs.keys():
            print(offset, key,':',obj.attrs[key])

def h5dump(path,group='/'):
    """
    print HDF5 file metadata

    group: you can give a specific group, defaults to the root group
    """
    print(f"{path} contents")
    with h5py.File(path,'r') as f:
         descend_obj(f[group])



filename = 'a.zip.h5'

f = h5py.File(filename, 'w')
print(f.name)

print(list(f.keys()))

grp1 = f.create_group("b.zip")  # no config
grp2 = f.create_group("b.zip/c.csv")

config = json.dumps({
    'import_as': 'csv',
    'newline': '\r\n',
    'text_qualifier':'"',
    'delimiter':",",
    'first_row_headers':True,
    'columns': {"col1": 'i8', "col2": 'int64'}
})

grp2.attrs['config']=config

dset = grp2.create_dataset("col1", dtype='i8', data=[1,2,3,4,5,6])

dset = grp2.create_dataset("col2", dtype='int64', data=[5,5,5,5,5,2**33])

grp3 = f.create_group("b.zip/x.xlsx/sheet1")
grp3.create_dataset("A", data=np.array([b'byte', b'stream']))
grp3.create_dataset("B", data=np.array([b'ascii', b'text', b'as bytes']))

grp4 = f.create_group("b.zip/x.xlsx/sheet2")
grp4.create_dataset("C", (200,), dtype='i8')
grp4.create_dataset("D", (200,), dtype='i8')

grp5 = f.create_group("c.zip/t.txt")
grp5.create_dataset('logs', (100,))

# grp6 = f.create_group("d.csv")
f.create_dataset('col_1', (9,))
f.create_dataset('col_2', (9,))
f.create_dataset('col_3', (9,))

f.close()

# Append to dataset
f = h5py.File(filename, 'a')
dset = f.create_dataset('/sha256sum', data=[2,5,6],chunks=True, maxshape=(None, ))
print(dset[:])
new_data = [3,8,4]
new_length = len(dset) + len(new_data)
dset.resize((new_length, ))
dset[-len(new_data):] = new_data
print(dset[:])

print(list(f.keys()))

f.close()

h5dump('a.zip.h5')

import pathlib
pathlib.Path(filename).unlink()
