import os


HDF5_IMPORT_ROOT = "__h5_import"  # the hdf5 base name for imports. f.x. f['/__h5_import/column A']
HDF5_PAGE_ROOT = "__H5_page"
HDF5_COLUMN_ROOT = "__H5_column"
HDF5_TABLE_ROOT = "__H5_table"
HDF5_CACHE_DIR = os.getcwd()
HDF5_CACHE_FILE = "tablite_cache.hdf5"
HDF5_PAGE_SIZE = 1_000_000
MEMORY_USAGE_CEILING = 0.9 # 90%
