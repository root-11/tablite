import os


# HDF5_IMPORT_ROOT = "__h5_import"  # the hdf5 base name for imports. f.x. f['/__h5_import/column A']
# ^--- disabled. Use page id instead.
HDF5_PAGE_ROOT = "page"
HDF5_COLUMN_ROOT = "column"
HDF5_TABLE_ROOT = "table"
HDF5_CACHE_DIR = os.getcwd()
HDF5_CACHE_FILE = "tablite_cache.hdf5"
HDF5_PAGE_SIZE = 1_000_000
MEMORY_USAGE_CEILING = 0.9 # 90%
