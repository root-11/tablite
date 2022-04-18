import os


HDF5_IMPORT_ROOT = "__h5_import"  # the hdf5 base name for imports. f.x. f['/__h5_import/column A']
MEMORY_MANAGER_CACHE_DIR = os.getcwd()
MEMORY_MANAGER_CACHE_FILE = "tablite_cache.hdf5"
MEMORY_MANAGER_PAGE_SIZE = 1_000_000
MEMORY_USAGE_CEILING = 0.9 # 90%
