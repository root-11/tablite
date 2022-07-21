import pathlib, tempfile


# The default location for the storage
H5_STORAGE = pathlib.Path(tempfile.gettempdir()) / "tablite.hdf5"
# to overwrite first import the config class:
# >>> from tablite.config import Config
# >>> Config.H5_STORAGE = /this/new/location
# then import the Table class 
# >>> from tablite import Table
# for every new table or record this path will be used.

H5_PAGE_SIZE = 1_000_000  # sets the page size limit.

H5_ENCODING = 'UTF-8'  # sets the page encoding when using bytes

SINGLE_PROCESSING_LIMIT = 1_000_000  
# when the number of fields (rows x columns) 
# exceed this value, multiprocessing is used.

TEMPDIR = pathlib.Path(tempfile.gettempdir()) / 'tablite-tmp'
if not TEMPDIR.exists():
    TEMPDIR.mkdir()
# tempdir for file_reader and other temporary files.





