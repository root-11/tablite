from collections import defaultdict
from tablite2 import Table, Column


def setup():
    pass

def teardown():
    pass

def test_000_columns():  
    pass

def test_001_memory_manager():
    pass

def test_002_task_manager():
    pass

def test_003_worker_functions():  # single proc
    pass

def test_004_multiproc_worker_functions():  # multi proc
    pass

def test_005_top_level_api():
    """
    Table
        .import  creates .hdf5 table
            includes import settings in hash.
        .load  <-- loads external references
        .source {name: (for each column hdf5 source + index)}
        .view  (read slice of source from index)

            sort AZ, ZA  <--- view only! the source doesn't change.
            unique values
            filter by condition [
                is empty, is not empty, 
                text {contains, does not contain, starts with, ends with, is exactly},
                date {is, is before, is after}
                value is {> >= < <= == != between, not between}
                formula (uses eval)
            ]
            filter by values [ unique values ]
            
        .materialize  (create hdf5 from source) 

        .__iter__  reads the view  -- ops uses __iter__ 
    """
    tbl1 = Table.import_file('d:\remove_duplicates.csv', table_key='2345eafd2faf')  # table_key is from datamap or just a counter in the script.
    tbl2 = Table.load_file('d:\remove_duplicates.csv.hdf5')  # no additional storage is needed because the key is the same.
    tbl3 = Table.from_json('d:\some.json')
    tbl3 = tbl1 + tbl2   # append
    tb3_copy = tbl3.copy()  # virtual dataset.
    tbl4 = tbl3.sort(date=True, order=False)  # date ascending, order descending.
    tbl5 = tbl1.join(tbl2, left_keys=['date'], right_key=['date'], left_columns=['date', 'order'], right_columns=['quantity'], type='left')
    tbl6 = tbl1.lookup(tbl2, left_keys=['date'], right_key=['date'], left_columns=['date', 'order'], right_columns=['quantity'], expr='==')
    dates,quantities = tbl6['date'], tbl6['quantity']
    
    def pct(A,B):  # custom ops.
        d = defaultdict(int)
        for a,b in zip(A,B):
            d[a]+=b        
        return [b/d[a] for a,b in zip(A,B)]

    tbl6['pct'] = Column(dtype=np.float, data=pct(dates,quantities))  # adding column to tbl6.
    # tbl6[int]  --> row
    # tbl6[slice] --> n rows
    # tbl6[text]  --> column
    # tbl6[args]  --> columns if text, rows if slice or int
    
    tbl7 = tbl2.filter('date')


def test_006_callisto_guarantee():
    """
    UPLOAD = HDF5 file
    APPEND = virtual dataset
    SORT = make sort index + autosave to HDF5
    GROUPBY = HDF5 file
    PIVOT = GROUPBY view.
    SELECTOR = virtual dataset OR HDF5 file
    JOIN = paired index + autosave to HDF5
    FILTER = mask + autosave to HDF5
    LOOKUP = paired index
    OPERATION = HDF5 update column or create new column.
    """
    pass


if __name__ == "__main__":
    setup()

    for k,v in {k:v for k,v in sorted(globals().items()) if k.startswith('test') and callable(v)}.items():
        print(20 * "-" + k + "-" * 20)
        v()

    teardown()
