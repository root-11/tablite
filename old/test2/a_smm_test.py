from multiprocessing.managers import SharedMemoryManager, SyncManager
from multiprocessing import Process


class Worker(Process):
    def __init__(self, tq,rq,ref_count):
        



class RefCountMgr(object):
    def __init__(self) -> None:
        self.smm = SharedMemoryManager()
        self.smm.start()
        self.sync_mgr = SyncManager()
        self.sync_mgr.start()
        self.ref_count = self.sync_mgr.dict()
    def __setitem__(self,key,value):
        self.ref_count[key]=value
    def __delitem__(self, key):
        del self.ref_count[key]
    def __getitem__(self,key):
        return self.ref_count[key]
    

if __name__ == "__main__":
    REFCOUNT = RefCountMgr()