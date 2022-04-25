

import psutil

print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)

mem = psutil.virtual_memory()
print(mem.available)  # 8,638,365,696

_memory = psutil.virtual_memory().available
_cpus = psutil.cpu_count()
_disk_space = psutil.disk_usage('/').free

print(_memory, _cpus, _disk_space)