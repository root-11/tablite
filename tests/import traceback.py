import traceback
import io

try:
    1/0
except Exception as e:
    f = io.StringIO()
    traceback.print_exc(limit=3, file=f)
    f.seek(0)
    x = f.read()
    f.close()

print(">>>>>>>", x)