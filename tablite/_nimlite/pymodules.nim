import nimpy as nimpy

var
    isInit = false
    iBuiltins: nimpy.PyObject
    iNumpy: nimpy.PyObject
    iDateTime: nimpy.PyObject
    PyNoneClass: nimpy.PyObject

proc importPy(): void =
    if isInit:
        return

    iBuiltins = nimpy.pyBuiltinsModule()
    iNumpy = nimpy.pyImport("numpy")
    iDateTime = nimpy.pyImport("datetime")
    
    PyNoneClass = iBuiltins.None.getattr("__class__")

    isInit = true

proc builtins*(): nimpy.PyObject =
    importPy()
    
    return iBuiltins

proc numpy*(): nimpy.PyObject =
    importPy()
    
    return iNumpy

proc datetime*(): nimpy.PyObject =
    importPy()
    
    return iDateTime