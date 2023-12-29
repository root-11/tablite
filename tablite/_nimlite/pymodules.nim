import nimpy as nimpy

var
    isInit = false
    iSys: nimpy.PyObject
    iBuiltins: nimpy.PyObject
    iNumpy: nimpy.PyObject
    iDateTime: nimpy.PyObject
    iTablite: nimpy.PyObject
    iTabliteBase: nimpy.PyObject
    iTabliteConfig: nimpy.PyObject
    PyNoneClass: nimpy.PyObject

proc importPy(): void =
    if isInit:
        return

    iBuiltins = nimpy.pyBuiltinsModule()

    iSys = nimpy.pyImport("sys")
    iDateTime = nimpy.pyImport("datetime")
    iTablite = nimpy.pyImport("tablite")
    iTabliteBase = nimpy.pyImport("tablite.base")
    iTabliteConfig = nimpy.pyImport("tablite.config")
    iNumpy = nimpy.pyImport("numpy")

    PyNoneClass = iBuiltins.None.getattr("__class__")

    isInit = true

proc sys*(): nimpy.PyObject =
    importPy()

    return iSys

proc builtins*(): nimpy.PyObject =
    importPy()

    return iBuiltins

proc numpy*(): nimpy.PyObject =
    importPy()

    return iNumpy

proc datetime*(): nimpy.PyObject =
    importPy()

    return iDateTime

proc tablite*(): nimpy.PyObject =
    importPy()

    return iTablite

proc tabliteBase*(): nimpy.PyObject =
    importPy()

    return iTabliteBase

proc tabliteConfig*(): nimpy.PyObject =
    importPy()

    return iTabliteConfig


proc isNone*(py: PyObject): bool {.inline.} =
    return builtins().isinstance(py, PyNoneClass).to(bool)
