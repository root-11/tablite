from std/os import getEnv
from std/strutils import split
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
    iMplite: nimpy.PyObject
    iNimlite: nimpy.PyObject
    PyNoneClass*: nimpy.PyObject

proc importPy(): void =
    if isInit:
        return

    let envs = getEnv("NIM_PYTHON_MODULES", "").split(":")
    iSys = nimpy.pyImport("sys")
    
    discard iSys.path.extend(envs)

    echo $iSys.path

    iBuiltins = nimpy.pyBuiltinsModule()

    iDateTime = nimpy.pyImport("datetime")
    iTablite = nimpy.pyImport("tablite")
    iTabliteBase = nimpy.pyImport("tablite.base")
    iTabliteConfig = nimpy.pyImport("tablite.config")
    iMplite = nimpy.pyImport("mplite")
    iNumpy = nimpy.pyImport("numpy")
    iNimlite = nimpy.pyImport("tablite.nimlite")

    PyNoneClass = iBuiltins.None.getattr("__class__")

    isInit = true

proc sys*(): nimpy.PyObject =
    importPy()

    return iSys

proc nimlite*(): nimpy.PyObject =
    importPy()

    return iNimlite

proc builtins*(): nimpy.PyObject =
    importPy()

    return iBuiltins

proc numpy*(): nimpy.PyObject =
    importPy()

    return iNumpy

proc mplite*(): nimpy.PyObject =
    importPy()

    return iMplite

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
