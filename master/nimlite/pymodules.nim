from std/os import getEnv
from std/strutils import split
from std/sugar import collect
import nimpy
import std/options

type PyModule[T] {.requiresInit.} = ref object of RootObj
    module*: nimpy.PyObject
    classes*: T

type PyDeepModule[T, K] {.requiresInit.} = ref object of PyModule[T]
    modules*: K

type PyEmpty = object
type PyEmptyModule = ref object of PyModule[PyEmpty]

type PyNumpy {.requiresInit.} = object
    NdArrayClass*: nimpy.PyObject

type PyBuiltins {.requiresInit.} = object
    NoneTypeClass*: nimpy.PyObject
    DictClass*: nimpy.PyObject
    ListClass*: nimpy.PyObject
    BoolClass*: nimpy.PyObject
    IntClass*: nimpy.PyObject
    FloatClass*: nimpy.PyObject
    StrClass*: nimpy.PyObject

type PyDatetime {.requiresInit.} = object
    DateClass*: nimpy.PyObject
    TimeClass*: nimpy.PyObject
    DateTimeClass*: nimpy.PyObject

type PyTablite {.requiresInit.} = object
    TableClass*: nimpy.PyObject

type PyMplite {.requiresInit.} = object
    TaskManager*: nimpy.PyObject

type PyTabliteConfig {.requiresInit.} = object
    Config*: nimpy.PyObject

type PyTabliteBase {.requiresInit.} = object
    ColumnClass*: nimpy.PyObject
    SimplePageClass*: nimpy.PyObject

type PyTabliteSubModules {.requiresInit.} = object
    config*: PyModule[PyTabliteConfig]
    base*: PyModule[PyTabliteBase]

type PyTqdm {.requiresInit.} = object
    TqdmClass*: nimpy.PyObject

type PyModules {.requiresInit.} = object
    sys*: PyEmptyModule
    builtins*: PyModule[PyBuiltins]
    datetime*: PyModule[PyDatetime]
    tablite*: PyDeepModule[PyTablite, PyTabliteSubModules]
    numpy*: PyModule[PyNumpy]
    mplite*: PyModule[PyMplite]
    nimlite*: PyEmptyModule
    tqdm*: PyModule[PyTqdm]
    math*: PyEmptyModule

proc newModule[K, T](Class: typedesc[K], module: nimpy.PyObject, classes: T): K {.inline.} = Class(module: module, classes: classes)
proc newModule[K, T1, T2](Class: typedesc[K], module: nimpy.PyObject, classes: T1, modules: T2): K {.inline.} = Class(module: module, classes: classes, modules: modules)
proc newEmptyModule(module: nimpy.PyObject): PyEmptyModule {.inline.} = PyEmptyModule.newModule(module, PyEmpty())

var py = none[PyModules]()

proc importPy(): void =
    if py.isSome:
        return

    let envs = getEnv("NIM_PYTHON_MODULES", "").split(":")
    let iSys = nimpy.pyImport("sys")

    echo iSys.path

    discard iSys.path.extend(envs)

    let iBuiltins = nimpy.pyBuiltinsModule()

    let iDateTime = nimpy.pyImport("datetime")
    let iTablite = nimpy.pyImport("tablite")
    let iTabliteBase = nimpy.pyImport("tablite.base")
    let iTabliteConfig = nimpy.pyImport("tablite.config")
    let iMplite = nimpy.pyImport("mplite")
    let iNumpy = nimpy.pyImport("numpy")
    let iTqdm = nimpy.pyImport("tqdm")
    let iNimlite = nimpy.pyImport("tablite.nimlite")
    let iMath = nimpy.pyImport("math")

    let iPyBuiltins = PyBuiltins(
        NoneTypeClass: iBuiltins.None.getattr("__class__"),
        DictClass: iBuiltins.getattr("dict"),
        ListClass: iBuiltins.getattr("list"),
        BoolClass: iBuiltins.getattr("bool"),
        IntClass: iBuiltins.getattr("int"),
        FloatClass: iBuiltins.getattr("float"),
        StrClass: iBuiltins.getattr("str"),
    )

    let iPyDateTime = PyDatetime(
        DateClass: iDateTime.date,
        TimeClass: iDateTime.time,
        DateTimeClass: iDateTime.datetime,
    )

    let iPyTablite = PyTablite(TableClass: iTablite.Table)
    let iPyTabliteConf = PyTabliteConfig(Config: iTabliteConfig.Config)
    let iPyTabliteBase = PyTabliteBase(ColumnClass: iTabliteBase.Column, SimplePageClass: iTabliteBase.SimplePage)
    let iPyTabliteSub = PyTabliteSubModules(
        config: PyModule[PyTabliteConfig].newModule(iTabliteConfig, iPyTabliteConf),
        base: PyModule[PyTabliteBase].newModule(iTabliteBase, iPyTabliteBase)
    )

    let iPyNumpy = PyNumpy(NdArrayClass: iBuiltins.getattr(iNumpy, "array"))
    let iPyMplite = PyMplite(TaskManager: iBuiltins.getattr(iMplite, "TaskManager"))
    let iPyTqdm = PyTqdm(TqdmClass: iBuiltins.getattr(iTqdm, "tqdm"))

    let pyModules = PyModules(
        sys: newEmptyModule(iSys),
        builtins: PyModule[PyBuiltins].newModule(iBuiltins, iPyBuiltins),
        numpy: PyModule[PyNumpy].newModule(iNumpy, iPyNumpy),
        datetime: PyModule[PyDatetime].newModule(iDateTime, iPyDateTime),
        tablite: PyDeepModule[PyTablite, PyTabliteSubModules].newModule(iTablite, iPyTablite, iPyTabliteSub),
        mplite: PyModule[PyMplite].newModule(iMplite, iPyMplite),
        nimlite: newEmptyModule(iNimlite),
        tqdm: PyModule[PyTqdm].newModule(iTqdm, iPyTqdm),
        math: newEmptyModule(iMath),
    )

    py = some(pyModules)


proc modules*(): PyModules =
    importPy()

    return py.get


proc isinstance*(inst: PyModule[PyBuiltins], obj: PyObject, other: nimpy.PyObject): bool {.inline.} = inst.module.isinstance(obj, other).to(bool)
proc getAttr*(inst: PyModule[PyBuiltins], attr: string): PyObject {.inline.} = inst.module.getattr(inst.module, attr)
proc getAttr*(inst: PyModule[PyBuiltins], obj: PyObject, attr: string): PyObject {.inline.} = inst.module.getattr(obj, attr)
proc getType*(inst: PyModule[PyBuiltins], obj: PyObject): PyObject {.inline.} = inst.module.type(obj)
proc getTypeName*(inst: PyModule[PyBuiltins], obj: PyObject): string {.inline.} = inst.getAttr(inst.getType(obj), "__name__").to(string)
proc toStr*(inst: PyModule[PyBuiltins], obj: PyObject): string {.inline.} = inst.module.str(obj).to(string)
proc toRepr*(inst: PyModule[PyBuiltins], obj: PyObject): string {.inline.} = inst.module.repr(obj).to(string)
proc getLen*(inst: PyModule[PyBuiltins], obj: PyObject): int {.inline.} = inst.module.len(obj).to(int)

proc fromFile*(inst: PyModule[PyTablite], path: string): PyObject {.inline.} = inst.classes.TableClass.from_file(path)
proc collectPages*(inst: PyModule[PyTabliteBase], column: PyObject): seq[string] {.inline.} =
    let builtins = modules().builtins
    
    if not builtins.isinstance(column, inst.classes.ColumnClass):
        raise newException(ValueError, "not a column")

    return collect:
        for p in column.pages:
            builtins.toStr(p.path)

proc isinstance*(self: PyModules, obj: PyObject, other: nimpy.PyObject): bool {.inline.} = self.builtins.isinstance(obj, other)
proc getAttr*(self: PyModules, obj: PyObject, attr: string): PyObject {.inline.} = self.builtins.getAttr(obj, attr)
proc getAttr*(self: PyModules, attr: string): PyObject {.inline.} = self.builtins.getAttr(attr)
proc getType*(self: PyModules, obj: PyObject): PyObject {.inline.} = self.builtins.getType(obj)
proc getTypeName*(self: PyModules, obj: PyObject): string {.inline.} = self.builtins.getTypeName(obj)
proc toStr*(self: PyModules, obj: PyObject): string {.inline.} = self.builtins.toStr(obj)
proc toRepr*(self: PyModules, obj: PyObject): string {.inline.} = self.builtins.toRepr(obj)
proc getLen*(self: PyModules, obj: PyObject): int {.inline.} = self.builtins.getLen(obj)


proc isNone*(obj: PyObject): bool {.inline.} = modules().builtins.isinstance(obj, py.get.builtins.classes.NoneTypeClass)
