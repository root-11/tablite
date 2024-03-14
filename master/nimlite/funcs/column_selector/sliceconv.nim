import std/[os, tables, paths, enumerate, sequtils]
from std/sugar import collect
import nimpy as nimpy
import ../../numpy
import ../../pytypes
import mask
from ../../utils import generateRandomString
from pagecasters import convertBasicPage
import infos



proc putPage*(page: BaseNDArray, infos: var Table[string, nimpy.PyObject], colName: string, col: ColSliceInfo): void {.inline.} =
    let (dir, pid) = col

    infos[colName] = newPyPage(pid, dir, page.len, page.getPageTypes())

proc finalizeSlice(indices: var seq[int], columnNames: seq[string], infos: var Table[string, nimpy.PyObject], castPaths: var Table[string, (Path, Path, Path, bool)], pages: var seq[(string, nimpy.PyObject)], resultInfo: ColInfo): void =
    if indices.len == 0:
        return

    for colName in columnNames:
        let (srcPath, dstPath, dstSliced, isTmp) = castPaths[colName]
        var castData = readNumpy(string srcPath)

        if castData.len != indices.len:
            castData = castData[indices]
            castData.putPage(infos, colName, resultInfo[colName])
            castData.save(string dstSliced)
        elif srcPath != dstPath and isTmp:
            moveFile(string srcPath, string dstPath)

        pages.add((colName, infos[colName]))

proc toColSliceInfo(path: Path): ColSliceInfo =
    let workdir = string path.parentDir.parentDir
    let pid = string path.extractFilename.changeFileExt("")

    return (workdir, pid)

proc doSliceConvert*(dirPid: Path, pageSize: int, columns: Table[string, (string, nimpy.PyObject)], rejectReasonName: string, resPass: ColInfo, resFail: ColInfo, desiredColumnMap: OrderedTable[string, DesiredColumnInfo], columnNames: seq[string], isCorrectType: Table[string, bool]): (seq[(string, nimpy.PyObject)], seq[(string, nimpy.PyObject)]) =
    var castPathsPass = initTable[string, (Path, Path, Path, bool)]()
    var castPathsFail = initTable[string, (Path, Path, Path, bool)]()
    var pageInfosPass = initTable[string, nimpy.PyObject]()
    var pageInfosFail = initTable[string, nimpy.PyObject]()
    var pagesPass = newSeq[(string, nimpy.PyObject)]()
    var pagesFail = newSeq[(string, nimpy.PyObject)]()

    try:
        var pagePaths = initTable[string, string]()
        for (colName, page) in columns.pairs:
            let (path, pageObj) = page
            
            pagePaths[colName] = path
            pageInfosFail[colName] = pageObj

        let processingWorkdir = dirPid / Path("processing")

        createDir(string processingWorkdir)

        var validMask = newSeq[Mask](pageSize)
        var reasonLst = newSeq[string](pageSize)

        for (originalName, v) in pagePaths.pairs:
            let (wd, pid) = resFail[originalName]
            let dstPath = Path(wd) / Path("pages") / Path($pid & ".npy")

            castPathsFail[originalName] = (Path v, dstPath, dstPath, false)

        let (rjwd, rjpid) = resFail[rejectReasonName]
        let rejectReasonPath = Path(rjwd) / Path("pages") / Path($rjpid & ".npy")
        castPathsFail[rejectReasonName] = (rejectReasonPath, rejectReasonPath, rejectReasonPath, false)

        for (desiredName, desiredInfo) in desiredColumnMap.pairs:
            let originalName = desiredInfo.originalName
            let originalPath = Path pagePaths[originalName]
            let originalData = readNumpy(string originalPath)
            let szData = originalData.len

            assert validMask.len >= szData, "Invalid mask size"

            let alreadyCast = isCorrectType[desiredName]
            let (workdir, pid) = resPass[desiredName]
            let pagedir = Path(workdir) / Path("pages")
            let dstPath = pagedir / Path($pid & ".npy")

            if alreadyCast:
                # we already know the type, just set the mask
                for i in 0..<szData:
                    if validMask[i] == INVALID:
                        continue

                    validMask[i] = VALID

                castPathsPass[desiredName] = (originalPath, originalPath, dstPath, false)
                originalData.putPage(pageInfosPass, desiredName, originalPath.toColSliceInfo)
                continue

            var castPath: Path
            var pathExists = true

            while pathExists:
                castPath = processingWorkdir / Path(generateRandomString(5) & ".npy")
                pathExists = fileExists(string castPath)

            castPathsPass[desiredName] = (castPath, dstPath, dstPath, true)

            let desiredType = desiredInfo.`type`
            let allowEmpty = desiredInfo.allowEmpty

            var convertedPage: BaseNDArray

            template castPage(T: typedesc) = T(originalData).convertBasicPage(
                desiredType, validMask, reasonLst, allowEmpty,
                originalName, desiredName
            )

            case originalData.kind:
            of K_BOOLEAN: convertedPage = BooleanNDArray.castPage
            of K_INT8: convertedPage = Int8NDArray.castPage
            of K_INT16: convertedPage = Int16NDArray.castPage
            of K_INT32: convertedPage = Int32NDArray.castPage
            of K_INT64: convertedPage = Int64NDArray.castPage
            of K_FLOAT32: convertedPage = Float32NDArray.castPage
            of K_FLOAT64: convertedPage = Float64NDArray.castPage
            of K_STRING: convertedPage = UnicodeNDArray.castPage
            of K_DATE: convertedPage = DateNDArray.castPage
            of K_DATETIME: convertedPage = DateTimeNDArray.castPage
            of K_OBJECT: convertedPage = ObjectNDArray.castPage

            convertedPage.putPage(pageInfosPass, desiredName, resPass[desiredName])
            convertedPage.save(string castPath)

        var maskSlice = 0..<unusedMaskSearch(validMask)

        validMask = validMask[maskSlice]

        var validIndices = newSeqOfCap[int](validMask.len - (validMask.len shr 2))
        var invalidIndices = newSeqOfCap[int](validMask.len shr 2) # quarter seems okay

        reasonLst = collect:
            for (i, m) in enumerate(validMask):
                if m != Mask.INVALID:
                    validIndices.add(i)
                    continue

                invalidIndices.add(i)
                reasonLst[i]

        validIndices.finalizeSlice(toSeq(desiredColumnMap.keys), pageInfosPass, castPathsPass, pagesPass, resPass)
        invalidIndices.finalizeSlice(toSeq(columns.keys), pageInfosFail, castPathsFail, pagesFail, resFail)

        if reasonLst.len > 0:
            let (dirpid, pid) = resFail[rejectReasonName]
            let pathpid = string (Path(dirpid) / Path("pages") / Path($pid & ".npy"))
            let page = newNDArray(reasonLst)

            page.putPage(pageInfosFail, rejectReasonName, resFail[rejectReasonName])
            page.save(pathpid)

            pagesFail.add((rejectReasonName, pageInfosFail[rejectReasonName]))

    finally:
        for (castPath, _, _, isTmp) in castPathsPass.values:
            if not isTmp:
                continue
            discard tryRemoveFile(string castPath)

    return (pagesPass, pagesFail)
