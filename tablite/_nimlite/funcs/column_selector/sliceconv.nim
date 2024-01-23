import std/[os, tables, paths, enumerate, sequtils]
from std/sugar import collect
from std/strutils import parseInt
import nimpy as nimpy
import ../../numpy
import ../../pytypes
import mask
from ../../utils import generate_random_string
from pagecasters import convertBasicPage
import infos



proc putPage(page: BaseNDArray, infos: var Table[string, nimpy.PyObject], colName: string, col: ColSliceInfo): void {.inline.} =
    let (dir, pid) = col

    infos[colName] = newPyPage(pid, dir, page.len, page.getPageTypes())

proc finalizeSlice(indices: var seq[int], column_names: seq[string], infos: var Table[string, nimpy.PyObject], cast_paths: var Table[string, (Path, Path, bool)], pages: var seq[(string, nimpy.PyObject)], result_info: ColInfo): void =
    if indices.len == 0:
        return

    for col_name in column_names:
        let (src_path, dst_path, is_tmp) = cast_paths[col_name]
        var cast_data = readNumpy(string src_path)

        if cast_data.len != indices.len:
            cast_data = cast_data[indices]
            cast_data.putPage(infos, col_name, result_info[col_name])
            cast_data.save(string dst_path)
        elif src_path != dst_path and is_tmp:
            moveFile(string src_path, string dst_path)

        pages.add((col_name, infos[col_name]))

proc toColSliceInfo(path: Path): ColSliceInfo =
    let workdir = string path.parentDir.parentDir
    let pid = parseInt(string path.extractFilename.changeFileExt(""))

    return (workdir, pid)

proc doSliceConvert*(dir_pid: Path, page_size: int, columns: Table[string, string], reject_reason_name: string, res_pass: ColInfo, res_fail: ColInfo, desired_column_map: OrderedTable[string, DesiredColumnInfo], column_names: seq[string], is_correct_type: Table[string, bool]): (seq[(string, nimpy.PyObject)], seq[(string, nimpy.PyObject)]) =
    var cast_paths_pass = initTable[string, (Path, Path, bool)]()
    var cast_paths_fail = initTable[string, (Path, Path, bool)]()
    var page_infos_pass = initTable[string, nimpy.PyObject]()
    var page_infos_fail = initTable[string, nimpy.PyObject]()
    var pages_pass = newSeq[(string, nimpy.PyObject)]()
    var pages_fail = newSeq[(string, nimpy.PyObject)]()

    try:
        let page_paths = collect(initTable()):
            for (key, path) in columns.pairs:
                {key: path}

        let workdir = dir_pid / Path("processing")

        createDir(string workdir)

        var valid_mask = newSeq[Mask](page_size)
        var reason_lst = newSeq[string](page_size)

        for (k, v) in page_paths.pairs:
            let (wd, pid) = res_fail[k]
            cast_paths_fail[k] = (Path v, Path(wd) / Path("pages") / Path($pid & ".npy"), false)

        let (rj_wd, rj_pid) = res_fail[reject_reason_name]
        let reject_reason_path = Path(rj_wd) / Path("pages") / Path($rj_pid & ".npy")
        cast_paths_fail[reject_reason_name] = (reject_reason_path, reject_reason_path, false)

        for (desired_name, desired_info) in desired_column_map.pairs:
            let original_name = desired_info.original_name
            let original_path = Path page_paths[original_name]
            let sz_data = getPageLen(string original_path)
            let original_data = readNumpy(string original_path)

            assert valid_mask.len >= sz_data, "Invalid mask size"

            let already_cast = is_correct_type[desired_name]

            original_data.putPage(page_infos_fail, original_name, original_path.toColSliceInfo)

            if already_cast:
                # we already know the type, just set the mask
                for i in 0..<sz_data:
                    if valid_mask[i] == INVALID:
                        continue

                    valid_mask[i] = VALID

                cast_paths_pass[desired_name] = (original_path, original_path, false)
                original_data.putPage(page_infos_pass, desired_name, original_path.toColSliceInfo)
                continue

            var cast_path: Path
            var path_exists = true

            while path_exists:
                cast_path = workdir / Path(generate_random_string(5) & ".npy")
                path_exists = fileExists(string cast_path)

            let (workdir, pid) = res_pass[desired_name]
            let pagedir = Path(workdir) / Path("pages")
            let dst_path = pagedir / Path($pid & ".npy")

            cast_paths_pass[desired_name] = (cast_path, dst_path, true)

            let desired_type = desired_info.`type`
            let allow_empty = desired_info.allow_empty

            var converted_page: BaseNDArray

            template castPage(T: typedesc) = T(original_data).convertBasicPage(
                desired_type, valid_mask, reason_lst, allow_empty,
                original_name, desired_name
            )

            case original_data.kind:
            of K_BOOLEAN: converted_page = BooleanNDArray.castPage
            of K_INT8: converted_page = Int8NDArray.castPage
            of K_INT16: converted_page = Int16NDArray.castPage
            of K_INT32: converted_page = Int32NDArray.castPage
            of K_INT64: converted_page = Int64NDArray.castPage
            of K_FLOAT32: converted_page = Float32NDArray.castPage
            of K_FLOAT64: converted_page = Float64NDArray.castPage
            of K_UNICODE: converted_page = UnicodeNDArray.castPage
            of K_DATE: converted_page = DateNDArray.castPage
            of K_DATETIME: converted_page = DateTimeNDArray.castPage
            of K_OBJECT: converted_page = ObjectNDArray.castPage

            converted_page.putPage(page_infos_pass, desired_name, res_pass[desired_name])
            converted_page.save(string cast_path)

        var mask_slice = 0..<unusedMaskSearch(valid_mask)

        valid_mask = valid_mask[mask_slice]

        var invalid_indices = newSeqOfCap[int](valid_mask.len shr 2) # quarter seems okay
        var valid_indices = newSeqOfCap[int](valid_mask.len - (valid_mask.len shr 2))

        reason_lst = collect:
            for (i, m) in enumerate(valid_mask):
                if m != Mask.INVALID:
                    valid_indices.add(i)
                    continue

                invalid_indices.add(i)
                reason_lst[i]

        valid_indices.finalizeSlice(toSeq(desired_column_map.keys), page_infos_pass, cast_paths_pass, pages_pass, res_pass)
        invalid_indices.finalizeSlice(toSeq(columns.keys), page_infos_fail, cast_paths_fail, pages_fail, res_fail)

        if reason_lst.len > 0:
            let (dirpid, pid) = res_fail[reject_reason_name]
            let pathpid = string (Path(dirpid) / Path("pages") / Path($pid & ".npy"))
            let page = newNDArray(reason_lst)

            page.save(pathpid)
            page.putPage(page_infos_fail, reject_reason_name, res_fail[reject_reason_name])

            pages_fail.add((reject_reason_name, page_infos_fail[reject_reason_name]))

    finally:
        for (cast_path, _, is_tmp) in cast_paths_pass.values:
            if not is_tmp:
                continue
            discard tryRemoveFile(string cast_path)

    return (pages_pass, pages_fail)
