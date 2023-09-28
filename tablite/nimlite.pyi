from typing import Literal, Type
from tqdm import tqdm as _tqdm
from tablite import Table

def text_reader(
        T: Type[Table],
        pid: str, path: str,
        encoding: Literal["ENC_UTF8"]|Literal["ENC_UTF16"]|Literal["ENC_WIN1250"] = "ENC_UTF8",
        *,
        first_row_has_headers: bool = True, header_row_index: int = 0,
        columns: list[str]|None = None,
        start: int|None = None, limit: int|None = None,
        guess_datatypes: bool = False,
        newline: str = '\n', delimiter: str = ',', text_qualifier: str = '"',
        quoting: str, strip_leading_and_tailing_whitespace: bool = True,
        tqdm=_tqdm
    ) -> Table:
    pass