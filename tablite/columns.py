import json
from abc import ABC

from tablite.datatypes import DataTypes
from tablite.stored_list import StoredList


class CommonColumn(ABC):
    def __init__(self, header, datatype, allow_empty, metadata=None):
        if not isinstance(header, str) and header != "":
            raise ValueError
        self.header = header
        if not isinstance(datatype, type):
            raise ValueError
        self.datatype = datatype
        if not isinstance(allow_empty, bool):
            raise TypeError
        self.allow_empty = allow_empty
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError
        self.metadata = metadata

    def _init(self, data):
        if isinstance(data, StoredList):
            self.extend(data)
        elif isinstance(data, (list, tuple)):
            for v in data:
                self.append(v)
        elif data is None:
            pass
        else:
            raise NotImplementedError(f"{type(data)} is not supported.")

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.header},{self.datatype},{self.allow_empty}) # ({len(self)} rows)"

    def __copy__(self):
        return self.copy()

    def copy(self):
        return self.__class__(self.header, self.datatype, self.allow_empty, metadata=self.metadata.copy(), data=self)

    def to_json(self):
        return json.dumps({
            'header': self.header,
            'datatype': self.datatype.__name__,
            'allow_empty': self.allow_empty,
            'metadata': self.metadata,
            'data': json.dumps([DataTypes.to_json(v) for v in self])
        })

    def type_check(self, value):
        """ helper that does nothing unless it raises an exception. """
        if value is None:
            if self.allow_empty:
                return True
            else:
                raise ValueError("None is not permitted.")
        elif isinstance(value, self.datatype):
            return True
        else:
            raise TypeError(f"{value} is not of type {self.datatype}")

    def append(self, value):
        self.type_check(value)
        super().append(value)

    def replace(self, values) -> None:
        assert isinstance(values, list)
        if len(values) != len(self):
            raise ValueError("input is not of same length as column.")
        if not all(self.type_check(v) for v in values):
            raise TypeError(f"input contains non-{self.datatype.__name__}")
        self.clear()
        self.extend(values)

    def __eq__(self, other):
        return all([
            self.header == other.header,
            self.datatype == other.datatype,
            self.allow_empty == other.allow_empty,
            len(self) == len(other),
            all(a == b for a, b in zip(self, other))
        ])

    def __setitem__(self, key, value):
        self.type_check(value)
        super().__setitem__(key, value)


class StoredColumn(CommonColumn, StoredList):  # MRO: CC first, then SL.
    """This is a sqlite backed mmaped list with headers and metadata."""

    def __init__(self, header, datatype, allow_empty, data=None, metadata=None, page_size=StoredList.default_page_size):
        CommonColumn.__init__(self, header, datatype, allow_empty, metadata=metadata)
        StoredList.__init__(self, page_size=page_size)
        if data is not None:
            self._init(data)

    @classmethod
    def from_json(cls, json_):
        j = json.loads(json_)
        j['datatype'] = dtype = getattr(DataTypes, j['datatype'])
        j['data'] = [DataTypes.from_json(v, dtype) for v in json.loads(j['data'])]
        return StoredColumn(**j)


class InMemoryColumn(CommonColumn, list):  # MRO: CC first, then list.
    """This is a list with headers and metadata."""

    def __init__(self, header, datatype, allow_empty, data=None, metadata=None):
        CommonColumn.__init__(self, header, datatype, allow_empty, metadata=metadata)
        list.__init__(self)  # then init the list attrs.
        if data is not None:
            self._init(data)

    @classmethod
    def from_json(cls, json_):
        j = json.loads(json_)
        j['datatype'] = dtype = getattr(DataTypes, j['datatype'])
        j['data'] = [DataTypes.from_json(v, dtype) for v in json.loads(j['data'])]
        return InMemoryColumn(**j)