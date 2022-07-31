from datetime import date, datetime, time, timedelta
from collections import defaultdict
import numpy as np


class DataTypes(object):
    # supported datatypes.
    int = int
    str = str
    float = float
    bool = bool
    date = date
    datetime = datetime
    time = time
    # timedelta = timedelta  # TODO ?

    numeric_types = {int, float, date, time, datetime}
    digits = '1234567890'
    decimals = set('1234567890-+eE.')
    integers = set('1234567890-+')
    nones = {'null', 'Null', 'NULL', '#N/A', '#n/a', "", 'None', None, np.nan}
    none_type = type(None)

    _type_codes ={
        type(None): 1,
        bool: 2,
        int: 3,
        float: 4,
        str: 5,
        bytes: 6,
        datetime: 7,
        date: 8,
        time: 9,
        timedelta: 10
    }

    @classmethod
    def type_code(cls, value): 
        if type(value) in cls._type_codes:
            return cls._type_codes[type(value)]
        elif hasattr(value,'dtype'):
            dtype = numpy_types.get(np.dtype(value).name)
            return cls._type_codes[dtype]
        else:
            raise TypeError(f"No handler for {type(value)}")
    
    def b_none(v):
        return b"None"
    def b_bool(v):
        return bytes(str(v), encoding='utf-8')
    def b_int(v):
        return bytes(str(v), encoding='utf-8')
    def b_float(v):
        return bytes(str(v), encoding='utf-8')
    def b_str(v):
        return v.encode('utf-8')
    def b_bytes(v):
        return v
    def b_datetime(v):
        return bytes(v.isoformat(), encoding='utf-8')
    def b_date(v):
        return bytes(v.isoformat(), encoding='utf-8')
    def b_time(v):
        return bytes(v.isoformat(), encoding='utf-8')
    def b_timedelta(v):
        return bytes(str(float(v.days + (v.seconds / (24*60*60)))), 'utf-8')
        
    bytes_functions = {
        type(None): b_none,
        bool: b_bool,
        int: b_int,
        float: b_float,
        str: b_str,
        bytes: b_bytes,
        datetime: b_datetime,
        date: b_date,
        time: b_time,
        timedelta: b_timedelta
    }

    @classmethod
    def to_bytes(cls, v):
        if type(v) in cls.bytes_functions:  # it's a python native type
            f = cls.bytes_functions[type(v)]
        elif hasattr(v, 'dtype'):  # it's a numpy/c type.
            dtype = numpy_types.get(np.dtype(v).name)
            f = cls.bytes_functions[dtype]
        else:
            raise TypeError(f"No handler for {type(v)}")
        return f(v)

    def _none(v):
        return None
    def _bool(v):
        return bool(v.decode('utf-8'))
    def _int(v):
        return int(v.decode('utf-8'))
    def _float(v):
        return float(v.decode('utf-8'))
    def _str(v):
        return v.decode('utf-8')
    def _bytes(v):
        return v
    def _datetime(v):
        return datetime.fromisoformat(v.decode('utf-8'))
    def _date(v):
        return date.fromisoformat(v.decode('utf-8'))
    def _time(v):
        return time.fromisoformat(v.decode('utf-8'))
    def _timedelta(v):
        days = float(v)
        seconds = 24 * 60 * 60 * ( float(v) - int( float(v) ) )
        return timedelta(int(days), seconds)
    
    type_code_functions = {
        1: _none,
        2: _bool,
        3: _int,
        4: _float,
        5: _str,
        6: _bytes,
        7: _datetime,
        8: _date,
        9: _time,
        10: _timedelta
    }

    pytype_from_type_code = {
        1: type(None),
        2: bool,
        3: int,
        4: float,
        5: str,
        6: bytes,
        7: datetime,
        8: date,
        9: time,
        10: timedelta,
    }

    @classmethod
    def from_type_code(cls, value, code):
        f = cls.type_code_functions[code]
        return f(value)

    date_formats = {  # Note: Only recognised ISO8601 formats are accepted.
        "NNNN-NN-NN": lambda x: date(*(int(i) for i in x.split("-"))),
        "NNNN-N-NN": lambda x: date(*(int(i) for i in x.split("-"))),
        "NNNN-NN-N": lambda x: date(*(int(i) for i in x.split("-"))),
        "NNNN-N-N": lambda x: date(*(int(i) for i in x.split("-"))),
        "NN-NN-NNNN": lambda x: date(*[int(i) for i in x.split("-")][::-1]),
        "N-NN-NNNN": lambda x: date(*[int(i) for i in x.split("-")][::-1]),
        "NN-N-NNNN": lambda x: date(*[int(i) for i in x.split("-")][::-1]),
        "N-N-NNNN": lambda x: date(*[int(i) for i in x.split("-")][::-1]),
        "NNNN.NN.NN": lambda x: date(*(int(i) for i in x.split("."))),
        "NNNN.N.NN": lambda x: date(*(int(i) for i in x.split("."))),
        "NNNN.NN.N": lambda x: date(*(int(i) for i in x.split("."))),
        "NNNN.N.N": lambda x: date(*(int(i) for i in x.split("."))),
        "NN.NN.NNNN": lambda x: date(*[int(i) for i in x.split(".")][::-1]),
        "N.NN.NNNN": lambda x: date(*[int(i) for i in x.split(".")][::-1]),
        "NN.N.NNNN": lambda x: date(*[int(i) for i in x.split(".")][::-1]),
        "N.N.NNNN": lambda x: date(*[int(i) for i in x.split(".")][::-1]),
        "NNNN/NN/NN": lambda x: date(*(int(i) for i in x.split("/"))),
        "NNNN/N/NN": lambda x: date(*(int(i) for i in x.split("/"))),
        "NNNN/NN/N": lambda x: date(*(int(i) for i in x.split("/"))),
        "NNNN/N/N": lambda x: date(*(int(i) for i in x.split("/"))),
        "NN/NN/NNNN": lambda x: date(*[int(i) for i in x.split("/")][::-1]),
        "N/NN/NNNN": lambda x: date(*[int(i) for i in x.split("/")][::-1]),
        "NN/N/NNNN": lambda x: date(*[int(i) for i in x.split("/")][::-1]),
        "N/N/NNNN": lambda x: date(*[int(i) for i in x.split("/")][::-1]),
        "NNNN NN NN": lambda x: date(*(int(i) for i in x.split(" "))),
        "NNNN N NN": lambda x: date(*(int(i) for i in x.split(" "))),
        "NNNN NN N": lambda x: date(*(int(i) for i in x.split(" "))),
        "NNNN N N": lambda x: date(*(int(i) for i in x.split(" "))),
        "NN NN NNNN": lambda x: date(*[int(i) for i in x.split(" ")][::-1]),
        "N N NNNN": lambda x: date(*[int(i) for i in x.split(" ")][::-1]),
        "NN N NNNN": lambda x: date(*[int(i) for i in x.split(" ")][::-1]),
        "N NN NNNN": lambda x: date(*[int(i) for i in x.split(" ")][::-1]),
        "NNNNNNNN": lambda x: date(*(int(x[:4]), int(x[4:6]), int(x[6:]))),
    }

    datetime_formats = {
        # Note: Only recognised ISO8601 formats are accepted.

        # year first
        'NNNN-NN-NNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x),  # -T
        'NNNN-NN-NNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x),

        'NNNN-NN-NN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, T=" "),  # - space
        'NNNN-NN-NN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, T=" "),

        'NNNN/NN/NNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/'),  # / T
        'NNNN/NN/NNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/'),

        'NNNN/NN/NN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', T=" "),  # / space
        'NNNN/NN/NN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', T=" "),

        'NNNN NN NNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd=' '),  # space T
        'NNNN NN NNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd=' '),

        'NNNN NN NN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd=' ', T=" "),  # space
        'NNNN NN NN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd=' ', T=" "),

        'NNNN.NN.NNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.'),  # dot T
        'NNNN.NN.NNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.'),

        'NNNN.NN.NN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.', T=" "),  # dot
        'NNNN.NN.NN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.', T=" "),


        # day first
        'NN-NN-NNNNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='-', T=' ', day_first=True),  # - T
        'NN-NN-NNNNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='-', T=' ', day_first=True),

        'NN-NN-NNNN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='-', T=' ', day_first=True),  # - space
        'NN-NN-NNNN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='-', T=' ', day_first=True),

        'NN/NN/NNNNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', day_first=True),  # / T
        'NN/NN/NNNNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', day_first=True),

        'NN/NN/NNNN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', T=' ', day_first=True),  # / space
        'NN/NN/NNNN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', T=' ', day_first=True),

        'NN NN NNNNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', day_first=True),  # space T
        'NN NN NNNNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', day_first=True),

        'NN NN NNNN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', day_first=True),  # space
        'NN NN NNNN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', day_first=True),

        'NN.NN.NNNNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.', day_first=True),  # space T
        'NN.NN.NNNNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.', day_first=True),

        'NN.NN.NNNN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.', day_first=True),  # space
        'NN.NN.NNNN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.', day_first=True),

        # compact formats - type 1
        'NNNNNNNNTNNNNNN': lambda x: DataTypes.pattern_to_datetime(x, compact=1),
        'NNNNNNNNTNNNN': lambda x: DataTypes.pattern_to_datetime(x, compact=1),
        'NNNNNNNNTNN': lambda x: DataTypes.pattern_to_datetime(x, compact=1),
        # compact formats - type 2
        'NNNNNNNNNN': lambda x: DataTypes.pattern_to_datetime(x, compact=2),
        'NNNNNNNNNNNN': lambda x: DataTypes.pattern_to_datetime(x, compact=2),
        'NNNNNNNNNNNNNN': lambda x: DataTypes.pattern_to_datetime(x, compact=2),
        # compact formats - type 3
        'NNNNNNNNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, compact=3),
    }

    @staticmethod
    def pattern_to_datetime(iso_string, ymd=None, T=None, compact=0, day_first=False):
        assert isinstance(iso_string, str)
        if compact:
            s = iso_string
            if compact == 1:  # has T
                slices = [(0, 4, "-"), (4, 6, "-"), (6, 8, "T"), (9, 11, ":"), (11, 13, ":"), (13, len(s), "")]
            elif compact == 2:  # has no T.
                slices = [(0, 4, "-"), (4, 6, "-"), (6, 8, "T"), (8, 10, ":"), (10, 12, ":"), (12, len(s), "")]
            elif compact == 3:  # has T and :
                slices = [(0, 4, "-"), (4, 6, "-"), (6, 8, "T"), (9, 11, ":"), (12, 14, ":"), (15, len(s), "")]
            else:
                raise TypeError
            iso_string = "".join([s[a:b] + c for a, b, c in slices if b <= len(s)])
            iso_string = iso_string.rstrip(":")

        if day_first:
            s = iso_string
            iso_string = "".join((s[6:10], "-", s[3:5], "-", s[0:2], s[10:]))

        if "," in iso_string:
            iso_string = iso_string.replace(",", ".")

        dot = iso_string[::-1].find('.')
        if 0 < dot < 10:
            ix = len(iso_string) - dot
            microsecond = int(float(f"0{iso_string[ix - 1:]}") * 10 ** 6)
            iso_string = iso_string[:len(iso_string) - dot] + str(microsecond).rjust(6, "0")
        if ymd:
            iso_string = iso_string.replace(ymd, '-', 2)
        if T:
            iso_string = iso_string.replace(T, "T")
        return datetime.fromisoformat(iso_string)

    @staticmethod
    def to_json(v):
        if hasattr(v, 'dtype'):
            pytype = numpy_types[v.dtype.name]
            v = pytype(v)
        if v is None:
            return v
        elif v is False:  # using isinstance(v, bool): won't work as False also is int of zero.
            return str(v)
        elif v is True:
            return str(v)
        elif isinstance(v, int):
            return v
        elif isinstance(v, str):
            return v
        elif isinstance(v, float):
            return v
        elif isinstance(v, datetime):
            return v.isoformat()
        elif isinstance(v, time):
            return v.isoformat()
        elif isinstance(v, date):
            return v.isoformat()
        elif isinstance(v, timedelta):
            return f"{v.days} days, {v.seconds + (v.microseconds / 1e6)} seconds"
        else:
            raise TypeError(f"The datatype {type(v)} is not supported.")

    @staticmethod
    def from_json(v, dtype):
        if v in DataTypes.nones:
            if dtype is str and v == "":
                return ""
            else:
                return None
        if dtype is int:
            return int(v)
        elif dtype is str:
            return str(v)
        elif dtype is float:
            return float(v)
        elif dtype is bool:
            if v == 'False':
                return False
            elif v == 'True':
                return True
            else:
                raise ValueError(v)
        elif dtype is date:
            return date.fromisoformat(v)
        elif dtype is datetime:
            return datetime.fromisoformat(v)
        elif dtype is time:
            return time.fromisoformat(v)
        elif dtype is timedelta:
            L = v.split(' ')
            days, seconds = int(L[0]), float(L[2])
            return timedelta(days, seconds)
        else:
            raise TypeError(f"The datatype {str(dtype)} is not supported.")

    # Order is very important!
    types = [datetime, date, time, int, bool, float, str]

    @staticmethod
    def guess_types(*values):
        """
        Attempts to guess the datatype for *values
        returns dict with matching datatypes and probabilities
        """

        d = defaultdict(int)
        probability = Rank(DataTypes.types[:])
        
        for value in values:
            if hasattr(value, 'dtype'):
                value = numpy_types[np.dtype(value).name](value)

            for dtype in probability:
                try:
                    _ = DataTypes.infer(value,dtype)
                    d[dtype] += 1 
                    probability.match(dtype)
                    break
                except (ValueError, TypeError):
                    pass
        if not d:
            d[str]=len(values)
        return {k:round(v/len(values),3) for k,v in d.items()}

    @staticmethod
    def guess(*values):
        """
        Makes a best guess the datatype for *values
        returns list of native python values
        """
        probability = Rank(*DataTypes.types[:])
        matches = [None for _ in values[0]]
        
        for ix, value in enumerate(values[0]):
            if hasattr(value, 'dtype'):
                value = numpy_types[np.dtype(value).name](value)
            for dtype in probability:
                try:
                    matches[ix] = DataTypes.infer(value,dtype)
                    probability.match(dtype)
                    break
                except (ValueError, TypeError):
                    pass
        return matches

    @classmethod
    def infer(cls, v, dtype):
        if v in DataTypes.nones:
            return None
        if dtype is int:
            return DataTypes._infer_int(v)
        elif dtype is str:
            return DataTypes._infer_str(v)
        elif dtype is float:
            return DataTypes._infer_float(v)
        elif dtype is bool:
            return DataTypes._infer_bool(v)
        elif dtype is date:
            return DataTypes._infer_date(v)
        elif dtype is datetime:
            return DataTypes._infer_datetime(v)
        elif dtype is time:
            return DataTypes._infer_time(v)
        else:
            raise TypeError(f"The datatype {str(dtype)} is not supported.")

    @classmethod
    def _infer_bool(cls, value):
        if isinstance(value, bool):
            return value
        elif isinstance(value, int):
            raise ValueError("it's an integer.")
        elif isinstance(value, float):
            raise ValueError("it's a float.")
        elif isinstance(value, str):
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            else:
                raise ValueError()
        else:
            raise ValueError()

    @classmethod
    def _infer_int(cls, value):
        if isinstance(value, bool):
            raise ValueError("it's a boolean")
        if isinstance(value, int):
            return value
        elif isinstance(value, float):
            if int(value) == value:
                return int(value)
            raise ValueError("it's a float")
        elif isinstance(value, str):
            value = value.replace('"', '')  # "1,234" --> 1,234
            value = value.replace(" ", "")  # 1 234 --> 1234
            value = value.replace(',', '')  # 1,234 --> 1234
            value_set = set(value)
            if value_set - DataTypes.integers:  # set comparison.
                raise ValueError
            try:
                return int(value)
            except Exception:
                raise ValueError(f"{value} is not an integer")
        else:
            raise ValueError()

    @classmethod
    def _infer_float(cls, value):
        if isinstance(value, int):
            raise ValueError("it's an integer")
        if isinstance(value, float):
            return value
        elif isinstance(value, str):
            value = value.replace('"', '')
            dot_index, comma_index = value.find('.'), value.find(',')
            if dot_index == comma_index == -1:
                pass  # there are no dots or commas.
            elif 0 < dot_index < comma_index:  # 1.234,567
                value = value.replace('.', '')  # --> 1234,567
                value = value.replace(',', '.')  # --> 1234.567
            elif dot_index > comma_index > 0:  # 1,234.678
                value = value.replace(',', '')

            elif comma_index and dot_index == -1:
                value = value.replace(',', '.')
            else:
                pass

            value_set = set(value)

            if not value_set.issubset(DataTypes.decimals):
                raise TypeError()

            # if it's a string, do also
            # check that reverse conversion is valid,
            # otherwise we have loss of precision. F.ex.:
            # int(0.532) --> 0
            try:
                float_value = float(value)
            except Exception:
                raise ValueError(f"{value} is not a float.")
            if value_set.intersection('Ee'):  # it's scientific notation.
                v = value.lower()
                if v.count('e') != 1:
                    raise ValueError("only 1 e in scientific notation")

                e = v.find('e')
                v_float_part = float(v[:e])
                v_exponent = int(v[e + 1:])
                return float(f"{v_float_part}e{v_exponent}")

            elif "." in str(float_value) and not "." in value_set:
                # when traversing through Datatype.types,
                # integer is presumed to have failed for the column,
                # so we ignore this and turn it into a float...
                reconstructed_input = str(int(float_value))

            elif "." in value:
                precision = len(value) - value.index(".") - 1
                formatter = '{0:.' + str(precision) + 'f}'
                reconstructed_input = formatter.format(float_value)

            else:
                reconstructed_input = str(float_value)

            if value.lower() != reconstructed_input:
                raise ValueError()

            return float_value
        else:
            raise ValueError()

    @classmethod
    def _infer_date(cls, value):
        if isinstance(value, date):
            return value
        elif isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                pattern = "".join(["N" if n in DataTypes.digits else n for n in value])
                f = DataTypes.date_formats.get(pattern, None)
                if f:
                    return f(value)
                else:
                    raise ValueError()
        else:
            raise ValueError()

    @classmethod
    def _infer_datetime(cls, value):
        if isinstance(value, datetime):
            return value
        elif isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                if '.' in value:
                    dot = value.find('.', 11)  # 11 = len("1999.12.12")
                elif ',' in value:
                    dot = value.find(',', 11)
                else:
                    dot = len(value)

                pattern = "".join(["N" if n in DataTypes.digits else n for n in value[:dot]])
                f = DataTypes.datetime_formats.get(pattern, None)
                if f:
                    return f(value)
                else:
                    raise ValueError()
        else:
            raise ValueError()

    @classmethod
    def _infer_time(cls, value):
        if isinstance(value, time):
            return value
        elif isinstance(value, str) and ":" in value:
            # beware time.fromisoformat reads "20" as "20:00:00", despite that it is more likely to be an integer.
            return time.fromisoformat(value)
        else:
            raise ValueError()

    @classmethod
    def _infer_str(cls, value):
        if isinstance(value, str):
            return value
        else:
            return str(value)

    @classmethod
    def _infer_none(cls, value):
        if value is None:
            return None
        if isinstance(value,str) and value == str(None):
            return None
        raise ValueError()


def _get_numpy_types():
    # d = {}
    # for name in dir(np):
    #     obj = getattr(np,name)
    #     if hasattr(obj, 'dtype'):
    #         try:
    #             if 'time' in name:
    #                 npn = obj(0, 'D')
    #             else:
    #                 npn = obj(0)
    #             nat = npn.item()
    #             d[name] = type(nat)
    #             d[npn.dtype.char] = type(nat)
    #         except:
    #             pass
    # return d
    return {
        "bool": bool,
        "bool_":bool,  #  ('?') -> <class 'bool'>
        "byte": int,  # ('b') -> <class 'int'>
        "bytes0": bytes,  #  ('S') -> <class 'bytes'>
        "bytes_": bytes,  #  ('S') -> <class 'bytes'>
        "cdouble": complex,   #('D') -> <class 'complex'>
        "cfloat": complex,  # ('D') -> <class 'complex'>
        "clongdouble": float,  # ('G') -> <class 'numpy.clongdouble'>
        "clongfloat": float,  # ('G') -> <class 'numpy.clongdouble'>
        "complex128": complex,  # ('D') -> <class 'complex'>
        "complex64": complex,  # ('F') -> <class 'complex'>
        "complex_": complex,  # ('D') -> <class 'complex'>
        "csingle": complex,  # ('F') -> <class 'complex'>
        "datetime64": date,  # ('M') -> <class 'datetime.date'>
        "double": float,  # ('d') -> <class 'float'>
        "float16": float,  # ('e') -> <class 'float'>
        "float32": float,  # ('f') -> <class 'float'>
        "float64": float,  # ('d') -> <class 'float'>
        "float_": float,  # ('d') -> <class 'float'>
        "half": float,  # ('e') -> <class 'float'>
        "int0": int,  # ('q') -> <class 'int'>
        "int16": int,  # ('h') -> <class 'int'>
        "int32": int,  # ('l') -> <class 'int'>
        "int64": int,  # ('q') -> <class 'int'>
        "int8": int,  # ('b') -> <class 'int'>
        "int_": int,  # ('l') -> <class 'int'>
        "intc": int,  # ('i') -> <class 'int'>
        "intp": int,  # ('q') -> <class 'int'>
        "longcomplex":float,  # ('G') -> <class 'numpy.clongdouble'>
        "longdouble": float,  # ('g') -> <class 'numpy.longdouble'>
        "longfloat":  float,  # ('g') -> <class 'numpy.longdouble'>
        "longlong": int,  # ('q') -> <class 'int'>
        "matrix": int,  # ('l') -> <class 'int'>
        "record": bytes,  # ('V') -> <class 'bytes'>
        "short": int,  # ('h') -> <class 'int'>
    }

numpy_types = _get_numpy_types()


class Rank(object):
    def __init__(self, *items):
        self.items = {i:ix for i,ix in zip(items,range(len(items)))}
        self.ranks = [0 for _ in items]
        self.items_list = [i for i in items]
    
    def match(self, k):  # k+=1
        ix = self.items[k]
        r = self.ranks
        r[ix]+=1

        if ix > 0:
            p = self.items_list
            while r[ix] > r[ix-1] and ix >0:  # use a simple bubble sort to maintain rank
                r[ix], r[ix-1] = r[ix-1], r[ix]
                p[ix], p[ix-1] = p[ix-1], p[ix]
                old = p[ix]
                self.items[old] = ix
                self.items[k] = ix-1
                ix -= 1
    
    def __iter__(self):
        return iter(self.items_list)






