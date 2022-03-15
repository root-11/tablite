import pathlib
import chardet
from test_multi_proc import detect_seperator, text_escape, Table
from ..tablite.datatypes import DataTypes

def interactive_file_import():
    """
    interactive file import for commandline use.
    """
    def get_file():
        while 1:
            path = input("| file path > ")
            path = pathlib.Path(path)
            if not path.exists():
                print(f"| file not found > {path}")
                continue
            print(f"| importing > {p.absolute()}")
            r = input("| is this correct? [Y/N] >")
            if "Y" in r.upper():
                return p
    
    def csv_menu(path):
        config = {
            'import_as': 'csv',
            'newline': None,
            'text_qualifier':None,
            'delimiter':None,
            'first_row_headers':None,
            'columns': {}
        }
        with path.open('rb', encoding=encoding) as fi:
            # read first 5 rows
            s, new_line_counter = [], 5
            while new_line_counter > 0:
                c = fi.read(1)
                s.append(c)
                if c == '\n':
                    new_line_counter -= 1
        s = s.join()
        encoding = chardet.detect(s)
        print(f"| file encoding > {encoding}")
        config.update(encoding)
        if b'\r\n' in s:
            newline = b'\r\n'
        else:
            newline = b'\n'
        print(f"| newline > {newline} ")

        config['newline'] = newline

        for line in s.split(newline):
            print(line)
        
        config['first_row_headers'] = "y" in input("| first row are column names? > Y/N").lower()

        text_escape_char = input("| text escape character > ")
        config['text escape']= None if text_escape_char=="" else text_escape_char
        
        seps = detect_seperator(s)
        
        print(f"| delimiters detected > {seps}")

        delimiter = input("| delimiter ? >")
        config['delimiter'] = delimiter

        def preview(s, text_esc, delimit):
            array = []
            for line in s.split(newline):
                fields = text_escape(line, text_esc, delimit)
                array.append(fields)
            return array

        def is_done(array):
            print("| rotating the first 5 lines > ")
            print("\n".join(map(" | ".join, zip(*(array)))))
            answer = input("| does this look right? Y/N >")
            return answer.lower(), array
        
        array = preview(s,text_escape_char,delimiter)
        if "n" in is_done(array):
            print("| update keys and values. Enter blank key when done.")
            while 1:
                key = input("| key > ")
                if key == "":
                    array = preview(s,text_escape_char,delimiter)
                    if "y" in is_done(array):
                        break
                value = input("| value > ")
                config[key]= value

        cols = input(f"| select columns ? [all/some] > ")
        config['columns'] = {}
        if "a" in cols:
            pass  # empty dict means import all .
            for ix, colname in enumerate(array[0]):
                sample = [array[i][ix] for i in range(1,len(array))]
                datatype = DataTypes.guess(*sample)
                for dtype in DataTypes.types: # strict order.
                    if dtype in datatype:
                        break
                config['columns'][colname] = dtype

        else:
            print("| Enter columns to keep. Enter blank when done.")
            while 1: 
                key = input("| column name > ")
                ix = array[0].index(key)
                if ix == -1:
                    print(f"| {key} not found.")
                    continue
                sample = [array[i][ix] for i in range(1,6)]
                datatype = DataTypes.guess(*sample)
                print(f"| > {datatype}")
                for dtype in DataTypes.types: # strict order.
                    if dtype in datatype:
                        break
                config['columns'][colname] = dtype
                while 1:
                    guess = input(f"| is {dtype} correct datatype ?\n| > Enter accepts / type name if different >")
                    if guess == "":
                        break
                    elif guess in [str(t) for t in DataTypes.types]:
                        config['columns'][colname] = dtype
                    else:
                        print(f"| {guess} > No such type.")
                
        print(f"| using config > \n{config}")
        return config       

    def xlsx_menu():
        raise NotImplementedError("coming soon!")

    def txt_menu():
        raise NotImplementedError("coming soon!")

    def get_config(path):
        assert isinstance(p, pathlib.Path)
        ext = path.name.split(".")[-1].lower()
        if ext == "csv":
            config = csv_menu(p)
        elif ext == "xlsx":
            config = xlsx_menu()
        elif ext == "txt":
            config = txt_menu(p)
        else:
            print(f"no method for .{ext}'s")
        
    try:
        p = get_file()
        config = get_config(p)
        new_p = Table.import_file(p,config)
        t = Table.load_file(new_p)
        return t
    except KeyboardInterrupt:
        return
