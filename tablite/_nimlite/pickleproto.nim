const PKL_MARK*            = '('   # push special markobject on stack
const PKL_STOP*            = '.'   # every pickle ends with STOP
const PKL_POP*             = '0'   # discard topmost stack item
const PKL_POP_MARK*        = '1'   # discard stack top through topmost markobject
const PKL_DUP*             = '2'   # duplicate top stack item
const PKL_FLOAT*           = 'F'   # push float object; decimal string argument
const PKL_INT*             = 'I'   # push integer or bool; decimal string argument
const PKL_BININT*          = 'J'   # push four-byte signed int
const PKL_BININT1*         = 'K'   # push 1-byte unsigned int
const PKL_LONG*            = 'L'   # push long; decimal string argument
const PKL_BININT2*         = 'M'   # push 2-byte unsigned int
const PKL_NONE*            = 'N'   # push None
const PKL_PERSID*          = 'P'   # push persistent object; id is taken from string arg
const PKL_BINPERSID*       = 'Q'   #  "       "         "  ;  "  "   "     "  stack
const PKL_REDUCE*          = 'R'   # apply callable to argtuple, both on stack
const PKL_STRING*          = 'S'   # push string; NL-terminated string argument
const PKL_BINSTRING*       = 'T'   # push string; counted binary string argument
const PKL_SHORT_BINSTRING* = 'U'   #  "     "   ;    "      "       "      " < 256 bytes
const PKL_UNICODE*         = 'V'   # push Unicode string; raw-unicode-escaped'd argument
const PKL_BINUNICODE*      = 'X'   #   "     "       "  ; counted UTF-8 string argument
const PKL_APPEND*          = 'a'   # append stack top to list below it
const PKL_BUILD*           = 'b'   # call __setstate__ or __dict__.update()
const PKL_GLOBAL*          = 'c'   # push self.find_class(modname, name); 2 string args
const PKL_DICT*            = 'd'   # build a dict from stack items
const PKL_EMPTY_DICT*      = '}'   # push empty dict
const PKL_APPENDS*         = 'e'   # extend list on stack by topmost stack slice
const PKL_GET*             = 'g'   # push item from memo on stack; index is string arg
const PKL_BINGET*          = 'h'   #   "    "    "    "   "   "  ;   "    " 1-byte arg
const PKL_INST*            = 'i'   # build & push class instance
const PKL_LONG_BINGET*     = 'j'   # push item from memo on stack; index is 4-byte arg
const PKL_LIST*            = 'l'   # build list from topmost stack items
const PKL_EMPTY_LIST*      = ']'   # push empty list
const PKL_OBJ*             = 'o'   # build & push class instance
const PKL_PUT*             = 'p'   # store stack top in memo; index is string arg
const PKL_BINPUT*          = 'q'   #   "     "    "   "   " ;   "    " 1-byte arg
const PKL_LONG_BINPUT*     = 'r'   #   "     "    "   "   " ;   "    " 4-byte arg
const PKL_SETITEM*         = 's'   # add key+value pair to dict
const PKL_TUPLE*           = 't'   # build tuple from topmost stack items
const PKL_EMPTY_TUPLE*     = ')'   # push empty tuple
const PKL_SETITEMS*        = 'u'   # modify dict by adding topmost key+value pairs
const PKL_BINFLOAT*        = 'G'   # push float; arg is 8-byte float encoding

# Protocol 2

const PKL_PROTO*           = '\x80'  # identify pickle protocol
const PKL_NEWOBJ*          = '\x81'  # build object by applying cls.__new__ to argtuple
const PKL_EXT1*            = '\x82'  # push object from extension registry; 1-byte index
const PKL_EXT2*            = '\x83'  # ditto, but 2-byte index
const PKL_EXT4*            = '\x84'  # ditto, but 4-byte index
const PKL_TUPLE1*          = '\x85'  # build 1-tuple from stack top
const PKL_TUPLE2*          = '\x86'  # build 2-tuple from two topmost stack items
const PKL_TUPLE3*          = '\x87'  # build 3-tuple from three topmost stack items
const PKL_NEWTRUE*         = '\x88'  # push True
const PKL_NEWFALSE*        = '\x89'  # push False
const PKL_LONG1*           = '\x8a'  # push long from < 256 bytes
const PKL_LONG4*           = '\x8b'  # push really big long

# Protocol 3 (Python 3.x)

const PKL_BINBYTES*        = 'B'   # push bytes; counted binary string argument
const PKL_SHORT_BINBYTES*  = 'C'   #  "     "   ;    "      "       "      " < 256 bytes

# Protocol 4

const PKL_SHORT_BINUNICODE*  = '\x8c'  # push short string; UTF-8 length < 256 bytes
const PKL_BINUNICODE8*       = '\x8d'  # push very long string
const PKL_BINBYTES8*         = '\x8e'  # push very long bytes string
const PKL_EMPTY_SET*         = '\x8f'  # push empty set on the stack
const PKL_ADDITEMS*          = '\x90'  # modify set by adding topmost stack items
const PKL_FROZENSET*         = '\x91'  # build frozenset from topmost stack items
const PKL_NEWOBJ_EX*         = '\x92'  # like NEWOBJ but work with keyword only arguments
const PKL_STACK_GLOBAL*      = '\x93'  # same as GLOBAL but using names on the stacks
const PKL_MEMOIZE*           = '\x94'  # store top of the stack in memo
const PKL_FRAME*             = '\x95'  # indicate the beginning of a new frame

# Protocol 5

const PKL_BYTEARRAY8*        = '\x96'  # push bytearray
const PKL_NEXT_BUFFER*       = '\x97'  # push next out-of-band buffer
const PKL_READONLY_BUFFER*   = '\x98'  # make top of stack readonly

const PKL_STRING_TERM*       = '\x0A'
const PKL_PROTO_VERSION*     = '\3'