import std/[lists, options, sugar, os, parseutils]
from ../../utils import implement

var MAX_FILES_OPEN = 4096

discard parseInt(os.getEnv("MAX_FILES_OPEN", $MAX_FILES_OPEN), MAX_FILES_OPEN, 0)

type FileTracker* = ref object of RootObj
    # base type for file open file management
    count: int

type FileTrackerSimple = ref object of FileTracker
    # we're on the fast path because we can handle all these files
    pageHandles: seq[File]

type FileTrackerDeferred = ref object of FileTracker
    # we're on the slow path because we can't handle all these files
    pageHandles: seq[Option[File]]
    pages: seq[string]
    pagesOpen: int
    openOrder: SinglyLinkedList[int]

proc newFileTracker*(pages: seq[string]): FileTracker =
    if pages.len <= MAX_FILES_OPEN:
        let handles = collect:
            for p in pages:
                open(p, fmWrite)

        return FileTrackerSimple(count: pages.len, pageHandles: handles)

    stderr.writeLine("Too many columns in file, using deferred importer instead. This will greatly reduce performance: " & $pages.len & " > " & $MAX_FILES_OPEN & ". You can increase MAX_FILES_OPEN if your OS allows.")

    for p in pages:
        # create empties
        open(p, fmWrite).close()

    return FileTrackerDeferred(count: pages.len, pages: pages, pageHandles: newSeq[Option[File]](pages.len), openOrder: initSinglyLinkedList[int]())

method getHandle(self: FileTrackerDeferred, index: int): File {.base.} =
    if self.pageHandles[index].isSome:
        return self.pageHandles[index].get

    let handle = open(self.pages[index], fmAppend)

    self.pageHandles[index] = some(handle)

    inc self.pagesOpen
    self.openOrder.add(newSinglyLinkedNode(index))

    return handle

method closeHandle(self: FileTrackerDeferred, index: int): void {.base.} =
    if self.pageHandles[index].isNone:
        return

    self.pageHandles[index].get.flushFile()
    self.pageHandles[index].get.close()
    self.pageHandles[index] = none[File]()
    dec self.pagesOpen

method `[]`*(self: FileTracker, index: int): File {.base, inline.} = implement("FileTracker.`[]` must be implemented by inheriting class.")
method `[]`(self: FileTrackerSimple, index: int): File = self.pageHandles[index]
method `[]`(self: FileTrackerDeferred, index: int): File =
    if self.pagesOpen < MAX_FILES_OPEN:
        return self.getHandle(index)

    self.closeHandle(self.openOrder.head.value)
    self.openOrder.remove(self.openOrder.head)

    return self.getHandle(index)

method close*(self: FileTracker): void {.base, inline.} = implement("FileTracker.close must be implemented by inheriting class.")
method close(self: FileTrackerSimple): void = (for h in self.pageHandles: h.close())
method close(self: FileTrackerDeferred): void = (for h in self.pageHandles: (if h.isSome: h.get.close()))

proc len*(self: FileTracker): int = self.count
