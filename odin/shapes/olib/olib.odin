package olib

foreign import olib "../../build/libolib.a"

Memory :: struct {
	capacity:       uint,
	allocated:      uint,
	numBlocks:      uint,
	numFreeBlocks:  uint,
	minBlockSize:   uint,
	freeHeadOffset: uint,
}

Array :: struct {
	items:           rawptr,
	capacity:        uint,
	size:            uint,
	elemSize:        uint,
	isCapacityFixed: bool,
	memory:          ^Memory,
}

@(link_prefix = "olib_")
foreign olib {
	// Memory
	InitializeMemory :: proc() -> ^Memory ---
	InitializeArena :: proc(arenaSize: uint, minBlockSize: uint) -> ^Memory ---
	InitializeArenaWithBuffer :: proc(buffer: rawptr, bufferSize: uint, minBlockSize: uint) -> ^Memory ---
	GetScratchArena :: proc(memory: ^Memory, scratchBufferSize: uint) -> ^Memory ---
	ResetArena :: proc(memory: ^Memory) ---
	Allocate :: proc(memory: ^Memory, size: uint) -> rawptr ---
	Reallocate :: proc(memory: ^Memory, ptr: rawptr, size: uint) -> rawptr ---
	FreeMemory :: proc(memory: ^Memory) ---
	PopScratch :: proc(ptr: rawptr) ---

	// General arrays
	MakeDynamicArray :: proc(memory: ^Memory, elemSize: uint) -> ^Array ---
	MakeArray :: proc(memory: ^Memory, elemSize: uint, capacity: uint) -> ^Array ---
	ArraySetAt :: proc(array: ^Array, idx: uint, ptr: rawptr) ---
	ArrayAppend :: proc(array: ^Array, ptr: rawptr) ---
	ArrayReset :: proc(array: ^Array) ---
	ArrayAppendStructPtr :: proc(array: ^Array, ptr: rawptr) ---
	ArrayStructPtrIdx :: proc(array: ^Array, idx: uint) -> rawptr ---
	ArraySlice :: proc(src: ^Array, start: uint, count: uint) -> ^Array ---

	// String arrays
	MakeString :: proc(memory: ^Memory, stringData: cstring) -> ^Array ---
	MakeStringN :: proc(memory: ^Memory, stringData: cstring, len: uint) -> ^Array ---
	ArrayAppendString :: proc(array: ^Array, str: ^Array) ---
	ArrayStringIdx :: proc(array: ^Array, idx: uint) -> ^Array ---
	StringAppendCString :: proc(str: ^Array, cstr: cstring) ---
	StringAppendFormat :: proc(str: ^Array, fmt: cstring, #c_vararg args: ..any) ---

	// f32 arrays
	MakeDynamicF32Array :: proc(memory: ^Memory) -> ^Array ---
	MakeF32Array :: proc(memory: ^Memory, capacity: uint) -> ^Array ---
	ArrayAppendF32 :: proc(array: ^Array, num: f32) ---
	ArrayAppendF32Buffer :: proc(array: ^Array, num: [^]f32, numItems: uint) ---
	ArrayF32Idx :: proc(array: ^Array, idx: uint) -> f32 ---
}

