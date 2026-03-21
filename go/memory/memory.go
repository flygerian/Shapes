package memory

/*
#include "memory.h"
*/
import "C"
import "runtime"

// Memory wraps the C Memory struct and provides a Go-friendly arena allocator.
// The arena is 1MB and is used internally by tensor operations.
// All memory is freed together when Free() is called.
type Memory struct {
	cMemory *C.Memory
}

// New creates a new memory arena allocator with 1MB capacity.
// Call Free() when done to release the arena.
func New() *Memory {
	m := &Memory{
		cMemory: C.initializeMemory(),
	}
	// Set finalizer as safety net, but explicit Free() is preferred
	runtime.SetFinalizer(m, (*Memory).Free)
	return m
}

// Free releases the memory arena. Safe to call multiple times.
func (m *Memory) Free() {
	if m.cMemory != nil {
		C.freeMemory(m.cMemory)
		m.cMemory = nil
		runtime.SetFinalizer(m, nil)
	}
}

// Ptr returns the underlying C pointer for passing to tensor operations.
func (m *Memory) Ptr() *C.Memory {
	return m.cMemory
}
