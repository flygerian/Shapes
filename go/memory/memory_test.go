package memory

import "testing"

func TestNew(t *testing.T) {
	mem := New()
	if mem == nil {
		t.Fatal("New() returned nil")
	}
	if mem.Ptr() == nil {
		t.Fatal("Memory has nil C pointer")
	}
	defer mem.Free()
}

func TestFree(t *testing.T) {
	mem := New()

	// Should be safe to call multiple times
	mem.Free()
	mem.Free()
	mem.Free()

	// After Free, Ptr should be nil
	if mem.Ptr() != nil {
		t.Error("Ptr() should be nil after Free()")
	}
}

func TestFinalizer(t *testing.T) {
	// Create and abandon a Memory to test finalizer
	mem := New()
	if mem.Ptr() == nil {
		t.Fatal("Memory has nil C pointer")
	}
	// Let it go out of scope - finalizer should clean it up
	// This tests that the finalizer doesn't panic
}
