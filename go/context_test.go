package shapes

import (
	stdctx "context"
	"testing"
	"time"
)

func TestNew(t *testing.T) {
	ctx := New(nil)
	if ctx == nil {
		t.Fatal("New() returned nil")
	}
	if ctx.Ptr() == nil {
		t.Fatal("Context has nil C pointer")
	}
	defer ctx.Close()
}

func TestNewWithParent(t *testing.T) {
	parent := stdctx.Background()
	ctx := New(parent)
	defer ctx.Close()

	// Should satisfy context.Context interface
	if ctx.Done() != parent.Done() {
		t.Error("Done() should match parent")
	}
}

func TestWithGrad(t *testing.T) {
	// Default: grad disabled
	ctx := New(nil)
	defer ctx.Close()
	if ctx.GradEnabled() {
		t.Error("Grad should be disabled by default")
	}

	// With grad enabled
	ctx2 := New(nil, WithGrad(true))
	defer ctx2.Close()
	if !ctx2.GradEnabled() {
		t.Error("Grad should be enabled")
	}

	// With grad explicitly disabled
	ctx3 := New(nil, WithGrad(false))
	defer ctx3.Close()
	if ctx3.GradEnabled() {
		t.Error("Grad should be disabled")
	}
}

func TestClose(t *testing.T) {
	ctx := New(nil)

	// Should be safe to call multiple times
	ctx.Close()
	ctx.Close()
	ctx.Close()
}

func TestContextInterface(t *testing.T) {
	// Test that Context satisfies context.Context interface
	parent, cancel := stdctx.WithTimeout(stdctx.Background(), 100*time.Millisecond)
	defer cancel()

	ctx := New(parent)
	defer ctx.Close()

	// Should have parent's deadline
	deadline, ok := ctx.Deadline()
	if !ok {
		t.Fatal("Should have deadline from parent")
	}
	if deadline.IsZero() {
		t.Error("Deadline should not be zero")
	}

	// Wait for cancellation
	select {
	case <-ctx.Done():
		// Expected
	case <-time.After(200 * time.Millisecond):
		t.Error("Context should have been cancelled")
	}

	// Should have cancellation error
	if ctx.Err() == nil {
		t.Error("Should have cancellation error")
	}
}

func TestMultipleOptions(t *testing.T) {
	ctx := New(nil,
		WithGrad(true),
	)
	defer ctx.Close()

	if !ctx.GradEnabled() {
		t.Error("Grad should be enabled")
	}
}
