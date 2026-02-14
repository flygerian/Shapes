package shapes

import (
	stdctx "context"
)

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "common.h"
#include "memory.h"
#include <stdlib.h>
*/
import "C"

// Context wraps both Go's context.Context and the C Context struct.
// It satisfies the context.Context interface while providing access to
// the shapes C library context.
type Context struct {
	stdctx.Context // Embedded Go context for cancellation/deadlines
	cCtx           C.Context
}

// New creates a new Context with the given parent Go context.
// If parent is nil, context.Background() is used.
//
// Options:
//   - WithGrad(bool) - Enable gradient tracking (default: false)
//
// The returned Context must be freed with Close() when done.
func New(parent stdctx.Context, opts ...Option) *Context {
	if parent == nil {
		parent = stdctx.Background()
	}

	ctx := &Context{
		Context: parent,
		cCtx: C.Context{
			memory:       C.initializeMemory(),
			grad:         C.bool(false),
			screenConfig: nil, // Visual config allocated lazily if needed
		},
	}

	// Apply options
	for _, opt := range opts {
		opt(ctx)
	}

	return ctx
}

// Close releases the C context resources.
// After calling Close, the Context should not be used.
func (c *Context) Close() {
	if c.cCtx.memory != nil {
		C.freeMemory(c.cCtx.memory)
		c.cCtx.memory = nil
	}
}

// Ptr returns the C Context pointer for passing to C functions.
// The returned pointer is only valid while the Context is alive.
func (c *Context) Ptr() *C.Context {
	return &c.cCtx
}

// GradEnabled returns whether gradient tracking is enabled.
func (c *Context) GradEnabled() bool {
	return bool(c.cCtx.grad)
}

func (c *Context) Memory() *C.Memory {
	return c.cCtx.memory
}

// Option is a function that configures a Context.
type Option func(*Context)

// WithGrad enables or disables gradient tracking.
func WithGrad(enabled bool) Option {
	return func(c *Context) {
		c.cCtx.grad = C.bool(enabled)
	}
}
