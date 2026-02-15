package shapes

import (
	stdctx "context"
	"unsafe"
)

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "common.h"
#include "memory.h"
#include <stdlib.h>

static inline Context *newContext(bool grad) {
	Memory *mem = initializeMemory();
	Context *ctx = allocate(mem, sizeof(Context));
	ctx->memory = mem;
	ctx->grad = grad;
	ctx->screenConfig = NULL;
	return ctx;
}
*/
import "C"

// Context wraps both Go's context.Context and the C Context struct.
// It satisfies the context.Context interface while providing access to
// the shapes C library context.
type Context struct {
	stdctx.Context  // Embedded Go context for cancellation/deadlines
	cCtx            *C.Context
	tensors         []*unsafe.Pointer // tracked C tensor pointers, nilled on Close TODO: be sure about the copying behaviour here
	intermediates   []unsafe.Pointer  // C tensor pointers to free after backward pass
	GradEnabled     bool              // Go-level grad tracking (C context always has grad=false)
	BackwardEnabled bool
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
		cCtx:    C.newContext(C.bool(false)),
	}

	for _, opt := range opts {
		opt(ctx)
	}

	return ctx
}

// Close releases the C context resources.
// All tensors created under this context become invalid after Close.
func (c *Context) Close() {
	if c.cCtx != nil && c.cCtx.memory != nil {
		for _, p := range c.tensors {
			*p = nil
		}
		c.tensors = nil
		C.freeMemory(c.cCtx.memory)
		c.cCtx = nil
	}
}

// Ptr returns the C Context pointer for passing to C functions.
// The returned pointer is only valid while the Context is alive.
func (c *Context) Ptr() *C.Context {
	return c.cCtx
}

// UnsafePtr returns the C Context as an unsafe.Pointer for cross-package CGo casts.
func (c *Context) UnsafePtr() unsafe.Pointer {
	return unsafe.Pointer(c.cCtx)
}

// NoGrad returns a new Context that shares the same C context and memory
// but has gradient tracking disabled. In this context gradients are not created for new tensors
func (c *Context) NoGrad() *Context {
	return &Context{
		Context:         c.Context,
		cCtx:            c.cCtx,
		tensors:         c.tensors,
		intermediates:   c.intermediates,
		GradEnabled:     false,
		BackwardEnabled: c.BackwardEnabled,
	}
}

// Fused returns a new Context that shares the same C context and memory
// but turns of backward passes for any ops used in that context.
// Meant for doing compund operations where the caller might want to specify the backward pass manually
func (c *Context) Fused() *Context {
	return &Context{
		Context:         c.Context,
		cCtx:            c.cCtx,
		tensors:         c.tensors,
		intermediates:   c.intermediates,
		GradEnabled:     c.GradEnabled,
		BackwardEnabled: false,
	}
}

// NoGraph() returns a new Context that shares the same C context and memory
// turns off all gradient tacking. For use in backward pass only

func (c *Context) NoGraph() *Context {
	return &Context{
		Context:         c.Context,
		cCtx:            c.cCtx,
		tensors:         c.tensors,
		intermediates:   c.intermediates,
		GradEnabled:     false,
		BackwardEnabled: false,
	}
}

func (c *Context) Memory() *C.Memory {
	return c.cCtx.memory
}

// UnsafeMemory returns the C Memory as an unsafe.Pointer for cross-package CGo casts.
func (c *Context) UnsafeMemory() unsafe.Pointer {
	return unsafe.Pointer(c.cCtx.memory)
}

// Track registers a C pointer so it gets nilled on Close().
// The tensor package calls this when creating tensors.
func (c *Context) Track(p *unsafe.Pointer) {
	c.tensors = append(c.tensors, p)
}

// MarkIntermediate registers a C tensor pointer for freeing after backward.
func (c *Context) MarkIntermediate(p unsafe.Pointer) {
	c.intermediates = append(c.intermediates, p)
}

// Intermediates returns the list of intermediate C tensor pointers.
func (c *Context) Intermediates() []unsafe.Pointer {
	return c.intermediates
}

// ClearIntermediates resets the intermediate list after they have been freed.
func (c *Context) ClearIntermediates() {
	c.intermediates = c.intermediates[:0]
}

// Option is a function that configures a Context.
type Option func(*Context)

// WithGrad enables or disables gradient tracking and backward passes.
func WithGrad(enabled bool) Option {
	return func(c *Context) {
		c.GradEnabled = enabled
		c.BackwardEnabled = enabled
	}
}
