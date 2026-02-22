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

static inline Context *newContext(bool grad, size_t arenaSize) {
	Memory *mem = initializeArena(arenaSize);
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
	root            *Context          // root context (nil if this is root)
	handles         []*unsafe.Pointer // root only: all tracked handles, nilled on Close
	locals          []*unsafe.Pointer // this ctx: tensors created through this ctx
	intermediates   []unsafe.Pointer  // root only: C tensor pointers queued for FreeIntermediates
	arenaSize       int
	ownsMemory      bool // true for root context
	GradEnabled     bool // Go-level grad tracking (C context always has grad=false)
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
		Context:       parent,
		handles:       make([]*unsafe.Pointer, 0),
		locals:        make([]*unsafe.Pointer, 0),
		intermediates: make([]unsafe.Pointer, 0),
		ownsMemory:    true,
	}
	ctx.root = ctx

	for _, opt := range opts {
		opt(ctx)
	}

	if ctx.arenaSize != 0 {
		ctx.cCtx = C.newContext(C.bool(false), C.size_t(ctx.arenaSize))
	} else {
		ctx.cCtx = C.newContext(C.bool(false), 1024*1024*64)
	}

	return ctx
}

// Close releases the C context resources.
// All tensors created under this context become invalid after Close.
// Only the root context frees memory; derived contexts are no-ops.
func (c *Context) Close() {
	if !c.ownsMemory {
		return
	}
	if c.cCtx != nil && c.cCtx.memory != nil {
		for _, p := range c.handles {
			if p != nil {
				*p = nil
			}
		}
		c.handles = nil
		c.locals = nil
		C.freeMemory(c.cCtx.memory)
		c.cCtx = nil
	}
}

func (c *Context) PrintMemoryFragmentationChart() {
	C.printMemoryFragmentationChart((*C.Memory)(c.UnsafeMemory()))

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

// derive creates a new derived context sharing the same C context and root.
// Fresh locals slice; handles and intermediates are root-owned.
func (c *Context) derive(gradEnabled, backwardEnabled bool) *Context {
	return &Context{
		Context:         c.Context,
		cCtx:            c.cCtx,
		root:            c.root,
		locals:          make([]*unsafe.Pointer, 0),
		ownsMemory:      false,
		GradEnabled:     gradEnabled,
		BackwardEnabled: backwardEnabled,
	}
}

// NoGrad returns a new Context that shares the same C context and memory
// but has gradient tracking disabled. In this context gradients are not created for new tensors
func (c *Context) NoGrad() *Context {
	return c.derive(false, c.BackwardEnabled)
}

// Fused returns a new Context that shares the same C context and memory
// but turns of backward passes for any ops used in that context.
// Meant for doing compund operations where the caller might want to specify the backward pass manually
func (c *Context) Fused() *Context {
	return c.derive(true, false)
}

// NoGraph() returns a new Context that shares the same C context and memory
// turns off all gradient tacking. For use in backward pass only
func (c *Context) NoGraph() *Context {
	return c.derive(false, false)
}

// OpSubcontext returns a fresh subcontext for a single backward op.
// Shares arena and root; backward/graph disabled; fresh locals for sweeping.
func (c *Context) OpSubcontext() *Context {
	return c.derive(false, false)
}

func (c *Context) Memory() *C.Memory {
	return c.cCtx.memory
}

// UnsafeMemory returns the C Memory as an unsafe.Pointer for cross-package CGo casts.
func (c *Context) UnsafeMemory() unsafe.Pointer {
	return unsafe.Pointer(c.cCtx.memory)
}

// Track registers a C pointer so it gets nilled on Close().
// Appends to root.handles (for cleanup) and c.locals (for intermediate marking).
// The tensor package calls this when creating tensors.
func (c *Context) Track(p *unsafe.Pointer) {
	c.root.handles = append(c.root.handles, p)
	c.locals = append(c.locals, p)
}

// MarkIntermediate registers a C tensor pointer for freeing after backward.
// Always appends to root's intermediates list.
func (c *Context) MarkIntermediate(p unsafe.Pointer) {
	c.root.intermediates = append(c.root.intermediates, p)
}

// Intermediates returns the list of intermediate C tensor pointers from root.
func (c *Context) Intermediates() []unsafe.Pointer {
	return c.root.intermediates
}

// ClearIntermediates resets the intermediate list after they have been freed.
func (c *Context) ClearIntermediates() {
	c.root.intermediates = c.root.intermediates[:0]
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

func WithArenaSize(size int) Option {
	return func(c *Context) {
		c.arenaSize = size
	}
}
