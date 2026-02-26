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
#include "tensor/tensor.h"
#include <stdlib.h>

static inline Context *newContext(bool grad, size_t arenaSize) {
	Memory *mem = initializeArena(arenaSize, 1);
	Context *ctx = allocate(mem, sizeof(Context));
	ctx->memory = mem;
	ctx->grad = grad;
	ctx->screenConfig = NULL;
	return ctx;
}

*/
import "C"

type Context interface {
	Mark(t *Tensor)
	Track(t *Tensor)

	NoGrad(options ...subContextOption) SubContext
	Fused(options ...subContextOption) SubContext
	NoGraph(options ...subContextOption) SubContext

	GradEnabled() bool
	BackwardEnabled() bool
}

type MainContext interface {
	Context

	UnsafePtr() unsafe.Pointer
	Finish()
}

type SubContext interface {
	Context
	Finish(...subContextOption)
}

type shapesCtx struct {
	stdctx.Context // Embedded Go context for cancellation/deadlines

	gradEnabled     bool
	backwardEnabled bool
}

// New creates a new Context with the given parent Go context.
// If parent is nil, context.Background() is used.
//
// Options:
//   - WithGrad(bool) - Enable gradient tracking (default: false)
//
// The returned Context must be freed with Close() when done.
func New(parent stdctx.Context, opts ...mainContextOption) MainContext {
	if parent == nil {
		parent = stdctx.Background()
	}

	ctx := &mainContext{
		handles:  make([]*Tensor, 0),
		freeList: make([]*Tensor, 0),
	}

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

// mainContext wraps both Go's context.mainContext and the C mainContext struct.
// It satisfies the context.Context interface while providing access to
// the shapes C library context.
type mainContext struct {
	shapesCtx
	cCtx      *C.Context
	handles   []*Tensor // root only: all tracked handles, nilled on Close
	freeList  []*Tensor // root only: C tensor pointers queued for FreeIntermediates
	arenaSize int
}

// Finish releases the C context resources.
// All tensors created under this context become invalid after Finish.
// Only the root context frees memory; derived contexts stage their locals
// into the freeList for later sweeping.
func (c *mainContext) Finish() {
	if c.cCtx != nil && c.cCtx.memory != nil {
		for _, p := range c.handles {
			if p != nil && p.cTensor != nil {
				c.Free(p)
			}
		}
		c.handles = nil
		C.freeMemory(c.cCtx.memory)
		c.cCtx = nil
	}
}

func (c *mainContext) Free(t *Tensor) {
	if bool(t.cTensor.isView) {
		c.freeView(t)
	} else {
		c.freeTensor(t)
	}

	t.cTensor = nil
}

func (c *mainContext) freeTensor(t *Tensor) {
	result := C.FreeTensor((*C.Context)(c.UnsafePtr()), t.cTensor)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}

// FreeView releases a view tensor's metadata (dims, multipliers, boundary)
// back to the arena without freeing the shared values.
func (c *mainContext) freeView(t *Tensor) {
	result := C.FreeViewTensor((*C.Context)(c.UnsafePtr()), t.cTensor)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}

// UnsafePtr returns the C Context as an unsafe.Pointer for cross-package CGo casts.
func (c *mainContext) UnsafePtr() unsafe.Pointer {
	return unsafe.Pointer(c.cCtx)
}

// NoGrad returns a new Context that shares the same C context and memory
// but has gradient tracking disabled. In this context gradients are not created for new tensors
func (c *mainContext) NoGrad(options ...subContextOption) SubContext {
	return noGrad(c, options...)
}

// Fused returns a new Context that shares the same C context and memory
// but turns of backward passes for any ops used in that context.
// Meant for doing compund operations where the caller might want to specify the backward pass manually
func (c *mainContext) Fused(options ...subContextOption) SubContext {
	return fused(c, options...)
}

// NoGraph() returns a new Context that shares the same C context and memory
// turns off all gradient tacking. For use in backward pass only
func (c *mainContext) NoGraph(options ...subContextOption) SubContext {
	return noGraph(c, options...)
}

func (c *mainContext) Memory() *C.Memory {
	return c.cCtx.memory
}

// UnsafeMemory returns the C Memory as an unsafe.Pointer for cross-package CGo casts.
func (c *mainContext) UnsafeMemory() unsafe.Pointer {
	return unsafe.Pointer(c.cCtx.memory)
}

// Track registers a C pointer so it gets nilled on Close().
// Appends to root.handles (for cleanup) and c.locals (for intermediate marking).
// The tensor package calls this when creating tensors.
func (c *mainContext) Track(t *Tensor) {
	c.handles = append(c.handles, t)
}

func (c *mainContext) PrintMemoryFragmentationChart() {
	C.printMemoryFragmentationChart((*C.Memory)(c.UnsafeMemory()))
}

func (c *mainContext) NumTrackTensors() int {
	return len(c.handles)
}

func (c *mainContext) NumFreeBlocks() int {
	return int(c.cCtx.memory.numFreeBlocks)
}

func (c *mainContext) NumAllocatedBlocks() int {
	return int(c.cCtx.memory.numBlocks) - c.NumFreeBlocks()
}

// MarkIntermediate registers a C tensor pointer for freeing after backward.
// Always appends to root's intermediates list.
func (c *mainContext) Mark(t *Tensor) {
	c.freeList = append(c.freeList, t)
}

func (ctx *mainContext) Sweep() {
	cCtx := (*C.Context)(ctx.UnsafePtr())

	if len(ctx.freeList) == 0 {
		return
	}

	// Deduplicate to prevent double-free; nil cTensor after freeing.
	seen := make(map[unsafe.Pointer]bool, len(ctx.freeList))
	for _, t := range ctx.freeList {
		if t.cTensor == nil {
			continue
		}
		p := unsafe.Pointer(t.cTensor)
		if seen[p] {
			t.cTensor = nil
			continue
		}
		seen[p] = true
		ct := (*C.Tensor)(p)
		if ct.isView {
			C.FreeViewTensor(cCtx, ct)
		} else {
			C.FreeTensor(cCtx, ct)
		}
		t.cTensor = nil
	}

	ctx.freeList = ctx.freeList[:0]

	// Compact root.handles: remove entries whose cTensor has been nilled.
	live := ctx.handles[:0]
	for _, t := range ctx.handles {
		if t.cTensor != nil {
			live = append(live, t)
		}
	}
	ctx.handles = live
}

func (c *mainContext) BackwardEnabled() bool {
	return c.backwardEnabled
}

func (c *mainContext) GradEnabled() bool {
	return c.gradEnabled
}

// ---------------------------- Subcontext ----------------------------

type subContext struct {
	shapesCtx
	parent Context
	locals []*Tensor // this ctx: tensors created through this ctx

	op OpType

	inputs      []*Tensor
	hiddenState []*Tensor
	result      *Tensor
	metadata    any
	backward    BackwardFn
}

func (sc *subContext) Mark(t *Tensor) {
	sc.parent.Mark(t)
}

func (sc *subContext) Finish(options ...subContextOption) {
	for _, t := range sc.locals {
		if t == sc.result {
			continue
		}

		sc.Mark(t)
	}

	sc.locals = nil
}

// NoGrad returns a new Context that shares the same C context and memory
// but has gradient tracking disabled. In this context gradients are not created for new tensors
func (c *subContext) NoGrad(options ...subContextOption) SubContext {
	return noGrad(c, options...)
}

// Fused returns a new Context that shares the same C context and memory
// but turns of backward passes for any ops used in that context.
// Meant for doing compund operations where the caller might want to specify the backward pass manually
func (c *subContext) Fused(options ...subContextOption) SubContext {
	return fused(c, options...)
}

// NoGraph() returns a new Context that shares the same C context and memory
// turns off all gradient tacking. For use in backward pass only
func (c *subContext) NoGraph(options ...subContextOption) SubContext {
	return noGraph(c, options...)
}

func (sc *subContext) Track(t *Tensor) {
	sc.parent.Track(t)
	sc.locals = append(sc.locals, t)
}

// NewComputationGraphNode attaches a computation graph node to result.
// Used by external packages (e.g., layer, activation) to register custom backward passes.
func (c *subContext) prepareResultForBackwardPass() {
	c.result.Computation.inputs = c.inputs
	c.result.Computation.backward = c.backward
	c.result.Computation.op = c.op

	if c.result.Computation.parameters == nil {
		c.result.Computation.parameters = []*Tensor{}
		c.result.Computation.parameters = append(c.result.Computation.parameters, c.hiddenState...)
	}
}

func (c *subContext) BackwardEnabled() bool {
	return c.backwardEnabled
}

func (c *subContext) GradEnabled() bool {
	return c.gradEnabled
}

// ---------------------------- funcs ----------------------------

// derive creates a new derived context sharing the same C context and root.
// Fresh locals slice; handles and intermediates are root-owned.
func derive(ctx Context, gradEnabled, backwardEnabled bool, options ...subContextOption) SubContext {
	sc := &subContext{
		parent: ctx,
		locals: make([]*Tensor, 0),
		shapesCtx: shapesCtx{
			backwardEnabled: gradEnabled,
			gradEnabled:     gradEnabled,
		},
	}

	for _, opt := range options {
		opt(sc)
	}

	return sc
}

func noGrad(c Context, options ...subContextOption) SubContext {
	if !c.GradEnabled() {
		panic("Creating NoGrad context from a context where gradients are not tracked")
	}

	return derive(c, false, c.BackwardEnabled(), options...)
}

// Fused returns a new Context that shares the same C context and memory
// but turns of backward passes for any ops used in that context.
// Meant for doing compund operations where the caller might want to specify the backward pass manually
func fused(c Context, options ...subContextOption) SubContext {
	if !c.GradEnabled() {
		panic("Creating fused context from a context where gradients are not tracked")
	}

	return derive(c, true, false, options...)
}

// NoGraph() returns a new Context that shares the same C context and memory
// turns off all gradient tacking. For use in backward pass only
func noGraph(c Context, options ...subContextOption) SubContext {
	if !c.GradEnabled() {
		panic("Creating NoGrad context from a context where gradients are not tracked")
	}
	return derive(c, false, false, options...)
}
