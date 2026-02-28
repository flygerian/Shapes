package shapes

import (
	stdctx "context"
	"fmt"
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
	Mark(t Tensor)
	Track(t Tensor)

	Epoch(options ...subContextOption) SubContext
	NoGrad(options ...subContextOption) SubContext
	Fused(options ...subContextOption) SubContext
	NoGraph(options ...subContextOption) SubContext

	GradEnabled() bool
	BackwardEnabled() bool

	UnsafeMemory() unsafe.Pointer
	UnsafePtr() unsafe.Pointer

	NumAllocatedBlocks() int
	NumFreeBlocks() int
	Sweep()
}

type MainContext interface {
	Context

	Finish()
}

type SubContext interface {
	Context
	Finish(...subContextOption)
}

type shapesCtx struct {
	stdctx.Context // Embedded Go context for cancellation/deadlines

	gradEnabled      bool
	backwardEnabled  bool
	persistant       bool
	sweepAfterFinish bool
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
		handles:  make([]Tensor, 0),
		freeList: make([]Tensor, 0),
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
	cCtx              *C.Context
	handles           []Tensor // root only: all tracked handles, nilled on Close
	freeList          []Tensor // root only: C tensor pointers queued for FreeIntermediates
	persistentTensors []Tensor
	arenaSize         int
}

// Finish releases the C context resources.
// All tensors created under this context become invalid after Finish.
// Only the root context frees memory; derived contexts stage their locals
// into the freeList for later sweeping.
func (c *mainContext) Finish() {
	if c.cCtx != nil && c.cCtx.memory != nil {
		for _, p := range c.handles {
			if p != nil {
				c.Free(p)
			}
		}
		c.handles = nil
		C.freeMemory(c.cCtx.memory)
		c.cCtx = nil
	}
}

func (c *mainContext) Free(t Tensor) {
	if bool(t.(*tensor).cTensor.isView) {
		c.freeView(t)
	} else {
		c.freeTensor(t)
	}

	t.(*tensor).cTensor = nil
}

func (c *mainContext) freeTensor(t Tensor) {
	result := C.FreeTensor((*C.Context)(c.UnsafePtr()), t.(*tensor).cTensor)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}

// FreeView releases a view tensor's metadata (dims, multipliers, boundary)
// back to the arena without freeing the shared values.
func (c *mainContext) freeView(t Tensor) {
	result := C.FreeViewTensor((*C.Context)(c.UnsafePtr()), t.(*tensor).cTensor)
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

// Epoch returns a short-lived context for one training/inference step.
// It preserves the parent context's grad/backward behavior while keeping
// locals isolated so callers can Finish/Sweep at epoch boundaries.
func (c *mainContext) Epoch(options ...subContextOption) SubContext {
	return epoch(c, options...)
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
func (c *mainContext) Track(t Tensor) {
	c.handles = append(c.handles, t)
	if c.persistant {
		c.persistentTensors = append(c.persistentTensors, t)
	}
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
func (c *mainContext) Mark(t Tensor) {
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
		p := unsafe.Pointer(t.(*tensor).cTensor)
		if seen[p] {
			continue
		}

		seen[p] = true
		ct := (*C.Tensor)(p)

		if ct == nil {
			continue
		}

		if ct.isView {
			C.FreeViewTensor(cCtx, ct)
		} else {
			C.FreeTensor(cCtx, ct)
		}

		t.(*tensor).cTensor = nil
	}

	ctx.freeList = ctx.freeList[:0]

	// Compact root.handles: remove entries whose cTensor has been nilled.
	live := ctx.handles[:0]
	for _, t := range ctx.handles {
		if t.(*tensor).cTensor != nil {
			live = append(live, t)
		}
	}
	fmt.Printf("[Sweep] handles after compact: %d\n", len(live))
	for i, t := range live {
		tt := t.(*tensor)
		hasGrad := tt.computation != nil && tt.computation.grad != nil
		hasBackward := tt.computation != nil && tt.computation.backward != nil
		fmt.Printf("  [%d] shape=%v grad=%v backward=%v\n", i, shapeOf(tt), hasGrad, hasBackward)
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
	locals []Tensor // this ctx: tensors created through this ctx

	op OpType

	inputs      []Tensor
	hiddenState []Tensor
	result      Tensor
	metadata    any
	backward    BackwardFn
}

func (sc *subContext) Mark(t Tensor) {
	sc.parent.Mark(t)
}

func (sc *subContext) Finish(options ...subContextOption) {
	applySubContextOptions(sc, options...)

	if sc.gradEnabled {
		if sc.result != nil {
			sc.prepareResultForBackwardPass()
		} else if sc.backward != nil || len(sc.inputs) > 0 || len(sc.hiddenState) > 0 || sc.op != OpNone {
			panic("Preparing for backward pass requires a result tensor")
		}
	}

	sc.cleanup()
	if sc.sweepAfterFinish {
		sc.parent.Sweep()
	}
}

func (sc *subContext) cleanup() {
	if sc.persistant {
		fmt.Printf("Ignoring tensors in persisitent context")
		return
	}

	for _, t := range sc.locals {
		// Some context may not produce a result so it's worth checking if the result is present
		if sc.result != nil && (t.(*tensor).cTensor == sc.result.(*tensor).cTensor || t.(*tensor).cTensor == sc.result.Computation().grad.(*tensor).cTensor) {
			continue
		}

		sc.Mark(t)
		// Also mark the grad tensor if it was allocated by leafNode — it won't be swept otherwise.
		if tt := t.(*tensor); tt.computation != nil && tt.computation.grad != nil {
			sc.Mark(tt.computation.grad.(Tensor))
		}
	}

	sc.locals = nil
}

// NoGrad returns a new Context that shares the same C context and memory
// but has gradient tracking disabled. In this context gradients are not created for new tensors
func (c *subContext) NoGrad(options ...subContextOption) SubContext {
	return noGrad(c, options...)
}

// Epoch returns a short-lived context for one training/inference step.
// It preserves the parent context's grad/backward behavior while keeping
// locals isolated so callers can Finish/Sweep at epoch boundaries.
func (c *subContext) Epoch(options ...subContextOption) SubContext {
	return epoch(c, options...)
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

func (sc *subContext) Track(t Tensor) {
	sc.parent.Track(t)
	sc.locals = append(sc.locals, t)

}

// NewComputationGraphNode attaches a computation graph node to result.
// Used by external packages (e.g., layer, activation) to register custom backward passes.
func (c *subContext) prepareResultForBackwardPass() {
	toComputationGraphNode(c.result, c.op, c.backward, c.inputs, c.hiddenState, c.metadata)
}

func (c *subContext) BackwardEnabled() bool {
	return c.backwardEnabled
}

func (c *subContext) GradEnabled() bool {
	return c.gradEnabled
}

// UnsafePtr returns the C Context as an unsafe.Pointer for cross-package CGo casts.
func (c *subContext) UnsafePtr() unsafe.Pointer {
	return c.parent.UnsafePtr()
}

// UnsafeMemory returns the C Memory as an unsafe.Pointer for cross-package CGo casts.
func (c *subContext) UnsafeMemory() unsafe.Pointer {
	return c.parent.UnsafeMemory()
}

func (c *subContext) NumFreeBlocks() int {
	return c.parent.NumFreeBlocks()
}

func (c *subContext) NumAllocatedBlocks() int {
	return c.parent.NumAllocatedBlocks()
}

func (ctx *subContext) Sweep() {
	ctx.parent.Sweep()
}

// ---------------------------- funcs ----------------------------

func applySubContextOptions(sc *subContext, options ...subContextOption) {
	for _, opt := range options {
		opt(sc)
	}
}

// derive creates a new derived context sharing the same C context and root.
// Fresh locals slice; handles and intermediates are root-owned.
func derive(ctx Context, gradEnabled, backwardEnabled bool, options ...subContextOption) SubContext {
	sc := &subContext{
		parent: ctx,
		locals: make([]Tensor, 0),
		shapesCtx: shapesCtx{
			backwardEnabled: backwardEnabled,
			gradEnabled:     gradEnabled,
		},
	}

	applySubContextOptions(sc, options...)
	return sc
}

func noGrad(c Context, options ...subContextOption) SubContext {
	if !c.GradEnabled() {
		panic("Creating NoGrad context from a context where gradients are not tracked")
	}

	return derive(c, false, c.BackwardEnabled(), options...)
}

// Epoch returns a short-lived context intended for one training step.
// It preserves grad/backward settings from the parent while providing
// independent local tracking for cleanup.
func epoch(c Context, options ...subContextOption) SubContext {
	if !c.GradEnabled() || !c.BackwardEnabled() {
		panic("Cannot create epoch context when gradients or backward are disabled")
	}

	if sc, ok := c.(*subContext); ok && sc.sweepAfterFinish {
		panic("Cannot create an epoch context from another epoch context")
	}

	sc := derive(c, c.GradEnabled(), c.BackwardEnabled(), options...)
	sc.(*subContext).sweepAfterFinish = true
	return sc
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
	return derive(c, false, false, options...)
}

func toComputationGraphNode(
	result Tensor,
	op OpType,
	backwardFn BackwardFn,
	inputs []Tensor,
	hiddenState []Tensor,
	metadata any,
) {

	if backwardFn == nil {
		panic("Preparing for backward pass but no backward function was provided")
	}

	resultTensor := result.(*tensor)

	resultTensor.computation.inputs = inputs
	resultTensor.computation.backward = backwardFn
	resultTensor.computation.op = op
	resultTensor.computation.meta = metadata

	if resultTensor.computation.hiddenState == nil {
		resultTensor.computation.hiddenState = []Tensor{}
		resultTensor.computation.hiddenState = hiddenState
	}
}
