package shapes

import (
	stdctx "context"
	"fmt"
	"os"
	"runtime/pprof"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "cwrappers.h"
*/
import "C"

type Context interface {
	Mark(t Tensor)
	Track(t Tensor)

	Epoch(epoch int, options ...subContextOption) EpochContext
	Forward(options ...subContextOption) SubContext
	Backward(options ...subContextOption) SubContext
	NoGrad(options ...subContextOption) SubContext
	Fused(options ...subContextOption) SubContext
	NoGraph(options ...subContextOption) SubContext

	GradEnabled() bool
	BackwardEnabled() bool
	IsTraining() bool

	UnsafeMemory() unsafe.Pointer
	UnsafePtr() unsafe.Pointer

	NumAllocatedBlocks() int
	NumFreeBlocks() int
	Sweep()

	main() *mainContext
}

type MainContext interface {
	Context

	Finish()
	Training(numEpochs int, options ...mainContextOption) MainContext
	Inference() MainContext
	TrainingStats() *TrainingStats
}

type SubContext interface {
	Context
	Finish(...subContextOption)
}

type EpochContext interface {
	SubContext
	SampleTensor(key string, t Tensor)
	CurrentEpochNum() int
}

type shapesCtx struct {
	stdctx.Context // Embedded Go context for cancellation/deadlines

	gradEnabled      bool
	backwardEnabled  bool
	isTraining       bool
	persistant       bool
	sweepAfterFinish bool

	training *trainingState
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

type TrainingStats struct {
	Epoch                int
	NumEpochs            int
	Loss                 float64
	MemorySampleHistoryX []int
	UsedBlocksHistory    []int
	LossHistoryX         []int
	LossHistory          []int
	Version              int
	TrainingDone         chan bool

	SampleTensors map[string]Tensor
}

type trainingState struct {
	mu       sync.Mutex
	doneOnce sync.Once
	inEpoch  atomic.Bool
	stats    *TrainingStats
}

type TrainingStatsRenderer interface {
	SetTrainingContext(trainingCtx MainContext)
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

func (c *mainContext) initTrainingStats(numEpochs int) {
	c.training = &trainingState{
		stats: &TrainingStats{
			NumEpochs:            numEpochs,
			Epoch:                0,
			Loss:                 0,
			MemorySampleHistoryX: make([]int, 0),
			UsedBlocksHistory:    make([]int, 0),
			LossHistoryX:         make([]int, 0),
			LossHistory:          make([]int, 0),
			Version:              0,
			TrainingDone:         make(chan bool),
			SampleTensors:        make(map[string]Tensor),
		},
	}
}

func (c *mainContext) Training(numEpochs int, options ...mainContextOption) MainContext {
	c.initTrainingStats(numEpochs)
	c.isTraining = true
	applyMainContextOptions(c, options...)

	f, err := os.Create("profile.prof")
	if err != nil {
		msg := fmt.Sprintf("Could not open profile file: %v", err)
		panic(msg)
	}

	err = pprof.StartCPUProfile(f)
	if err != nil {
		_ = f.Close()
		return c
	}

	go startMemoryTicker(c)

	return c
}

func (c *mainContext) Inference() MainContext {
	c.isTraining = false
	return c
}

func (c *mainContext) sampleMemory(sampleCount int) {
	if c.training == nil || c.training.stats == nil {
		return
	}
	if c.cCtx == nil || c.cCtx.memory == nil {
		return
	}

	usedBlocks := c.NumAllocatedBlocks()
	c.training.mu.Lock()
	c.training.stats.MemorySampleHistoryX = append(c.training.stats.MemorySampleHistoryX, sampleCount)
	c.training.stats.UsedBlocksHistory = append(c.training.stats.UsedBlocksHistory, usedBlocks)
	c.training.stats.Version++
	c.training.mu.Unlock()
}

func (c *mainContext) TrainingStats() *TrainingStats {
	if c.training == nil {
		return nil
	}

	return c.training.stats
}

// Epoch returns a short-lived context for one training/inference step.
// It preserves the parent context's grad/backward behavior while keeping
// locals isolated so callers can Finish/Sweep at epoch boundaries.
func (c *mainContext) Epoch(currentEpoch int, options ...subContextOption) EpochContext {
	return epoch(c, currentEpoch, options...)
}

// Forward returns a context intended for forward-pass fused operations.
// This is an alias for Fused.
func (c *mainContext) Forward(options ...subContextOption) SubContext {
	return fused(c, options...)
}

// Backward returns a context intended for backward-pass computation.
// This is an alias for NoGraph.
func (c *mainContext) Backward(options ...subContextOption) SubContext {
	return noGraph(c, options...)
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
	ctx.handles = live
}

func (c *mainContext) BackwardEnabled() bool {
	return c.backwardEnabled
}

func (c *mainContext) GradEnabled() bool {
	return c.gradEnabled
}

func (c *mainContext) main() *mainContext {
	return c
}

func (c *mainContext) IsTraining() bool {
	return c.isTraining
}

// ---------------------------- Subcontext ----------------------------

type SubContextType int

const (
	SubContextTypeEpoch SubContextType = iota
	SubContextTypeForward
	SubContextTypeFused
	SubContextTypeNoGrad
	SubContextTypeNoGraph
)

type subContext struct {
	shapesCtx
	parent Context
	locals []Tensor // this ctx: tensors created through this ctx

	subContextType SubContextType
	op             OpType

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

	mainCtx := sc.main()
	if mainCtx.training != nil && mainCtx.training.inEpoch.Load() && sc.subContextType == SubContextTypeEpoch {
		mainCtx.training.inEpoch.Store(false)
	}

	// Signal training completion exactly once at the end of the final epoch context.
	// Closing the channel broadcasts completion to all listeners without risking a blocked send.
	if sc.sweepAfterFinish && sc.training != nil && sc.training.stats != nil &&
		sc.training.stats.Epoch == sc.training.stats.NumEpochs {
		sc.training.doneOnce.Do(func() {
			sc.main().isTraining = false
			close(sc.training.stats.TrainingDone)
			pprof.StopCPUProfile()
		})
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
			sc.Mark(tt.computation.grad)
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
func (c *subContext) Epoch(currentEpoch int, options ...subContextOption) EpochContext {
	return epoch(c, currentEpoch, options...)
}

// Forward returns a context intended for forward-pass fused operations.
// This is an alias for Fused.
func (c *subContext) Forward(options ...subContextOption) SubContext {
	return fused(c, options...)
}

// Backward returns a context intended for backward-pass computation.
// This is an alias for NoGraph.
func (c *subContext) Backward(options ...subContextOption) SubContext {
	return noGraph(c, options...)
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

func (c *subContext) IsTraining() bool {
	return c.isTraining
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

func (c *subContext) CurrentEpochNum() int {
	return c.training.stats.Epoch
}

func (ctx *subContext) Sweep() {
	ctx.parent.Sweep()
}

func (ctx *subContext) SampleTensor(key string, t Tensor) {
	if ctx.training == nil {
		panic("Cannot sample a tensor in a non training context")
	}

	ctx.training.mu.Lock()
	defer ctx.training.mu.Unlock()

	ctx.training.stats.SampleTensors[key] = t.Clone(ctx.main())
}

func (ctx *subContext) main() *mainContext {
	return ctx.parent.main()
}

// ---------------------------- funcs ----------------------------

func applySubContextOptions(sc *subContext, options ...subContextOption) {
	for _, opt := range options {
		opt(sc)
	}
}

func applyMainContextOptions(sc *mainContext, options ...mainContextOption) {
	for _, opt := range options {
		opt(sc)
	}
}

// derive creates a new derived context sharing the same C context and root.
// Fresh locals slice; handles and intermediates are root-owned.
func derive(ctx Context, subContextType SubContextType, gradEnabled, backwardEnabled bool, options ...subContextOption) SubContext {
	var training *trainingState
	switch parent := ctx.(type) {
	case *mainContext:
		training = parent.training
	case *subContext:
		training = parent.training
	}

	sc := &subContext{
		parent:         ctx,
		locals:         make([]Tensor, 0),
		subContextType: subContextType,
		shapesCtx: shapesCtx{
			backwardEnabled: backwardEnabled,
			gradEnabled:     gradEnabled,
			isTraining:      ctx.IsTraining(),
			training:        training,
		},
	}

	applySubContextOptions(sc, options...)
	return sc
}

func noGrad(c Context, options ...subContextOption) SubContext {
	if !c.GradEnabled() {
		panic("Creating NoGrad context from a context where gradients are not tracked")
	}

	return derive(c, SubContextTypeNoGrad, false, c.BackwardEnabled(), options...)
}

// Epoch returns a short-lived context intended for one training step.
// It preserves grad/backward settings from the parent while providing
// independent local tracking for cleanup.
func epoch(c Context, epochNum int, options ...subContextOption) EpochContext {

	if epochNum < 1 {
		panic("Epoch number cannot be less than 1")
	}

	if !c.GradEnabled() || !c.BackwardEnabled() {
		panic("Cannot create epoch context when gradients or backward are disabled")
	}

	if sc, ok := c.(*subContext); ok && sc.sweepAfterFinish {
		panic("Cannot create an epoch context from another epoch context")
	}

	if mc, ok := c.(*mainContext); ok {
		if mc.training == nil || mc.training.stats == nil {
			panic("Can only call epoch on a training context")
		}

		if mc.training.inEpoch.Load() {
			panic("Cannot create a new epoch context while one is still running. Did you forget to finish the epoch?")
		}

		mc.training.inEpoch.Store(true)

		mc.training.mu.Lock()
		mc.training.stats.Epoch = epochNum
		mc.training.mu.Unlock()
	} else {
		panic("You can only call Epoch() in the main context")
	}

	sc := derive(c, SubContextTypeEpoch, c.GradEnabled(), c.BackwardEnabled(), options...)
	sc.(*subContext).sweepAfterFinish = true
	return sc.(EpochContext)
}

// Fused returns a new Context that shares the same C context and memory
// but turns of backward passes for any ops used in that context.
// Meant for doing compund operations where the caller might want to specify the backward pass manually
func fused(c Context, options ...subContextOption) SubContext {
	if !c.GradEnabled() {
		panic("Creating fused context from a context where gradients are not tracked")
	}

	return derive(c, SubContextTypeFused, true, false, options...)
}

// NoGraph() returns a new Context that shares the same C context and memory
// turns off all gradient tacking. For use in backward pass only
func noGraph(c Context, options ...subContextOption) SubContext {
	return derive(c, SubContextTypeNoGraph, false, false, options...)
}

func startMemoryTicker(c *mainContext) {
	if c.training == nil || c.training.stats == nil {
		return
	}

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	sampleCount := 1

	c.sampleMemory(sampleCount)
	for {
		select {
		case <-c.training.stats.TrainingDone:
			c.sampleMemory(sampleCount)
			sampleCount++
			return
		case <-ticker.C:
			c.sampleMemory(sampleCount)
			sampleCount++
		}
	}
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
