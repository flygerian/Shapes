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
#include "cwrappers.h"
*/
import "C"

type Context interface {
	Mark(t Tensor)
	Track(t Tensor)

	Epoch(epoch int, options ...subContextOption) EpochContext
	Forward(options ...subContextOption) SubContext
	BackwardDisabled(options ...subContextOption) SubContext
	Backward(options ...subContextOption) SubContext
	NoGrad(options ...subContextOption) SubContext
	Fused(options ...subContextOption) SubContext
	NoGraph(options ...subContextOption) SubContext
	Test(options ...subContextOption) SubContext

	GradEnabled() bool
	BackwardEnabled() bool
	IsTraining() bool

	Training(numEpochs int, options ...subContextOption) SubContext
	Inference() SubContext
	TrainingStats() *TrainingStats

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
}

type SubContext interface {
	Context
	Finish(...subContextOption)
}

type EpochContext interface {
	SubContext
	Step(options ...subContextOption) EpochContext
	SampleTensor(key string, t Tensor)
	CurrentEpochNum() int
	SetLoss(loss float64)
	SetValidationLoss(loss float64)
	SetTestLoss(loss float64)
	SetAccuracy(accuracy float64)

	SetStepLoss(loss float64)
	CurrentStep() int
	Fused(options ...subContextOption) SubContext
}

type StepContext interface {
	SubContext
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
		ctx.cCtx = C.CreateContext(C.size_t(ctx.arenaSize), 1, C.bool(ctx.withCuda))
	} else {
		ctx.cCtx = C.CreateContext(1024*1024*64, 1, C.bool(ctx.withCuda))
	}

	return ctx
}

type TrainingStats struct {
	Epoch                  int
	EpochStart             time.Time
	Step                   int
	NumEpochs              int
	NumSteps               int
	Loss                   float64
	StepLoss               float64
	TestLoss               float64
	ValidationLoss         float64
	Accuracy               float64
	MemorySampleHistoryX   []int
	UsedBlocksHistory      []int
	LossHistoryX           []int
	LossHistory            []int
	StepLossHistoryX       []int
	StepLossHistory        []int
	TestLossHistoryX       []int
	TestLossHistory        []int
	ValidationLossHistoryX []int
	ValidationLossHistory  []int
	AccuracyHistoryX       []int
	AccuracyHistory        []int
	Version                int
	TrainingDone           chan bool

	SampleTensors map[string]Tensor
}

type trainingState struct {
	mu       sync.Mutex
	doneOnce sync.Once
	inEpoch  atomic.Bool
	stats    *TrainingStats
}

type TrainingStatsRenderer interface {
	SetTrainingContext(trainingCtx SubContext)
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
	withCuda          bool
}

// Finish releases the C context resources.
// All tensors created under this context become invalid after Finish.
// Only the root context frees memory; derived contexts stage their locals
// into the freeList for later sweeping.
func (c *mainContext) Finish() {
	if c.cCtx != nil && c.cCtx.memory != nil {
		if result := C.Flush(c.cCtx); result != C.OK {
			panic("shapes: " + resultString(uint32(result)))
		}

		for _, p := range c.handles {
			if p != nil && p.(*tensor).cTensor != nil {
				c.Free(p)
			}
		}

		c.handles = nil
		C.FreeContext(c.cCtx)
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

func (c *mainContext) Inference() SubContext {
	subCtx := derive(c, SubContextTypeInference, c.gradEnabled, c.backwardEnabled)
	subCtx.main().isTraining = false
	return subCtx
}

func (c *mainContext) TrainingStats() *TrainingStats {
	if c.training == nil {
		return nil
	}

	return c.training.stats
}

// Epoch returns a short-lived context for one training epoch.
// It preserves the parent context's grad/backward behavior while keeping
// locals isolated so callers can Finish/Sweep at epoch boundaries.
func (c *mainContext) Epoch(currentEpoch int, options ...subContextOption) EpochContext {
	return epoch(c, currentEpoch, options...)
}

// Forward returns a context intended for forward-pass composite operations.
// This is an alias for BackwardDisabled.
func (c *mainContext) Forward(options ...subContextOption) SubContext {
	return backwardDisabled(c, options...)
}

// BackwardDisabled returns a new Context that shares the same C context and memory
// but turns off backward passes for any ops used in that context.
// Meant for compound operations where the caller provides a custom backward pass.
func (c *mainContext) BackwardDisabled(options ...subContextOption) SubContext {
	return backwardDisabled(c, options...)
}

// Backward returns a context intended for backward-pass computation.
// This is an alias for NoGraph.
func (c *mainContext) Backward(options ...subContextOption) SubContext {
	return noGraph(c, options...)
}

// Fused returns a new Context that shares the same C context and memory
// and marks Finish as an execution boundary that should flush queued device work.
func (c *mainContext) Fused(options ...subContextOption) SubContext {
	return fused(c, options...)
}

// NoGraph() returns a new Context that shares the same C context and memory
// turns off all gradient tacking. For use in backward pass only
func (c *mainContext) NoGraph(options ...subContextOption) SubContext {
	return noGraph(c, options...)
}

// Test returns a new Context that shares the same C context and memory
// with gradients and backward pass disabled. Suitable for validation/testing.
func (c *mainContext) Test(options ...subContextOption) SubContext {
	return test(c, options...)
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
	if len(ctx.freeList) == 0 {
		return
	}

	// Deduplicate to prevent double-free and remember every freed C tensor so
	// all Go wrappers pointing at it can be invalidated afterward.
	seen := make(map[unsafe.Pointer]bool, len(ctx.freeList))
	freed := make(map[unsafe.Pointer]struct{}, len(ctx.freeList))
	dedupFreeList := make([]*C.Tensor, 0)

	for _, t := range ctx.freeList {
		if t == nil || t.(*tensor).cTensor == nil {
			continue
		}

		p := unsafe.Pointer(t.(*tensor).cTensor)
		if seen[p] {
			continue
		}

		seen[p] = true
		freed[p] = struct{}{}
		dedupFreeList = append(dedupFreeList, t.(*tensor).cTensor)
	}

	if len(dedupFreeList) > 0 {
		C.FreeTensors((*C.Context)(ctx.UnsafePtr()), &dedupFreeList[0], C.int(len(dedupFreeList)))
	}

	for _, t := range ctx.freeList {
		if t == nil || t.(*tensor).cTensor == nil {
			continue
		}

		if _, ok := freed[unsafe.Pointer(t.(*tensor).cTensor)]; ok {
			t.(*tensor).cTensor = nil
		}
	}

	ctx.freeList = ctx.freeList[:0]

	// Compact root.handles: drop anything swept here, including alias wrappers
	// that were not present in freeList but still point at a freed C tensor.
	live := ctx.handles[:0]
	for _, t := range ctx.handles {
		if t == nil {
			continue
		}

		if t.(*tensor).cTensor != nil {
			if _, ok := freed[unsafe.Pointer(t.(*tensor).cTensor)]; ok {
				t.(*tensor).cTensor = nil
			}
		}

		if t.(*tensor).cTensor != nil {
			live = append(live, t)
		}
	}
	ctx.handles = live
}

func (c *mainContext) initTrainingStats(numEpochs int) {
	c.training = &trainingState{
		stats: &TrainingStats{
			NumEpochs:              numEpochs,
			NumSteps:               0,
			Epoch:                  0,
			Step:                   0,
			Loss:                   0,
			StepLoss:               0,
			TestLoss:               0,
			ValidationLoss:         0,
			Accuracy:               0,
			MemorySampleHistoryX:   make([]int, 0),
			UsedBlocksHistory:      make([]int, 0),
			LossHistoryX:           make([]int, 0),
			LossHistory:            make([]int, 0),
			StepLossHistoryX:       make([]int, 0),
			StepLossHistory:        make([]int, 0),
			TestLossHistoryX:       make([]int, 0),
			TestLossHistory:        make([]int, 0),
			ValidationLossHistoryX: make([]int, 0),
			ValidationLossHistory:  make([]int, 0),
			AccuracyHistoryX:       make([]int, 0),
			AccuracyHistory:        make([]int, 0),
			Version:                0,
			TrainingDone:           make(chan bool),
			SampleTensors:          make(map[string]Tensor),
		},
	}
}

func (c *mainContext) sampleMemory(sampleCount int) {
	if c.training == nil || c.training.stats == nil {
		return
	}
	if c.main().cCtx == nil || c.main().cCtx.memory == nil {
		return
	}

	usedBlocks := c.NumAllocatedBlocks()
	c.training.mu.Lock()
	c.training.stats.MemorySampleHistoryX = append(c.training.stats.MemorySampleHistoryX, sampleCount)
	c.training.stats.UsedBlocksHistory = append(c.training.stats.UsedBlocksHistory, usedBlocks)
	c.training.stats.Version++
	c.training.mu.Unlock()
}

func (c *mainContext) Training(numEpochs int, options ...subContextOption) SubContext {

	c.initTrainingStats(numEpochs)
	c.isTraining = true

	trainingCtx := derive(c, SubContextTypeTraining, true, true, options...)

	f, err := os.Create("profile.prof")
	if err != nil {
		msg := fmt.Sprintf("Could not open profile file: %v", err)
		panic(msg)
	}

	err = pprof.StartCPUProfile(f)
	if err != nil {
		msg := fmt.Sprintf("Could not start CPU profile: %v", err)
		panic(msg)
		_ = f.Close()
		return trainingCtx
	}

	go startMemoryTicker(trainingCtx.(*subContext))

	return trainingCtx
}

func (ctx *subContext) SetAccuracy(accuracy float64) {
	if ctx.training == nil {
		panic("Cannot set accuracy in a non training context")
	}

	ctx.training.mu.Lock()
	defer ctx.training.mu.Unlock()

	ctx.training.stats.Accuracy = accuracy
	ctx.training.stats.AccuracyHistoryX = append(ctx.training.stats.AccuracyHistoryX, ctx.training.stats.Epoch)
	ctx.training.stats.AccuracyHistory = append(ctx.training.stats.AccuracyHistory, int(accuracy*100))
	ctx.training.stats.Version++
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
	SubContextTypeTraining
	SubContextTypeInference
	SubContextTypeFused
	SubContextTypeNoGrad
	SubContextTypeNoGraph
	SubContextTypeTest
)

type subContext struct {
	shapesCtx
	parent Context
	locals []Tensor // this ctx: tensors created through this ctx

	subContextType SubContextType
	fused          bool
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

func (sc *subContext) hasFusedAncestor() bool {
	for parent := sc.parent; parent != nil; {
		sub, ok := parent.(*subContext)
		if !ok {
			return false
		}
		if sub.fused {
			return true
		}
		parent = sub.parent
	}

	return false
}

func (sc *subContext) shouldFlushOnFinish() bool {
	hasFusedAncestor := sc.hasFusedAncestor()
	if sc.fused && !hasFusedAncestor {
		return true
	}

	if hasFusedAncestor {
		return false
	}

	return true
}

func (sc *subContext) Finish(options ...subContextOption) {
	start := time.Now()
	applySubContextOptions(sc, options...)

	if sc.subContextType == SubContextTypeEpoch {
		fmt.Printf("Ending epoch...%d", sc.CurrentEpochNum())
	}

	if sc.subContextType == SubContextTypeTraining {
		fmt.Printf("Stopping cpu profile...")
		pprof.StopCPUProfile()
	}

	if sc.gradEnabled {
		if sc.result != nil {
			sc.prepareResultForBackwardPass()
		} else if sc.backward != nil || len(sc.inputs) > 0 || len(sc.hiddenState) > 0 || sc.op != OpNone {
			panic("Preparing for backward pass requires a result tensor")
		}
	}

	if sc.shouldFlushOnFinish() {
		if result := C.Flush((*C.Context)(sc.UnsafePtr())); result != C.OK {
			panic("shapes: " + resultString(uint32(result)))
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
		sc.training.stats.Epoch > sc.training.stats.NumEpochs {
		sc.training.doneOnce.Do(func() {
			sc.main().isTraining = false
			close(sc.training.stats.TrainingDone)
		})
	}

	if value := os.Getenv("SHAPES_LOG_CONTEXT_FINISH"); value != "" && value != "0" {
		fmt.Fprintf(os.Stderr, "[contextFinish] type=%d fused=%t flush=%t locals=%d ms=%.3f\n",
			sc.subContextType, sc.fused, sc.shouldFlushOnFinish(), len(sc.locals),
			float64(time.Since(start))/float64(time.Millisecond))
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
			sc.main().Track(t)
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

// Epoch returns a short-lived context for one training epoch.
// It preserves the parent context's grad/backward behavior while keeping
// locals isolated so callers can Finish/Sweep at epoch boundaries.
func (c *subContext) Epoch(currentEpoch int, options ...subContextOption) EpochContext {
	return epoch(c, currentEpoch, options...)
}

func (c *subContext) Step(options ...subContextOption) EpochContext {
	return step(c, options...)
}

// Forward returns a context intended for forward-pass composite operations.
// This is an alias for BackwardDisabled.
func (c *subContext) Forward(options ...subContextOption) SubContext {
	return backwardDisabled(c, options...)
}

// BackwardDisabled returns a new Context that shares the same C context and memory
// but turns off backward passes for any ops used in that context.
// Meant for compound operations where the caller provides a custom backward pass.
func (c *subContext) BackwardDisabled(options ...subContextOption) SubContext {
	return backwardDisabled(c, options...)
}

// Backward returns a context intended for backward-pass computation.
// This is an alias for NoGraph.
func (c *subContext) Backward(options ...subContextOption) SubContext {
	return noGraph(c, options...)
}

// Fused returns a new Context that shares the same C context and memory
// and marks Finish as an execution boundary that should flush queued device work.
func (c *subContext) Fused(options ...subContextOption) SubContext {
	return fused(c, options...)
}

// NoGraph() returns a new Context that shares the same C context and memory
// turns off all gradient tacking. For use in backward pass only
func (c *subContext) NoGraph(options ...subContextOption) SubContext {
	return noGraph(c, options...)
}

// Test returns a new Context that shares the same C context and memory
// with gradients and backward pass disabled. Suitable for validation/testing.
func (c *subContext) Test(options ...subContextOption) SubContext {
	return test(c, options...)
}

func (sc *subContext) Track(t Tensor) {
	// sc.main().Track(t)
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

func (ctx *subContext) SetLoss(loss float64) {
	if ctx.training == nil {
		panic("Cannot set validation loss in a non training context")
	}

	ctx.training.mu.Lock()
	defer ctx.training.mu.Unlock()

	ctx.training.stats.Loss = loss
	ctx.training.stats.LossHistoryX = append(ctx.training.stats.LossHistoryX, ctx.training.stats.Epoch)
	ctx.training.stats.LossHistory = append(ctx.training.stats.LossHistory, int(loss*1000))
	ctx.training.stats.Version++
}

func (ctx *subContext) SetStepLoss(loss float64) {
	if ctx.training == nil {
		panic("Cannot set step loss in a non training context")
	}
	if ctx.subContextType != SubContextTypeEpoch {
		panic("Step loss can only be set on an Epoch context")
	}

	ctx.training.mu.Lock()
	defer ctx.training.mu.Unlock()

	ctx.training.stats.StepLoss = loss
	ctx.training.stats.StepLossHistoryX = append(ctx.training.stats.StepLossHistoryX, ctx.training.stats.Step)
	ctx.training.stats.StepLossHistory = append(ctx.training.stats.StepLossHistory, int(loss*1000))
	ctx.training.stats.Version++
}

func (ctx *subContext) CurrentStep() int {

	if ctx.training == nil {
		panic("Cannot set step loss in a non training context")
	}
	if ctx.subContextType != SubContextTypeEpoch {
		panic("Step loss can only be set on an epoch context")
	}

	return ctx.training.stats.Step
}

func (ctx *subContext) SetValidationLoss(loss float64) {
	if ctx.training == nil {
		panic("Cannot set validation loss in a non training context")
	}

	ctx.training.mu.Lock()
	defer ctx.training.mu.Unlock()

	ctx.training.stats.ValidationLoss = loss
	ctx.training.stats.ValidationLossHistoryX = append(ctx.training.stats.ValidationLossHistoryX, ctx.training.stats.Epoch)
	ctx.training.stats.ValidationLossHistory = append(ctx.training.stats.ValidationLossHistory, int(loss*1000))
	ctx.training.stats.Version++
}

func (ctx *subContext) SetTestLoss(loss float64) {
	if ctx.training == nil {
		panic("Cannot set test loss in a non training context")
	}

	ctx.training.mu.Lock()
	defer ctx.training.mu.Unlock()

	ctx.training.stats.TestLoss = loss
	ctx.training.stats.TestLossHistoryX = append(ctx.training.stats.TestLossHistoryX, ctx.training.stats.Epoch)
	ctx.training.stats.TestLossHistory = append(ctx.training.stats.TestLossHistory, int(loss*1000))
	ctx.training.stats.Version++
}

func (c *subContext) initTrainingStats(numEpochs int) {
	c.main().initTrainingStats(numEpochs)
}

func (c *subContext) sampleMemory(sampleCount int) {
	c.main().sampleMemory(sampleCount)
}

func (c *subContext) Training(numEpochs int, options ...subContextOption) SubContext {
	return c.main().Training(numEpochs, options...)
}

func (c *subContext) TrainingStats() *TrainingStats {
	return c.main().TrainingStats()
}

func (c *subContext) Inference() SubContext {
	return c.main().Inference()
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
func derive(ctx Context, subContextType SubContextType, gradEnabled bool, backwardEnabled bool, options ...subContextOption) SubContext {
	var training *trainingState
	var fused bool
	switch parent := ctx.(type) {
	case *mainContext:
		training = parent.training
	case *subContext:
		training = parent.training
		fused = parent.fused
	}

	sc := &subContext{
		parent:         ctx,
		locals:         make([]Tensor, 0),
		subContextType: subContextType,
		fused:          fused,
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

// Epoch returns a short-lived context intended for one training epoch.
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

	if mc, ok := c.(*subContext); ok {
		if mc.training == nil || mc.training.stats == nil {
			panic("Can only call epoch on a training context")
		}

		if mc.training.inEpoch.Load() {
			panic("Cannot create a new epoch context while one is still running. Did you forget to finish the epoch?")
		}

		mc.training.inEpoch.Store(true)

		mc.training.mu.Lock()
		mc.training.stats.Epoch = epochNum
		mc.training.stats.EpochStart = time.Now()
		mc.training.stats.Step = 0
		mc.training.mu.Unlock()
	} else {
		panic("You can only call Epoch() in the main context")
	}

	sc := derive(c, SubContextTypeEpoch, c.GradEnabled(), c.BackwardEnabled(), options...)
	sc.(*subContext).sweepAfterFinish = true
	return sc.(EpochContext)
}

func step(c Context, options ...subContextOption) EpochContext {
	sc, ok := c.(*subContext)
	if !ok || sc.subContextType != SubContextTypeEpoch {
		panic("You can only call Step() on an epoch context")
	}
	if sc.training == nil || sc.training.stats == nil {
		panic("Can only call Step() in training mode")
	}

	sc.training.mu.Lock()
	sc.training.stats.Step++
	sc.training.mu.Unlock()

	stepCtx := derive(c, SubContextTypeEpoch, c.GradEnabled(), c.BackwardEnabled(), options...)
	stepCtx.(*subContext).sweepAfterFinish = true
	return stepCtx.(EpochContext)
}

// BackwardDisabled returns a new Context that shares the same C context and memory
// but turns off backward passes for any ops used in that context.
// Meant for compound operations where the caller might want to specify the backward pass manually.
func backwardDisabled(c Context, options ...subContextOption) SubContext {
	return derive(c, SubContextTypeForward, true, false, options...)
}

// Fused returns a new Context that shares the same C context and memory
// and marks Finish as an execution boundary that should flush queued device work.
func fused(c Context, options ...subContextOption) SubContext {
	subContextType := SubContextTypeFused
	sweepAfterFinish := false
	persistant := false
	if parent, ok := c.(*subContext); ok {
		subContextType = parent.subContextType
		sweepAfterFinish = parent.sweepAfterFinish
		persistant = parent.persistant
	}

	sc := derive(c, subContextType, c.GradEnabled(), c.BackwardEnabled(), options...)
	sc.(*subContext).fused = true
	sc.(*subContext).sweepAfterFinish = sweepAfterFinish
	sc.(*subContext).persistant = persistant
	return sc
}

// NoGraph() returns a new Context that shares the same C context and memory
// turns off all gradient tacking. For use in backward pass only
func noGraph(c Context, options ...subContextOption) SubContext {
	return derive(c, SubContextTypeNoGraph, false, false, options...)
}

// Test returns a new Context that shares the same C context and memory
// with gradients and backward pass disabled. Suitable for validation/testing.
func test(c Context, options ...subContextOption) SubContext {
	return derive(c, SubContextTypeTest, false, false, options...)
}

func startMemoryTicker(c *subContext) {
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
