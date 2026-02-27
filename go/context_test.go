package shapes

import (
	stdctx "context"
	"testing"
	"time"
)

// TestNew verifies that New() creates a valid context.
func TestNew(t *testing.T) {
	ctx := New(stdctx.Background())
	if ctx == nil {
		t.Fatal("New() returned nil")
	}
	if ctx.Ptr() == nil {
		t.Fatal("Context has nil C pointer")
	}
	defer ctx.Finish()
}

// TestNewWithParent verifies that the embedded Go context is propagated correctly.
func TestNewWithParent(t *testing.T) {
	parent := stdctx.Background()
	ctx := New(parent)
	defer ctx.Finish()

	// Done() should return the embedded stdctx channel (not a void cleanup method).
	if ctx.Done() != parent.Done() {
		t.Error("Done() should match parent's cancellation channel")
	}
}

// TestWithGrad verifies the WithGrad option.
func TestWithGrad(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()
	if ctx.gradEnabled {
		t.Error("grad should be disabled by default")
	}

	ctx2 := New(stdctx.TODO(), WithGrad(true))
	defer ctx2.Finish()
	if !ctx2.gradEnabled {
		t.Error("grad should be enabled with WithGrad(true)")
	}

	ctx3 := New(stdctx.Background(), WithGrad(false))
	defer ctx3.Finish()
	if ctx3.gradEnabled {
		t.Error("grad should be disabled with WithGrad(false)")
	}
}

// TestFinish verifies that Finish() releases root memory and nils tensor pointers.
func TestFinish(t *testing.T) {
	ctx := New(stdctx.Background())
	t1 := Float(ctx, Shape{2, 2}, 1.0)
	t2 := Float(ctx, Shape{2, 2}, 2.0)

	if t1.cTensor == nil || t2.cTensor == nil {
		t.Fatal("tensors should be valid before Finish")
	}

	ctx.Finish()

	// All C pointers should be nilled after root Finish.
	if t1.cTensor != nil || t2.cTensor != nil {
		t.Fatal("tensor C pointers should be nilled after root Finish")
	}
}

// TestContextInterface verifies that Context satisfies the context.Context interface.
func TestContextInterface(t *testing.T) {
	parent, cancel := stdctx.WithTimeout(stdctx.Background(), 100*time.Millisecond)
	defer cancel()

	ctx := New(parent)
	defer ctx.Finish()

	deadline, ok := ctx.Deadline()
	if !ok {
		t.Fatal("should have deadline from parent")
	}
	if deadline.IsZero() {
		t.Error("deadline should not be zero")
	}

	select {
	case <-ctx.Done():
		// Expected: parent timeout fires the cancellation channel.
	case <-time.After(200 * time.Millisecond):
		t.Error("context should have been cancelled via Done() channel")
	}

	if ctx.Err() == nil {
		t.Error("should have cancellation error after timeout")
	}
}

// TestRootContextFreesMemory verifies that only the root context frees the arena.
func TestRootContextFreesMemory(t *testing.T) {
	root := New(stdctx.Background())

	t1 := Float(root, Shape{3, 3}, 1.0)
	t2 := Float(root, Shape{3, 3}, 2.0)

	fusedCtx := root.Fused()
	noGradCtx := root.NoGrad()
	noGraphCtx := root.NoGraph()

	t3 := Float(fusedCtx, Shape{2, 2}, 3.0)
	t4 := Float(noGradCtx, Shape{2, 2}, 4.0)
	t5 := Float(noGraphCtx, Shape{2, 2}, 5.0)

	// Derived Finish() should not free the arena.
	fusedCtx.Finish()
	noGradCtx.Finish()
	noGraphCtx.Finish()

	if t3.cTensor == nil || t4.cTensor == nil || t5.cTensor == nil {
		t.Fatal("tensors from derived contexts should still be valid after derived Finish")
	}

	// Root Finish() frees everything.
	root.Finish()

	if t1.cTensor != nil || t2.cTensor != nil || t3.cTensor != nil ||
		t4.cTensor != nil || t5.cTensor != nil {
		t.Fatal("all tensor C pointers should be nilled after root Finish")
	}
}

// TestSubcontextsShareRootMemory verifies derived contexts share the C arena and root pointer.
func TestSubcontextsShareRootMemory(t *testing.T) {
	root := New(stdctx.Background())
	defer root.Finish()

	fused := root.Fused()
	noGrad := root.NoGrad()
	noGraph := root.NoGraph()

	if fused.root != root || noGrad.root != root || noGraph.root != root {
		t.Fatal("derived contexts should share the same root")
	}

	if fused.ownsMemory || noGrad.ownsMemory || noGraph.ownsMemory {
		t.Fatal("derived contexts should not own memory")
	}

	if !root.ownsMemory {
		t.Fatal("root context should own memory")
	}

	if fused.cCtx != root.cCtx || noGrad.cCtx != root.cCtx || noGraph.cCtx != root.cCtx {
		t.Fatal("derived contexts should share the same C context pointer")
	}
}

// TestLocalsTracking verifies each context tracks its own locals independently.
func TestLocalsTracking(t *testing.T) {
	root := New(stdctx.Background())
	defer root.Finish()

	if len(root.locals) != 0 {
		t.Fatal("root should start with empty locals")
	}

	_ = Float(root, Shape{2, 2}, 1.0)
	_ = Float(root, Shape{2, 2}, 2.0)

	if len(root.locals) != 2 {
		t.Fatalf("root should have 2 locals, got %d", len(root.locals))
	}

	fused := root.Fused()
	_ = Float(fused, Shape{3, 3}, 3.0)

	if len(fused.locals) != 1 {
		t.Fatalf("fused should have 1 local, got %d", len(fused.locals))
	}
	if len(root.locals) != 2 {
		t.Fatalf("root should still have 2 locals after fused creation, got %d", len(root.locals))
	}
}

// TestHandlesTracking verifies all tensors regardless of context end up in root.handles.
func TestHandlesTracking(t *testing.T) {
	root := New(stdctx.Background())
	defer root.Finish()

	tensor1 := Float(root, Shape{2, 2}, 1.0)
	fused := root.Fused()
	tensor2 := Float(fused, Shape{2, 2}, 2.0)
	noGrad := root.NoGrad()
	tensor3 := Float(noGrad, Shape{2, 2}, 3.0)

	if len(root.handles) < 3 {
		t.Fatalf("root should have at least 3 handles, got %d", len(root.handles))
	}

	if tensor1.Get(root, 0, 0).Item().(float32) != 1.0 {
		t.Fatal("tensor1 should be accessible")
	}
	if tensor2.Get(root, 0, 0).Item().(float32) != 2.0 {
		t.Fatal("tensor2 should be accessible")
	}
	if tensor3.Get(root, 0, 0).Item().(float32) != 3.0 {
		t.Fatal("tensor3 should be accessible")
	}
}

// TestDerivedFinishStagesToFreeList verifies that Finish() on a derived context
// routes its locals into the root's freeList via Mark().
func TestDerivedFinishStagesToFreeList(t *testing.T) {
	root := New(stdctx.Background())
	defer root.Finish()

	fused := root.Fused()
	_ = Float(fused, Shape{2, 2}, 1.0)
	_ = Float(fused, Shape{2, 2}, 2.0)

	if len(fused.locals) != 2 {
		t.Fatalf("fused should have 2 locals before Finish, got %d", len(fused.locals))
	}

	fused.Finish()

	if len(fused.locals) != 0 {
		t.Fatalf("fused.locals should be empty after Finish, got %d", len(fused.locals))
	}
	// Mark() routes to root.freeList, not fused.freeList.
	if len(root.freeList) != 2 {
		t.Fatalf("root.freeList should have 2 entries after fused.Finish, got %d", len(root.freeList))
	}
}

// TestSweepClearsFreeList verifies that Sweep() frees queued tensors and empties root.freeList.
func TestSweepClearsFreeList(t *testing.T) {
	root := New(stdctx.Background())
	defer root.Finish()

	fused := root.Fused()
	_ = Float(fused, Shape{2, 2}, 1.0)
	_ = Float(fused, Shape{2, 2}, 2.0)

	fused.Finish()

	if len(root.freeList) != 2 {
		t.Fatalf("expected 2 entries in root.freeList before Sweep, got %d", len(root.freeList))
	}

	root.Sweep()

	if len(root.freeList) != 0 {
		t.Fatalf("root.freeList should be empty after Sweep, got %d", len(root.freeList))
	}
}

// TestNewComputationGraphNodeMetadata verifies that metadata passed to
// NewComputationGraphNode is stored and accessible in the backward function.
func TestNewComputationGraphNodeMetadata(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	probs := Float(ctx, Shape{2, 2}, 0.5)
	logits := Float(ctx, Shape{2, 2}, 1.0)
	result := Float(ctx, Shape{1}, 0.0)

	metadataSeen := false
	backward := BackwardFn(func(bctx *Context, node ComputationGraphNode) {
		m := node.Metadata()
		if len(m) == 1 && m[0] == probs {
			metadataSeen = true
		}
		logits.Grad().Accumulate(bctx, node.Grad().(*Tensor))
	})

	ctx.NewComputationGraphNode(
		result,
		OpCrossEntropy,
		backward,
		[]*Tensor{logits},
		[]*Tensor{},
		[]*Tensor{probs},
	)

	result.Backward(ctx)

	if !metadataSeen {
		t.Fatal("metadata tensor should be accessible in backward function")
	}
}

// TestMultipleOptions verifies that multiple functional options compose correctly.
func TestMultipleOptions(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	if !ctx.gradEnabled {
		t.Error("grad should be enabled")
	}
	if !ctx.backwardEnabled {
		t.Error("backward should be enabled with WithGrad(true)")
	}
}
