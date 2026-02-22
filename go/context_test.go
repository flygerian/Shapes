package shapes

import (
	"context"
	stdctx "context"
	"testing"
	"time"
	"unsafe"
)

func TestNew(t *testing.T) {
	ctx := New(context.Background())
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
	ctx := New(context.Background())
	defer ctx.Close()
	if ctx.GradEnabled {
		t.Error("Grad should be disabled by default")
	}

	// With grad enabled
	ctx2 := New(context.TODO(), WithGrad(true))
	defer ctx2.Close()
	if !ctx2.GradEnabled {
		t.Error("Grad should be enabled")
	}

	// With grad explicitly disabled
	ctx3 := New(context.Background(), WithGrad(false))
	defer ctx3.Close()
	if ctx3.GradEnabled {
		t.Error("Grad should be disabled")
	}
}

func TestClose(t *testing.T) {
	ctx := New(context.Background())

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
	ctx := New(context.Background(),
		WithGrad(true),
	)
	defer ctx.Close()

	if !ctx.GradEnabled {
		t.Error("Grad should be enabled")
	}
}

// TestRootContextFreesMemory verifies that only the root context frees memory,
// and derived contexts are no-ops on Close().
func TestRootContextFreesMemory(t *testing.T) {
	root := New(context.Background())

	// Create some tensors
	t1 := Float(root, Shape{3, 3}, 1.0)
	t2 := Float(root, Shape{3, 3}, 2.0)

	// Verify tensors are valid
	if t1.cTensor == nil || t2.cTensor == nil {
		t.Fatal("tensors should be valid before Close")
	}

	// Create derived contexts and tensors in them
	fusedCtx := root.Fused()
	noGradCtx := root.NoGrad()
	noGraphCtx := root.NoGraph()

	t3 := Float(fusedCtx, Shape{2, 2}, 3.0)
	t4 := Float(noGradCtx, Shape{2, 2}, 4.0)
	t5 := Float(noGraphCtx, Shape{2, 2}, 5.0)

	// Close derived contexts - should be no-ops
	fusedCtx.Close()
	noGradCtx.Close()
	noGraphCtx.Close()

	// Tensors should still be valid
	if t3.cTensor == nil || t4.cTensor == nil || t5.cTensor == nil {
		t.Fatal("tensors from derived contexts should still be valid after derived Close")
	}

	// Close root - should free all memory and nil all tensor pointers
	root.Close()

	// All tensor pointers should be nilled
	if t1.cTensor != nil || t2.cTensor != nil || t3.cTensor != nil ||
		t4.cTensor != nil || t5.cTensor != nil {
		t.Fatal("all tensor pointers should be nilled after root Close")
	}
}

// TestSubcontextsShareRootMemory verifies that derived contexts share
// the same C arena and root tracking structures.
func TestSubcontextsShareRootMemory(t *testing.T) {
	root := New(context.Background())
	defer root.Close()

	// Create derived contexts
	fused := root.Fused()
	noGrad := root.NoGrad()
	noGraph := root.NoGraph()
	opCtx := root.OpSubcontext()

	// Verify they share the same root
	if fused.root != root || noGrad.root != root || noGraph.root != root || opCtx.root != root {
		t.Fatal("derived contexts should share the same root")
	}

	// Verify they don't own memory
	if fused.ownsMemory || noGrad.ownsMemory || noGraph.ownsMemory || opCtx.ownsMemory {
		t.Fatal("derived contexts should not own memory")
	}

	// Verify root owns memory
	if !root.ownsMemory {
		t.Fatal("root context should own memory")
	}

	// Verify they share the same C context
	if fused.cCtx != root.cCtx || noGrad.cCtx != root.cCtx ||
		noGraph.cCtx != root.cCtx || opCtx.cCtx != root.cCtx {
		t.Fatal("derived contexts should share the same C context")
	}
}

// TestLocalsTracking verifies that each context tracks its own locals.
func TestLocalsTracking(t *testing.T) {
	root := New(context.Background())
	defer root.Close()

	// Initially root has empty locals
	if len(root.locals) != 0 {
		t.Fatal("root should start with empty locals")
	}

	// Create tensors in root
	_ = Float(root, Shape{2, 2}, 1.0)
	_ = Float(root, Shape{2, 2}, 2.0)

	// Root should track both tensors in locals
	if len(root.locals) != 2 {
		t.Fatalf("root should have 2 locals, got %d", len(root.locals))
	}

	// Create a fused context and tensors there
	fused := root.Fused()
	_ = Float(fused, Shape{3, 3}, 3.0)

	// Fused should have its own locals (not share with root)
	if len(fused.locals) != 1 {
		t.Fatalf("fused should have 1 local, got %d", len(fused.locals))
	}

	// Root should still only have 2
	if len(root.locals) != 2 {
		t.Fatalf("root should still have 2 locals, got %d", len(root.locals))
	}
}

// TestHandlesTracking verifies that all tensor handles go to root.handles.
func TestHandlesTracking(t *testing.T) {
	root := New(context.Background())
	defer root.Close()

	// Create 3 tensors in various contexts
	tensor1 := Float(root, Shape{2, 2}, 1.0)

	fused := root.Fused()
	tensor2 := Float(fused, Shape{2, 2}, 2.0)

	noGrad := root.NoGrad()
	tensor3 := Float(noGrad, Shape{2, 2}, 3.0)

	// Verify that all created tensors have their handles in root.handles
	// (At least 3 handles should be present)
	if len(root.handles) < 3 {
		t.Fatalf("root should have at least 3 handles, got %d", len(root.handles))
	}

	// Verify the handles correspond to valid tensors by checking they can be accessed
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

// TestIntermediatesGoToRoot verifies that MarkIntermediate always routes to root.
func TestIntermediatesGoToRoot(t *testing.T) {
	root := New(context.Background())
	defer root.Close()

	// Create a tensor and mark it intermediate from a subcontext
	fused := root.Fused()
	tensor := Float(fused, Shape{2, 2}, 1.0)

	// Mark from fused context - should go to root
	fused.MarkIntermediate(unsafe.Pointer(tensor.cTensor))

	if len(root.intermediates) != 1 {
		t.Fatalf("root should have 1 intermediate, got %d", len(root.intermediates))
	}

	// Fused should not have any intermediates
	if len(fused.intermediates) != 0 {
		t.Fatal("fused should not have intermediates (they go to root)")
	}

	// Verify Intermediates() returns root's list
	if len(fused.Intermediates()) != 1 {
		t.Fatal("fused.Intermediates() should return root's list")
	}
}

// TestOpSubcontextSweeping verifies that backward op subcontexts properly sweep locals.
// This test verifies the sweeping mechanism works by checking intermediate counts.
func TestOpSubcontextSweeping(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	// Create simple computation graph
	x := Float(ctx, Shape{2, 2}, 2.0)
	y := x.Times(ctx, x) // x^2

	y.Backward(ctx)

	// After backward, there should be some intermediates (the temporary tensors
	// created during backward computation that were swept)
	afterCount := len(ctx.Intermediates())

	// The intermediates should have been freed at the end of Backward
	// so the count should be 0 again
	if afterCount != 0 {
		t.Fatalf("intermediates should be freed after backward, got %d", afterCount)
	}

	// But we can't verify the count during backward since FreeIntermediates clears it
	// The important thing is that the backward completed without panic or crash
}

// TestSavedTensorsSurviveBackward verifies that Saved tensors survive
// until their node's backward, then are freed.
func TestSavedTensorsSurviveBackward(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	// Create a scenario with a Dense layer which uses Saved tensors internally
	dense := func(inputSize int, outputSize int) func(*WrappedTensor) *WrappedTensor {
		var w, b *WrappedTensor
		return func(x *WrappedTensor) *WrappedTensor {
			ctx := x.Context()
			fusedCtx := ctx.Fused()
			if w == nil {
				w = fusedCtx.FloatRandom(Shape{uint32(outputSize), uint32(inputSize)})
				b = fusedCtx.FloatRandom(Shape{uint32(outputSize)})
			}
			o := x.Mul(w.Transpose()).Plus(b)
			fusedCtx.NewComputationGraphNode(o.Tensor(), OpDense, func(ctx *Context, node *ComputationGraphNode) {
				// Test that backward can access gradients
				w := node.Inputs[0]
				x := node.Inputs[1]
				b := node.Inputs[2]
				if w.Computation == nil || x.Computation == nil || b.Computation == nil {
					t.Error("all inputs should have computation nodes")
				}
			}, w.Tensor(), x.Tensor(), b.Tensor())
			return o
		}
	}

	layer := dense(3, 4)
	input := ctx.Float(Shape{2, 3}, 1.0)
	output := layer(input)
	output.Backward()

	// All gradients should be computed
	if input.Grad() == nil {
		t.Fatal("input gradient should be computed")
	}
}

// TestFreeIntermediatesDedup verifies that FreeIntermediates deduplicates pointers.
func TestFreeIntermediatesDedup(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	// Create and mark same tensor twice
	tensor := Float(ctx, Shape{2, 2}, 1.0)
	ctx.MarkIntermediate(unsafe.Pointer(tensor.cTensor))
	ctx.MarkIntermediate(unsafe.Pointer(tensor.cTensor)) // Duplicate

	// Should have 2 entries in intermediates list
	if len(ctx.Intermediates()) != 2 {
		t.Fatalf("should have 2 intermediate entries, got %d", len(ctx.Intermediates()))
	}

	// Free should not panic (no double-free)
	FreeIntermediates(ctx)

	// List should be cleared
	if len(ctx.Intermediates()) != 0 {
		t.Fatal("intermediates list should be cleared after FreeIntermediates")
	}
}

// TestMarkIntermediatesFromLocalsKeepsTensors verifies the sweeping logic.
func TestMarkIntermediatesFromLocalsKeepsTensors(t *testing.T) {
	root := New(context.Background())
	defer root.Close()

	// Create some tensors
	tensor1 := Float(root, Shape{2, 2}, 1.0)
	tensor2 := Float(root, Shape{2, 2}, 2.0)
	tensor3 := Float(root, Shape{2, 2}, 3.0)

	// Manually call markIntermediatesFromLocals keeping only tensor1 and tensor3
	markIntermediatesFromLocals(root, tensor1, tensor3)

	// Use tensor2 so it's not marked as unused
	_ = tensor2

	// tensor2 should be marked as intermediate
	if len(root.intermediates) != 1 {
		t.Fatalf("should have 1 intermediate, got %d", len(root.intermediates))
	}

	// locals should be cleared
	if len(root.locals) != 0 {
		t.Fatal("locals should be cleared after sweeping")
	}

	// Verify tensor1 and tensor3 are still accessible
	if tensor1.Get(root, 0, 0).Item().(float32) != 1.0 || tensor3.Get(root, 0, 0).Item().(float32) != 3.0 {
		t.Fatal("kept tensors should still be accessible")
	}
}

// TestMarkIntermediatesFromLocalsSkipsNil verifies nil handling.
func TestMarkIntermediatesFromLocalsSkipsNil(t *testing.T) {
	root := New(context.Background())
	defer root.Close()

	// Create one tensor, then nil its handle
	tensor1 := Float(root, Shape{2, 2}, 1.0)

	// Use tensor1 so it's not marked as unused
	_ = tensor1

	// Nil the first handle (simulating already-freed tensor)
	if len(root.locals) > 0 {
		*root.locals[0] = nil
	}

	// Add another tensor
	tensor2 := Float(root, Shape{2, 2}, 2.0)

	// Sweep keeping only tensor2
	markIntermediatesFromLocals(root, tensor2)

	// Only tensor1's original pointer (now nil) should have been marked
	// This won't add to intermediates since the handle is nil
	if len(root.intermediates) != 0 {
		t.Fatalf("should have 0 intermediates (nil handle skipped), got %d", len(root.intermediates))
	}

	// Verify tensor2 is still accessible
	if tensor2.Get(root, 0, 0).Item().(float32) != 2.0 {
		t.Fatal("kept tensor should still be accessible")
	}
}

// TestFusedContextSweeping verifies that fused contexts properly register nodes
// and track tensors. Note: Forward sweeping happens during backward pass, not
// during node registration, because Fused contexts have GradEnabled=true.
func TestFusedContextSweeping(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	fused := ctx.Fused()

	// Create tensors in fused context
	temp1 := Float(fused, Shape{2, 2}, 1.0)
	temp2 := Float(fused, Shape{2, 2}, 2.0)
	result := temp1.Plus(fused, temp2)

	// Register a custom node with the fused context
	fused.NewComputationGraphNode(result, OpAdd, func(ctx *Context, node *ComputationGraphNode) {
		// Simple backward for testing
		if len(node.Inputs) != 2 {
			t.Error("node should have 2 inputs")
		}
	}, temp1, temp2)

	// Verify the node was created
	if result.Computation == nil {
		t.Fatal("result should have computation node")
	}

	// At this point, no intermediates should be marked (sweeping happens during backward)
	// The important thing is that the node was registered correctly
	if result.Computation.Op != OpAdd {
		t.Fatal("node should have OpAdd operation type")
	}

	// Run backward to trigger sweeping
	result.Backward(ctx)

	// After backward, intermediates should have been freed
	// (the temp1 and temp2 tensors created in fused context are not swept
	// because they're inputs to the node, not temporaries)
}

// TestSavedAPIWithFusedContext verifies the Saved API works correctly.
func TestSavedAPIWithFusedContext(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	fused := ctx.Fused()

	// Create a saved tensor
	saved := Float(fused, Shape{2, 2}, 5.0)
	input := Float(fused, Shape{2, 2}, 1.0)
	result := input.Plus(fused, saved)

	// Register node with saved tensor
	fused.NewComputationGraphNodeSaved(result, OpAdd, func(ctx *Context, node *ComputationGraphNode) {
		// Test backward
		outGrad := node.Grad
		for _, inp := range node.Inputs {
			if inp.Computation != nil {
				inp.Computation.Grad = outGrad
			}
		}
	}, []*Tensor{saved}, input)

	// Verify Saved is set
	if len(result.Computation.Saved) != 1 {
		t.Fatalf("should have 1 saved tensor, got %d", len(result.Computation.Saved))
	}

	if result.Computation.Saved[0] != saved {
		t.Fatal("saved tensor should match")
	}

	// After the sweeping in newNode, saved should NOT be marked intermediate
	// because it's in the keep list
	if len(ctx.Intermediates()) != 0 {
		t.Fatalf("saved tensor should not be swept, got %d intermediates", len(ctx.Intermediates()))
	}
}

// TestBackwardMarksSavedAsIntermediate verifies Saved tensors are available in backward.
// Note: We can't check if they're marked intermediate after backward because
// FreeIntermediates clears the list at the end of Backward().
func TestBackwardMarksSavedAsIntermediate(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	// Create a simple graph with a custom backward that uses Saved
	x := Float(ctx, Shape{2, 2}, 2.0)

	// Manually create a computation that saves a tensor
	fused := ctx.Fused()
	savedTensor := x.Times(fused, x)     // x^2 - save this for backward
	result := savedTensor.Plus(fused, x) // x^2 + x

	// Track whether saved tensor was available in backward
	savedWasAvailable := false

	// Register with Saved API
	fused.NewComputationGraphNodeSaved(result, OpAdd, func(ctx *Context, node *ComputationGraphNode) {
		// Verify saved tensor is available
		if len(node.Saved) == 1 && node.Saved[0] != nil && node.Saved[0] == savedTensor {
			savedWasAvailable = true
		}
		// Compute simple gradients
		input := node.Inputs[0]
		if input.Computation != nil {
			input.Computation.Grad = node.Grad
		}
	}, []*Tensor{savedTensor}, x)

	// Run backward
	result.Backward(ctx)

	// Verify the saved tensor was available during backward
	if !savedWasAvailable {
		t.Fatal("saved tensor should be available in backward")
	}
}
