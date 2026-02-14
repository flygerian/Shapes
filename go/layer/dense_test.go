package layer

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/tensor"
)

func approxEq(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) < tol
}

func TestDenseReturnsResult(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	dense := Dense(3, 2)
	// 1x3 input (single sample, 3 features)
	x := tensor.Float(ctx, tensor.Shape{1, 3}, 1.0)

	o := dense(ctx, x)
	if o == nil {
		t.Fatal("Dense returned nil")
	}

	// w is [2,3] filled with 0.001, xᵀ is [3,1]
	// w @ xᵀ = [2,1], each element = 3 * 0.001 = 0.003
	// + b[2,1](0.001) = 0.004
	shape := tensor.ShapeOf(o)
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 1 {
		t.Fatalf("expected shape [2,1], got %v", shape)
	}

	for i := range uint32(2) {
		got, err := o.GetF32(i, 0)
		if err != nil {
			t.Fatalf("GetF32(%d,0): %v", i, err)
		}
		if !approxEq(got, 0.004, 1e-4) {
			t.Errorf("Dense output[%d,0] = %f, want 0.004", i, got)
		}
	}
}

func TestDenseBatchInput(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	dense := Dense(4, 2)
	// 3x4 input: batch of 3, each with 4 features
	x := tensor.Float(ctx, tensor.Shape{3, 4}, 1.0)

	o := dense(ctx, x)
	if o == nil {
		t.Fatal("Dense returned nil for batched input")
	}

	// w is [2,4], xᵀ is [4,3]
	// w @ xᵀ = [2,3], each element = 4 * 0.001 = 0.004
	// + b[2,1](0.001) broadcast → each element = 0.005
	shape := tensor.ShapeOf(o)
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("expected shape [2,3], got %v", shape)
	}

	for i := range uint32(2) {
		for j := range uint32(3) {
			got, err := o.GetF32(i, j)
			if err != nil {
				t.Fatalf("GetF32(%d,%d): %v", i, j, err)
			}
			if !approxEq(got, 0.005, 1e-4) {
				t.Errorf("Dense output[%d,%d] = %f, want 0.005", i, j, got)
			}
		}
	}
}

func TestDense1DInput(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	dense := Dense(5, 3)
	x := tensor.Float(ctx, tensor.Shape{5}, 2.0)

	o := dense(ctx, x)
	if o == nil {
		t.Fatal("Dense returned nil for 1D input")
	}

	// w is [3,5] filled with 0.001, x is [5] filled with 2.0
	// x unsqueezed to [1,5], transposed to [5,1]
	// w @ xᵀ = [3,1], each = 5*0.001*2.0 = 0.01, + b(0.001) = 0.011
	// squeezed back to [3]
	shape := tensor.ShapeOf(o)
	if len(shape) != 1 || shape[0] != 3 {
		t.Fatalf("expected shape [3], got %v", shape)
	}

	for i := range uint32(3) {
		got, err := o.GetF32(i)
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if !approxEq(got, 0.011, 1e-4) {
			t.Errorf("Dense output[%d] = %f, want 0.011", i, got)
		}
	}
}

func TestDenseNoGradNoGraph(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	dense := Dense(3, 2)
	x := tensor.Float(ctx, tensor.Shape{1, 3}, 1.0)

	o := dense(ctx, x)

	if o.RequiresGrad() {
		t.Error("expected no grad tracking when context has grad disabled")
	}
}

func TestDenseWithGradAttachesNode(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense := Dense(3, 2)
	x := tensor.Float(ctx, tensor.Shape{1, 3}, 1.0)

	o := dense(ctx, x)

	if !o.RequiresGrad() {
		t.Fatal("expected grad tracking when context has grad enabled")
	}

	if o.Computation.Op != tensor.OpDense {
		t.Errorf("expected OpDense, got %s", o.Computation.Op)
	}

	// The node should have 3 inputs: w, x, b
	if len(o.Computation.Inputs) != 3 {
		t.Fatalf("expected 3 inputs (w, x, b), got %d", len(o.Computation.Inputs))
	}
}

func TestDenseInternalOpsHaveNoBackward(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense := Dense(3, 2)
	x := tensor.Float(ctx, tensor.Shape{1, 3}, 1.0)

	o := dense(ctx, x)

	// The internal ops (Mul, Plus) should NOT have their own backward
	// passes because Dense uses NoGrad for those. Only the Dense node
	// itself should have a backward function.
	w := o.Computation.Inputs[0]
	b := o.Computation.Inputs[2]

	if w.Computation != nil && w.Computation.Backward != nil {
		t.Error("w should not have its own backward pass")
	}
	if b.Computation != nil && b.Computation.Backward != nil {
		t.Error("b should not have its own backward pass")
	}

	if o.Computation.Backward == nil {
		t.Error("Dense output should have a backward function")
	}
}
