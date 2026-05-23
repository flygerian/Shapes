package layer

import (
	"context"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestDenseReturnsResult(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := Dense(ctx, 3, 2)
	x := shapes.Float(ctx, shapes.Shape{1, 3}, 1.0)

	o := dense.Forward(ctx, x)
	if o == nil {
		t.Fatal("Dense returned nil")
	}

	// x @ wᵀ + b: [1,3] @ [3,2] = [1,2]
	shape := o.Shape()
	if len(shape) != 2 || shape[0] != 1 || shape[1] != 2 {
		t.Fatalf("expected shape [1,2], got %v", shape)
	}
}

func TestDenseBatchInput(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := Dense(ctx, 4, 2)
	x := shapes.Float(ctx, shapes.Shape{3, 4}, 1.0)

	o := dense.Forward(ctx, x)
	if o == nil {
		t.Fatal("Dense returned nil for batched input")
	}

	// x @ wᵀ + b: [3,4] @ [4,2] = [3,2]
	shape := o.Shape()
	if len(shape) != 2 || shape[0] != 3 || shape[1] != 2 {
		t.Fatalf("expected shape [3,2], got %v", shape)
	}
}

func TestDense1DInput(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := Dense(ctx, 5, 3)
	x := shapes.Float(ctx, shapes.Shape{5}, 2.0)

	o := dense.Forward(ctx, x)
	if o == nil {
		t.Fatal("Dense returned nil for 1D input")
	}

	// x unsqueezed to [1,5], x @ wᵀ = [1,5] @ [5,3] = [1,3]
	// squeezed back to [3]
	shape := o.Shape()
	if len(shape) != 1 || shape[0] != 3 {
		t.Fatalf("expected shape [3], got %v", shape)
	}
}

func TestDenseAlwaysAttachesGraph(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := Dense(ctx, 3, 2)
	x := shapes.Float(ctx, shapes.Shape{1, 3}, 1.0)

	o := dense.Forward(ctx, x)

	if !o.(interface{ RequiresGrad() bool }).RequiresGrad() {
		t.Error("expected graph node to always be attached")
	}
}

func TestDenseWithGradAttachesNode(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := Dense(ctx, 3, 2)
	x := shapes.Float(ctx, shapes.Shape{1, 3}, 1.0)

	o := dense.Forward(ctx, x)

	if !o.(interface{ RequiresGrad() bool }).RequiresGrad() {
		t.Fatal("expected grad tracking when context has grad enabled")
	}

	if o.Op() != shapes.OpDense {
		t.Errorf("expected OpDense, got %s", o.Op())
	}

	if len(o.Inputs()) != 1 {
		t.Fatalf("expected 1 input (x), got %d", len(o.Inputs()))
	}
}

func TestDenseBackward1D(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := Dense(ctx, 3, 2)
	x := shapes.Float(ctx, shapes.Shape{3}, 1.0)

	o := dense.Forward(ctx, x)
	o.Backward(ctx)

	// w is [2,3], x is [3], b is [2]
	hidden := o.HiddenState()
	wGrad := hidden[0].Grad()
	xGrad := o.Inputs()[0].Grad()
	bGrad := hidden[1].Grad()

	wGradShape := wGrad.Shape()
	if len(wGradShape) != 2 || wGradShape[0] != 2 || wGradShape[1] != 3 {
		t.Fatalf("expected wGrad shape [2,3], got %v", wGradShape)
	}

	xGradShape := xGrad.Shape()
	if len(xGradShape) != 1 || xGradShape[0] != 3 {
		t.Fatalf("expected xGrad shape [3], got %v", xGradShape)
	}

	bGradShape := bGrad.Shape()
	if len(bGradShape) != 1 || bGradShape[0] != 2 {
		t.Fatalf("expected bGrad shape [2], got %v", bGradShape)
	}
}

func TestDenseBackward2D(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := Dense(ctx, 4, 2)
	x := shapes.Float(ctx, shapes.Shape{3, 4}, 1.0)

	o := dense.Forward(ctx, x)
	o.Backward(ctx)

	// w is [2,4], x is [3,4], b is [2]
	hidden := o.HiddenState()
	wGrad := hidden[0].Grad()
	xGrad := o.Inputs()[0].Grad()
	bGrad := hidden[1].Grad()

	wGradShape := wGrad.Shape()
	if len(wGradShape) != 2 || wGradShape[0] != 2 || wGradShape[1] != 4 {
		t.Fatalf("expected wGrad shape [2,4], got %v", wGradShape)
	}

	xGradShape := xGrad.Shape()
	if len(xGradShape) != 2 || xGradShape[0] != 3 || xGradShape[1] != 4 {
		t.Fatalf("expected xGrad shape [3,4], got %v", xGradShape)
	}

	bGradShape := bGrad.Shape()
	if len(bGradShape) != 1 || bGradShape[0] != 2 {
		t.Fatalf("expected bGrad shape [2], got %v", bGradShape)
	}
}

func TestDenseBackward3D(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := Dense(ctx, 5, 3)
	x := shapes.Float(ctx, shapes.Shape{2, 4, 5}, 1.0)

	o := dense.Forward(ctx, x)
	o.Backward(ctx)

	// w is [3,5], x is [2,4,5], b is [3]
	hidden := o.HiddenState()
	wGrad := hidden[0].Grad()
	xGrad := o.Inputs()[0].Grad()
	bGrad := hidden[1].Grad()

	wGradShape := wGrad.Shape()
	if len(wGradShape) != 2 || wGradShape[0] != 3 || wGradShape[1] != 5 {
		t.Fatalf("expected wGrad shape [3,5], got %v", wGradShape)
	}

	xGradShape := xGrad.Shape()
	if len(xGradShape) != 3 || xGradShape[0] != 2 || xGradShape[1] != 4 || xGradShape[2] != 5 {
		t.Fatalf("expected xGrad shape [2,4,5], got %v", xGradShape)
	}

	bGradShape := bGrad.Shape()
	if len(bGradShape) != 1 || bGradShape[0] != 3 {
		t.Fatalf("expected bGrad shape [3], got %v", bGradShape)
	}
}

func TestDenseBackward4D(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := Dense(ctx, 6, 4)
	x := shapes.Float(ctx, shapes.Shape{2, 3, 5, 6}, 1.0)

	o := dense.Forward(ctx, x)
	o.Backward(ctx)

	// w is [4,6], x is [2,3,5,6], b is [4]
	hidden := o.HiddenState()
	wGrad := hidden[0].Grad()
	xGrad := o.Inputs()[0].Grad()
	bGrad := hidden[1].Grad()

	wGradShape := wGrad.Shape()
	if len(wGradShape) != 2 || wGradShape[0] != 4 || wGradShape[1] != 6 {
		t.Fatalf("expected wGrad shape [4,6], got %v", wGradShape)
	}

	xGradShape := xGrad.Shape()
	if len(xGradShape) != 4 || xGradShape[0] != 2 || xGradShape[1] != 3 || xGradShape[2] != 5 || xGradShape[3] != 6 {
		t.Fatalf("expected xGrad shape [2,3,5,6], got %v", xGradShape)
	}

	bGradShape := bGrad.Shape()
	if len(bGradShape) != 1 || bGradShape[0] != 4 {
		t.Fatalf("expected bGrad shape [4], got %v", bGradShape)
	}
}

func TestDenseBackwardUsesTensorContextOverCallerContext(t *testing.T) {
	cpuCtx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer cpuCtx.Finish()
	cudaCtx := shapes.New(context.Background(), shapes.WithGrad(true), shapes.WithCuda())
	defer cudaCtx.Finish()

	dense := Dense(cudaCtx, 4, 2)
	x := shapes.Float(cudaCtx, shapes.Shape{3, 4}, 1.0)

	o := dense.Forward(cudaCtx, x)
	o.Backward(cpuCtx)

	hidden := o.HiddenState()
	wGradShape := hidden[0].Grad().Shape()
	if len(wGradShape) != 2 || wGradShape[0] != 2 || wGradShape[1] != 4 {
		t.Fatalf("expected wGrad shape [2,4], got %v", wGradShape)
	}

	xGradShape := o.Inputs()[0].Grad().Shape()
	if len(xGradShape) != 2 || xGradShape[0] != 3 || xGradShape[1] != 4 {
		t.Fatalf("expected xGrad shape [3,4], got %v", xGradShape)
	}

	bGradShape := hidden[1].Grad().Shape()
	if len(bGradShape) != 1 || bGradShape[0] != 2 {
		t.Fatalf("expected bGrad shape [2], got %v", bGradShape)
	}
}

func TestDenseInternalOpsHaveNoBackward(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := Dense(ctx, 3, 2)
	x := shapes.Float(ctx, shapes.Shape{1, 3}, 1.0)

	o := dense.Forward(ctx, x)

	w := o.HiddenState()[0]
	b := o.HiddenState()[1]
	if w.Op() != shapes.OpNone {
		t.Errorf("expected w to be a leaf tensor, got op %s", w.Op())
	}
	if b.Op() != shapes.OpNone {
		t.Errorf("expected b to be a leaf tensor, got op %s", b.Op())
	}
	if o.Op() != shapes.OpDense {
		t.Error("Dense output should keep OpDense")
	}
}

func TestDensePanicsOnMismatchedInputSize(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	dense := Dense(ctx, 3, 2)
	x := shapes.Float(ctx, shapes.Shape{1, 5}, 1.0)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for mismatched input size, got nil")
		}
	}()

	dense.Forward(ctx, x)
}
