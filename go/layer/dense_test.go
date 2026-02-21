package layer

import (
	"context"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestDenseReturnsResult(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	dense := Dense(3, 2)
	x := ctx.Float(shapes.Shape{1, 3}, 1.0)

	o := dense(x)
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
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	dense := Dense(4, 2)
	x := ctx.Float(shapes.Shape{3, 4}, 1.0)

	o := dense(x)
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
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	dense := Dense(5, 3)
	x := ctx.Float(shapes.Shape{5}, 2.0)

	o := dense(x)
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
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	dense := Dense(3, 2)
	x := ctx.Float(shapes.Shape{1, 3}, 1.0)

	o := dense(x)

	if !o.RequiresGrad() {
		t.Error("expected graph node to always be attached")
	}
}

func TestDenseWithGradAttachesNode(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense := Dense(3, 2)
	x := ctx.Float(shapes.Shape{1, 3}, 1.0)

	o := dense(x)

	if !o.RequiresGrad() {
		t.Fatal("expected grad tracking when context has grad enabled")
	}

	if o.Tensor().Computation.Op != shapes.OpDense {
		t.Errorf("expected OpDense, got %s", o.Tensor().Computation.Op)
	}

	if len(o.Tensor().Computation.Inputs) != 3 {
		t.Fatalf("expected 3 inputs (w, x, b), got %d", len(o.Tensor().Computation.Inputs))
	}
}

func TestDenseBackward1D(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense := Dense(3, 2)
	x := ctx.Float(shapes.Shape{3}, 1.0)

	o := dense(x)
	o.Backward()

	// w is [2,3], x is [3], b is [2]
	wGrad := o.Tensor().Computation.Inputs[0].Grad()
	xGrad := o.Tensor().Computation.Inputs[1].Grad()
	bGrad := o.Tensor().Computation.Inputs[2].Grad()

	wGradShape := shapes.ShapeOf(wGrad)
	if len(wGradShape) != 2 || wGradShape[0] != 2 || wGradShape[1] != 3 {
		t.Fatalf("expected wGrad shape [2,3], got %v", wGradShape)
	}

	xGradShape := shapes.ShapeOf(xGrad)
	if len(xGradShape) != 1 || xGradShape[0] != 3 {
		t.Fatalf("expected xGrad shape [3], got %v", xGradShape)
	}

	bGradShape := shapes.ShapeOf(bGrad)
	if len(bGradShape) != 1 || bGradShape[0] != 2 {
		t.Fatalf("expected bGrad shape [2], got %v", bGradShape)
	}
}

func TestDenseBackward2D(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense := Dense(4, 2)
	x := ctx.Float(shapes.Shape{3, 4}, 1.0)

	o := dense(x)
	o.Backward()

	// w is [2,4], x is [3,4], b is [2]
	wGrad := o.Tensor().Computation.Inputs[0].Grad()
	xGrad := o.Tensor().Computation.Inputs[1].Grad()
	bGrad := o.Tensor().Computation.Inputs[2].Grad()

	wGradShape := shapes.ShapeOf(wGrad)
	if len(wGradShape) != 2 || wGradShape[0] != 2 || wGradShape[1] != 4 {
		t.Fatalf("expected wGrad shape [2,4], got %v", wGradShape)
	}

	xGradShape := shapes.ShapeOf(xGrad)
	if len(xGradShape) != 2 || xGradShape[0] != 3 || xGradShape[1] != 4 {
		t.Fatalf("expected xGrad shape [3,4], got %v", xGradShape)
	}

	bGradShape := shapes.ShapeOf(bGrad)
	if len(bGradShape) != 1 || bGradShape[0] != 2 {
		t.Fatalf("expected bGrad shape [2], got %v", bGradShape)
	}
}

func TestDenseBackward3D(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense := Dense(5, 3)
	x := ctx.Float(shapes.Shape{2, 4, 5}, 1.0)

	o := dense(x)
	o.Backward()

	// w is [3,5], x is [2,4,5], b is [3]
	wGrad := o.Tensor().Computation.Inputs[0].Grad()
	xGrad := o.Tensor().Computation.Inputs[1].Grad()
	bGrad := o.Tensor().Computation.Inputs[2].Grad()

	wGradShape := shapes.ShapeOf(wGrad)
	if len(wGradShape) != 2 || wGradShape[0] != 3 || wGradShape[1] != 5 {
		t.Fatalf("expected wGrad shape [3,5], got %v", wGradShape)
	}

	xGradShape := shapes.ShapeOf(xGrad)
	if len(xGradShape) != 3 || xGradShape[0] != 2 || xGradShape[1] != 4 || xGradShape[2] != 5 {
		t.Fatalf("expected xGrad shape [2,4,5], got %v", xGradShape)
	}

	bGradShape := shapes.ShapeOf(bGrad)
	if len(bGradShape) != 1 || bGradShape[0] != 3 {
		t.Fatalf("expected bGrad shape [3], got %v", bGradShape)
	}
}

func TestDenseBackward4D(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense := Dense(6, 4)
	x := ctx.Float(shapes.Shape{2, 3, 5, 6}, 1.0)

	o := dense(x)
	o.Backward()

	// w is [4,6], x is [2,3,5,6], b is [4]
	wGrad := o.Tensor().Computation.Inputs[0].Grad()
	xGrad := o.Tensor().Computation.Inputs[1].Grad()
	bGrad := o.Tensor().Computation.Inputs[2].Grad()

	wGradShape := shapes.ShapeOf(wGrad)
	if len(wGradShape) != 2 || wGradShape[0] != 4 || wGradShape[1] != 6 {
		t.Fatalf("expected wGrad shape [4,6], got %v", wGradShape)
	}

	xGradShape := shapes.ShapeOf(xGrad)
	if len(xGradShape) != 4 || xGradShape[0] != 2 || xGradShape[1] != 3 || xGradShape[2] != 5 || xGradShape[3] != 6 {
		t.Fatalf("expected xGrad shape [2,3,5,6], got %v", xGradShape)
	}

	bGradShape := shapes.ShapeOf(bGrad)
	if len(bGradShape) != 1 || bGradShape[0] != 4 {
		t.Fatalf("expected bGrad shape [4], got %v", bGradShape)
	}
}

func TestDenseInternalOpsHaveNoBackward(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense := Dense(3, 2)
	x := ctx.Float(shapes.Shape{1, 3}, 1.0)

	o := dense(x)

	w := o.Tensor().Computation.Inputs[0]
	b := o.Tensor().Computation.Inputs[2]

	if w.Computation != nil && w.Computation.Backward != nil {
		t.Error("w should not have its own backward pass")
	}
	if b.Computation != nil && b.Computation.Backward != nil {
		t.Error("b should not have its own backward pass")
	}

	if o.Tensor().Computation.Backward == nil {
		t.Error("Dense output should have a backward function")
	}
}
