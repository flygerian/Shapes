package layer

import (
	"context"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestFlattenReturnsBatchPreserving2DOutput(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	flatten := Flatten(ctx)
	x := shapes.Float(ctx, shapes.Shape{2, 3, 4, 5}, 1.0)

	out := flatten.Forward(ctx, x)
	shape := out.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 60 {
		t.Fatalf("expected shape [2,60], got %v", shape)
	}

	if out.Op() != shapes.OpReshape {
		t.Fatalf("expected OpReshape, got %s", out.Op())
	}
}

func TestFlattenLeaves1DInputUnchanged(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	flatten := Flatten(ctx)
	x := shapes.Float(ctx, shapes.Shape{5}, 1.0)

	out := flatten.Forward(ctx, x)
	shape := out.Shape()
	if len(shape) != 1 || shape[0] != 5 {
		t.Fatalf("expected shape [5], got %v", shape)
	}
}

func TestFlattenBackwardRestoresOriginalShape(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	flatten := Flatten(ctx)
	x := shapes.Float(ctx, shapes.Shape{2, 3, 4}, 1.0)

	out := flatten.Forward(ctx, x)
	out.Backward(ctx)

	xGradShape := x.Grad().Shape()
	if len(xGradShape) != 3 || xGradShape[0] != 2 || xGradShape[1] != 3 || xGradShape[2] != 4 {
		t.Fatalf("expected x grad shape [2,3,4], got %v", xGradShape)
	}
}
