package extract

import (
	"context"
	"testing"

	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/layer"
)

func TestParametersFromSingleDense(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense := layer.Dense(3, 2)
	x := shapes.Float(ctx, shapes.Shape{1, 3}, 1.0)

	o := dense(ctx, x)
	graph := o.Backward(ctx)

	params := Parameters(graph)
	// Dense has w and b => 2 parameters.
	if len(params) != 2 {
		t.Fatalf("expected 2 parameters, got %d", len(params))
	}

	wShape := shapes.ShapeOf(params[0])
	if len(wShape) != 2 || wShape[0] != 2 || wShape[1] != 3 {
		t.Errorf("expected w shape [2,3], got %v", wShape)
	}

	bShape := shapes.ShapeOf(params[1])
	if len(bShape) != 1 || bShape[0] != 2 {
		t.Errorf("expected b shape [2], got %v", bShape)
	}
}

func TestParametersFromMultipleDenseLayers(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	dense1 := layer.Dense(3, 4)
	dense2 := layer.Dense(4, 2)
	x := shapes.Float(ctx, shapes.Shape{1, 3}, 1.0)

	h := dense1(ctx, x)
	o := dense2(ctx, h)
	graph := o.Backward(ctx)

	params := Parameters(graph)
	// Two dense layers, each with w and b => 4 parameters.
	if len(params) != 4 {
		t.Fatalf("expected 4 parameters, got %d", len(params))
	}
}

func TestParametersEmptyWithNoLayers(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	a := shapes.Float(ctx, shapes.Shape{2}, 3.0)
	b := shapes.Float(ctx, shapes.Shape{2}, 5.0)

	c := a.Plus(ctx, b)
	graph := c.Backward(ctx)

	params := Parameters(graph)
	if len(params) != 0 {
		t.Fatalf("expected 0 parameters for plain ops, got %d", len(params))
	}
}
