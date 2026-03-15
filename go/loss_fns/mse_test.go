package loss_fns

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func approxEq(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) < tol
}

func TestMse(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	yGround := shapes.FromFloat32(ctx, shapes.Shape{4}, []float32{1.0, -1.0, -1.0, 1.0})
	yPred := shapes.FromFloat32(ctx, shapes.Shape{4}, []float32{0.5, -0.5, -0.8, 0.9})

	loss := Mse(ctx, yGround, yPred)(nil)

	// Sum of squared errors: 0.25 + 0.25 + 0.04 + 0.01 = 0.55
	got := loss.Get(ctx, 0).Item().(float32)
	if !approxEq(got, 0.55, 1e-4) {
		t.Errorf("Mse = %f, want 0.55", got)
	}
}

func TestMsePerfectPrediction(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	y := shapes.FromFloat32(ctx, shapes.Shape{3}, []float32{1.0, 2.0, 3.0})

	loss := Mse(ctx, y, y)(nil)

	got := loss.Get(ctx, 0).Item().(float32)
	if !approxEq(got, 0.0, 1e-6) {
		t.Errorf("Mse = %f, want 0.0", got)
	}
}

func TestMse2D(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	yGround := shapes.FromFloat32(ctx, shapes.Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	yPred := shapes.FromFloat32(ctx, shapes.Shape{2, 3}, []float32{2, 2, 2, 2, 2, 2})

	loss := Mse(ctx, yGround, yPred)(nil)

	// Per-element squared errors: 1, 0, 1, 4, 9, 16. Sum = 31.0
	shape := loss.Shape()

	var got float32
	if len(shape) == 2 {
		got = loss.Get(ctx, 0, 0).Item().(float32)
	} else {
		got = loss.Get(ctx, 0).Item().(float32)
	}
	if !approxEq(got, 31.0, 1e-4) {
		t.Errorf("Mse = %f, want 31.0", got)
	}
}

func TestMseBackward(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	// Scalar case: pred=1.5, ground=1.0 => loss = (0.5)^2 = 0.25
	// d(loss)/d(pred) = 2*(pred-ground) = 1.0
	yGround := shapes.Float(ctx, shapes.Shape{1}, 1.0)
	yPred := shapes.Float(ctx, shapes.Shape{1}, 1.5)

	loss := Mse(ctx, yGround, yPred)(nil)
	loss.Backward(ctx)

	predGrad := yPred.Grad()
	if predGrad == nil {
		t.Fatal("expected gradient on yPred")
	}

	got := predGrad.Get(ctx, 0).Item().(float32)
	if !approxEq(got, 1.0, 1e-4) {
		t.Errorf("d(loss)/d(yPred) = %f, want 1.0", got)
	}
}
