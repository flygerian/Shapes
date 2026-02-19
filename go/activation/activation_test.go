package activation

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
	
)

func approxEq(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) < tol
}

func TestTanh(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	// tanh(0) = 0, tanh(1) ≈ 0.7616
	a := shapes.Float(ctx, shapes.Shape{2}, 0.0)
	result := Tanh(ctx, a)

	got := result.Get(ctx, 0).Item().(float32)
	if !approxEq(got, 0.0, 1e-5) {
		t.Errorf("tanh(0) = %f, want 0.0", got)
	}
}

func TestTanhValues(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := shapes.Float(ctx, shapes.Shape{1}, 1.0)
	result := Tanh(ctx, a)

	got := result.Get(ctx, 0).Item().(float32)
	expected := float32(math.Tanh(1.0))
	if !approxEq(got, expected, 1e-5) {
		t.Errorf("tanh(1) = %f, want %f", got, expected)
	}
}

func TestTanhBackward(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// tanh'(x) = 1 - tanh(x)^2
	// At x=0: tanh(0)=0, tanh'(0) = 1 - 0 = 1
	x := shapes.Float(ctx, shapes.Shape{1}, 0.0)
	y := Tanh(ctx, x)

	y.Backward(ctx)

	got := x.Grad().Get(ctx, 0).Item().(float32)
	if !approxEq(got, 1.0, 1e-5) {
		t.Errorf("tanh'(0) = %f, want 1.0", got)
	}
}

func TestTanhBackwardNonZero(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// At x=1: tanh(1) ≈ 0.7616, tanh'(1) = 1 - 0.7616^2 ≈ 0.4200
	x := shapes.Float(ctx, shapes.Shape{1}, 1.0)
	y := Tanh(ctx, x)
	y.Backward(ctx)

	got := x.Grad().Get(ctx, 0).Item().(float32)
	tanhVal := float32(math.Tanh(1.0))
	expected := 1.0 - tanhVal*tanhVal
	if !approxEq(got, expected, 1e-4) {
		t.Errorf("tanh'(1) = %f, want %f", got, expected)
	}
}
