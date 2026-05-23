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
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	// tanh(0) = 0, tanh(1) ≈ 0.7616
	a := shapes.Float(ctx, shapes.Shape{2}, 0.0)
	result := Tanh().Forward(ctx, a)

	got := result.Get(ctx, 0).Item().(float32)
	if !approxEq(got, 0.0, 1e-5) {
		t.Errorf("tanh(0) = %f, want 0.0", got)
	}
}

func TestRelu(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	a := shapes.FromFloat32(ctx, shapes.Shape{3}, []float32{-1.0, 0.0, 2.0})
	result := Relu().Forward(ctx, a)

	want := []float32{0.0, 0.0, 2.0}
	got := result.Values().([]float32)
	for i := range want {
		if !approxEq(got[i], want[i], 1e-5) {
			t.Errorf("relu[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestReluBackward(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	x := shapes.FromFloat32(ctx, shapes.Shape{3}, []float32{-1.0, 0.0, 2.0})
	y := Relu().Forward(ctx, x)
	y.Backward(ctx)

	want := []float32{0.0, 0.0, 1.0}
	got := x.Grad().(shapes.Tensor).Values().([]float32)
	for i := range want {
		if !approxEq(got[i], want[i], 1e-5) {
			t.Errorf("relu'[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestReluBackwardAccumulatesMultipleUses(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	x := shapes.FromFloat32(ctx, shapes.Shape{3}, []float32{-1.0, 0.0, 2.0})
	y := Relu().Forward(ctx, x)
	z := y.Plus(ctx, y)
	loss := z.Sum(ctx, 0)
	loss.Backward(ctx)

	want := []float32{0.0, 0.0, 2.0}
	got := x.Grad().(shapes.Tensor).Values().([]float32)
	for i := range want {
		if !approxEq(got[i], want[i], 1e-5) {
			t.Errorf("relu accumulate[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestTanhValues(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	a := shapes.Float(ctx, shapes.Shape{1}, 1.0)
	result := Tanh().Forward(ctx, a)

	got := result.Get(ctx, 0).Item().(float32)
	expected := float32(math.Tanh(1.0))
	if !approxEq(got, expected, 1e-5) {
		t.Errorf("tanh(1) = %f, want %f", got, expected)
	}
}

func TestTanhBackward(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	// tanh'(x) = 1 - tanh(x)^2
	// At x=0: tanh(0)=0, tanh'(0) = 1 - 0 = 1
	x := shapes.Float(ctx, shapes.Shape{1}, 0.0)
	y := Tanh().Forward(ctx, x)

	y.Backward(ctx)

	got := x.Grad().Get(ctx, 0).Item().(float32)
	if !approxEq(got, 1.0, 1e-5) {
		t.Errorf("tanh'(0) = %f, want 1.0", got)
	}
}

func TestTanhBackwardNonZero(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	// At x=1: tanh(1) ≈ 0.7616, tanh'(1) = 1 - 0.7616^2 ≈ 0.4200
	x := shapes.Float(ctx, shapes.Shape{1}, 1.0)
	y := Tanh().Forward(ctx, x)
	y.Backward(ctx)

	got := x.Grad().Get(ctx, 0).Item().(float32)
	tanhVal := float32(math.Tanh(1.0))
	expected := 1.0 - tanhVal*tanhVal
	if !approxEq(got, expected, 1e-4) {
		t.Errorf("tanh'(1) = %f, want %f", got, expected)
	}
}
