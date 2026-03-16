package shapes

import (
	"context"
	"math"
	"testing"
)

func TestReluFunction(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{3}, []float32{-1.0, 0.0, 2.0})
	out := Relu(ctx, x)

	want := []float32{0.0, 0.0, 2.0}
	got := out.Values().([]float32)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("relu[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestReluFunctionBackward(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{3}, []float32{-1.0, 0.0, 2.0})
	y := Relu(ctx, x)
	y.Backward(ctx)

	want := []float32{0.0, 0.0, 1.0}
	got := x.Grad().(Tensor).Values().([]float32)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("relu'[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestTanhFunction(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1}, []float32{1.0})
	out := Tanh(ctx, x)

	got := out.Values().([]float32)[0]
	want := float32(math.Tanh(1.0))
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("tanh = %f, want %f", got, want)
	}
}

func TestTanhFunctionBackward(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1}, []float32{1.0})
	y := Tanh(ctx, x)
	y.Backward(ctx)

	got := x.Grad().Get(ctx, 0).Item().(float32)
	tanhVal := float32(math.Tanh(1.0))
	want := 1.0 - tanhVal*tanhVal
	if math.Abs(float64(got-want)) > 1e-4 {
		t.Fatalf("tanh' = %f, want %f", got, want)
	}
}
