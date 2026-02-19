package tensor

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestPow(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{2, 2}, 3.0)
	result := a.Pow(ctx, 2.0)

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(ctx, i, j).Item().(float32)
			if math.Abs(float64(got)-9.0) > 1e-4 {
				t.Errorf("Pow[%d,%d] = %f, want 9.0", i, j, got)
			}
		}
	}
}

func TestPowFractional(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{2}, 4.0)
	result := a.Pow(ctx, 0.5)

	for i := range uint32(2) {
		got := result.Get(ctx, i).Item().(float32)
		if math.Abs(float64(got)-2.0) > 1e-4 {
			t.Errorf("Pow(0.5)[%d] = %f, want 2.0", i, got)
		}
	}
}

func TestPowBackwardSquare(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// x=3, x^2=9, d(x^2)/dx = 2x = 6
	x := Float(ctx, Shape{1}, 3.0)
	y := x.Pow(ctx, 2.0)
	y.Backward(ctx)

	got := x.Grad().Get(ctx, 0).Item().(float32)
	if !approxEq(got, 6.0, 1e-4) {
		t.Errorf("d(x^2)/dx at x=3 = %f, want 6.0", got)
	}
}

func TestPowBackwardCube(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// x=2, x^3=8, d(x^3)/dx = 3x^2 = 12
	x := Float(ctx, Shape{1}, 2.0)
	y := x.Pow(ctx, 3.0)
	y.Backward(ctx)

	got := x.Grad().Get(ctx, 0).Item().(float32)
	if !approxEq(got, 12.0, 1e-4) {
		t.Errorf("d(x^3)/dx at x=2 = %f, want 12.0", got)
	}
}

func TestPowBackwardSqrt(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// x=4, x^0.5=2, d(x^0.5)/dx = 0.5 * x^(-0.5) = 0.5/2 = 0.25
	x := Float(ctx, Shape{1}, 4.0)
	y := x.Pow(ctx, 0.5)
	y.Backward(ctx)

	got := x.Grad().Get(ctx, 0).Item().(float32)
	if !approxEq(got, 0.25, 1e-4) {
		t.Errorf("d(x^0.5)/dx at x=4 = %f, want 0.25", got)
	}
}

func TestPowBackwardMultiElement(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// x=[1,2,3], x^2=[1,4,9], d(x^2)/dx = 2x = [2,4,6]
	x := FromFloat32(ctx, Shape{3}, []float32{1.0, 2.0, 3.0})
	y := x.Pow(ctx, 2.0)
	y.Backward(ctx)

	expected := []float32{2.0, 4.0, 6.0}
	for i, want := range expected {
		got := x.Grad().Get(ctx, uint32(i)).Item().(float32)
		if !approxEq(got, want, 1e-4) {
			t.Errorf("d(x^2)/dx[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestPowBackwardIdentity(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// x^1, d(x^1)/dx = 1
	x := Float(ctx, Shape{2}, 5.0)
	y := x.Pow(ctx, 1.0)
	y.Backward(ctx)

	for i := range uint32(2) {
		got := x.Grad().Get(ctx, i).Item().(float32)
		if !approxEq(got, 1.0, 1e-4) {
			t.Errorf("d(x^1)/dx[%d] = %f, want 1.0", i, got)
		}
	}
}

func TestExp(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{2, 2}, 1.0)
	result := a.Exp(ctx)

	want := float32(math.E)
	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(ctx, i, j).Item().(float32)
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Errorf("Exp[%d,%d] = %f, want %f", i, j, got, want)
			}
		}
	}
}

func TestNegate(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := FromFloat32(ctx, Shape{3}, []float32{1.0, -2.0, 3.0})
	result := a.Negate(ctx)

	expected := []float32{-1.0, 2.0, -3.0}
	for i, want := range expected {
		got := result.Get(ctx, uint32(i)).Item().(float32)
		if !approxEq(got, want, 1e-5) {
			t.Errorf("Negate[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestNegateZeros(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Zeros(ctx, Shape{2, 2})
	result := a.Negate(ctx)

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(ctx, i, j).Item().(float32)
			if !approxEq(got, 0.0, 1e-5) {
				t.Errorf("Negate(0)[%d,%d] = %f, want 0.0", i, j, got)
			}
		}
	}
}

func TestNegateBackward(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// -x, d(-x)/dx = -1
	x := FromFloat32(ctx, Shape{3}, []float32{1.0, -2.0, 3.0})
	y := x.Negate(ctx)
	y.Backward(ctx)

	for i := range uint32(3) {
		got := x.Grad().Get(ctx, i).Item().(float32)
		if !approxEq(got, -1.0, 1e-5) {
			t.Errorf("grad[%d] = %f, want -1.0", i, got)
		}
	}
}

func TestExpZero(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{3}, 0.0)
	result := a.Exp(ctx)

	for i := range uint32(3) {
		got := result.Get(ctx, i).Item().(float32)
		if math.Abs(float64(got)-1.0) > 1e-4 {
			t.Errorf("Exp(0)[%d] = %f, want 1.0", i, got)
		}
	}
}
