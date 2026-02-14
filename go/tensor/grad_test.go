package tensor

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func approxEq(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) < tol
}

func TestBackwardAdd(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// a + b => da = 1, db = 1
	a := Float(ctx, Shape{2}, 3.0)
	b := Float(ctx, Shape{2}, 5.0)

	c := a.Plus(b)
	c.Backward()

	for i := range uint32(2) {
		ga, err := a.Grad().GetF32(i)
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if !approxEq(ga, 1.0, 1e-5) {
			t.Errorf("grad_a[%d] = %f, want 1.0", i, ga)
		}

		gb, err := b.Grad().GetF32(i)
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if !approxEq(gb, 1.0, 1e-5) {
			t.Errorf("grad_b[%d] = %f, want 1.0", i, gb)
		}
	}
}

func TestBackwardMultiply(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// c = a * b => da = b, db = a
	a := Float(ctx, Shape{2}, 3.0)
	b := Float(ctx, Shape{2}, 5.0)

	c := a.Times(b)
	c.Backward()

	for i := range uint32(2) {
		ga, err := a.Grad().GetF32(i)
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if !approxEq(ga, 5.0, 1e-5) {
			t.Errorf("grad_a[%d] = %f, want 5.0 (value of b)", i, ga)
		}

		gb, err := b.Grad().GetF32(i)
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if !approxEq(gb, 3.0, 1e-5) {
			t.Errorf("grad_b[%d] = %f, want 3.0 (value of a)", i, gb)
		}
	}
}

func TestBackwardSubtract(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// c = a - b => da = 1, db = -1
	a := Float(ctx, Shape{2}, 7.0)
	b := Float(ctx, Shape{2}, 2.0)

	c := a.Minus(b)
	c.Backward()

	for i := range uint32(2) {
		ga, err := a.Grad().GetF32(i)
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if !approxEq(ga, 1.0, 1e-5) {
			t.Errorf("grad_a[%d] = %f, want 1.0", i, ga)
		}

		gb, err := b.Grad().GetF32(i)
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if !approxEq(gb, -1.0, 1e-5) {
			t.Errorf("grad_b[%d] = %f, want -1.0", i, gb)
		}
	}
}

func TestBackwardChain(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// Reproduce Karpathy's micrograd example:
	// x1=2, x2=0, w1=-3, w2=1, b=6.8813735870195432
	// x1w1 = x1*w1 = -6
	// x2w2 = x2*w2 = 0
	// x1w1x2w2 = x1w1 + x2w2 = -6
	// n = x1w1x2w2 + b = 0.8813735870195432
	//
	// Backward from n:
	// dn/dn = 1
	// dn/d(x1w1x2w2) = 1, dn/db = 1
	// dn/d(x1w1) = 1, dn/d(x2w2) = 1
	// dn/dx1 = w1 = -3, dn/dw1 = x1 = 2
	// dn/dx2 = w2 = 1, dn/dw2 = x2 = 0

	x1 := Float(ctx, Shape{1}, 2.0)
	x2 := Float(ctx, Shape{1}, 0.0)
	w1 := Float(ctx, Shape{1}, -3.0)
	w2 := Float(ctx, Shape{1}, 1.0)
	b := Float(ctx, Shape{1}, 6.8813735870195432)

	x1w1 := x1.Times(w1)
	x2w2 := x2.Times(w2)
	x1w1x2w2 := x1w1.Plus(x2w2)
	n := x1w1x2w2.Plus(b)

	n.Backward()

	tests := []struct {
		name string
		t    *Tensor
		want float32
	}{
		{"dx1", x1, -3.0},
		{"dw1", w1, 2.0},
		{"dx2", x2, 1.0},
		{"dw2", w2, 0.0},
		{"db", b, 1.0},
	}

	for _, tc := range tests {
		got, err := tc.t.Grad().GetF32(0)
		if err != nil {
			t.Fatalf("%s: GetF32: %v", tc.name, err)
		}
		if !approxEq(got, tc.want, 1e-5) {
			t.Errorf("%s = %f, want %f", tc.name, got, tc.want)
		}
	}
}

func TestBackwardNoGradNoPanic(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{2}, 3.0)
	b := Float(ctx, Shape{2}, 5.0)

	// No grad enabled, ops should not build graph.
	c := a.Plus(b)
	if c.RequiresGrad() {
		t.Error("expected no grad tracking when grad is disabled")
	}
}

func TestBackwardPanicsNoGraph(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{2}, 3.0)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic when calling Backward on tensor with no graph")
		}
	}()

	a.Backward()
}

func TestLeafTensorHasGrad(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	a := Float(ctx, Shape{2}, 3.0)

	if !a.RequiresGrad() {
		t.Fatal("leaf tensor should have grad node when grad is enabled")
	}

	// Grad should be zeros by default.
	for i := range uint32(2) {
		got, err := a.Grad().GetF32(i)
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if !approxEq(got, 0.0, 1e-5) {
			t.Errorf("leaf grad[%d] = %f, want 0.0", i, got)
		}
	}
}

func TestLeafBackwardNoOp(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// Calling Backward on a leaf tensor should not panic.
	a := Float(ctx, Shape{1}, 5.0)
	a.Backward()

	got, err := a.Grad().GetF32(0)
	if err != nil {
		t.Fatalf("GetF32: %v", err)
	}
	if !approxEq(got, 1.0, 1e-5) {
		t.Errorf("leaf backward grad = %f, want 1.0", got)
	}
}

func TestGradPanicsNoNode(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{2}, 3.0)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic when calling Grad on tensor with no node")
		}
	}()
	a.Grad()
}
