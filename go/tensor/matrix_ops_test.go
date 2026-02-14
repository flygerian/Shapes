package tensor

import (
	"context"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestMul(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	// 2x3 of 1.0 * 3x2 of 1.0 → 2x2 of 3.0
	a := Float(ctx, Shape{2, 3}, 1.0)
	b := Float(ctx, Shape{3, 2}, 1.0)

	result := a.Mul(ctx, b)

	for i := range uint32(2) {
		for j := range uint32(2) {
			got, err := result.GetF32(i, j)
			if err != nil {
				t.Fatalf("GetF32(%d,%d): %v", i, j, err)
			}
			if got < 2.99 || got > 3.01 {
				t.Errorf("Mul[%d,%d] = %f, want 3.0", i, j, got)
			}
		}
	}
}

func TestMulInnerDimMismatch(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{2, 3}, 1.0)
	b := Float(ctx, Shape{2, 2}, 1.0)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for inner dim mismatch, got nil")
		}
	}()
	a.Mul(ctx, b)
}

func TestDot(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	// [2.0, 2.0, 2.0] · [3.0, 3.0, 3.0] = 18.0
	a := Float(ctx, Shape{3}, 2.0)
	b := Float(ctx, Shape{3}, 3.0)

	result := a.Dot(ctx, b)

	got, err := result.GetF32(0)
	if err != nil {
		t.Fatalf("GetF32(0): %v", err)
	}
	if got < 17.99 || got > 18.01 {
		t.Errorf("Dot = %f, want 18.0", got)
	}
}

func TestDotDimMismatch(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Float(ctx, Shape{3}, 1.0)
	b := Float(ctx, Shape{2, 3}, 1.0)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for non-1D tensors, got nil")
		}
	}()
	a.Dot(ctx, b)
}
