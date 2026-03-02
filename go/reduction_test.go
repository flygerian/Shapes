package shapes

import (
	"context"
	"testing"
)

func TestSumAlongDim0(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 2x3 tensor filled with 2s, sum along dim 0 → 1x3 of 4s
	a := Int(ctx, Shape{2, 3}, 2)

	result := a.Sum(ctx, 0)

	for j := range uint32(3) {
		got := result.Get(ctx, 0, j).Item().(int8)
		if got != 4 {
			t.Errorf("Sum dim0[0,%d] = %d, want 4", j, got)
		}
	}
}

func TestSumAlongDim1(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 2x3 tensor filled with 3s, sum along dim 1 → 2x1 of 9s
	a := Int(ctx, Shape{2, 3}, 3)

	result := a.Sum(ctx, 1)

	for i := range uint32(2) {
		got := result.Get(ctx, i, 0).Item().(int8)
		if got != 9 {
			t.Errorf("Sum dim1[%d,0] = %d, want 9", i, got)
		}
	}
}

func TestSumDimOutOfBounds(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int(ctx, Shape{2, 3}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for out-of-bounds dim, got nil")
		}
	}()
	a.Sum(ctx, 5)
}
