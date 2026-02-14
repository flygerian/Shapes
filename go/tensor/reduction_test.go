package tensor

import (
	"context"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestSumAlongDim0(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	// 2x3 tensor filled with 2s, sum along dim 0 → 1x3 of 4s
	a := Int(ctx, Shape{2, 3}, 2)

	result := a.Sum(0)

	for j := range uint32(3) {
		got, err := result.GetI8(0, j)
		if err != nil {
			t.Fatalf("GetI8(0,%d): %v", j, err)
		}
		if got != 4 {
			t.Errorf("Sum dim0[0,%d] = %d, want 4", j, got)
		}
	}
}

func TestSumAlongDim1(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	// 2x3 tensor filled with 3s, sum along dim 1 → 2x1 of 9s
	a := Int(ctx, Shape{2, 3}, 3)

	result := a.Sum(1)

	for i := range uint32(2) {
		got, err := result.GetI8(i, 0)
		if err != nil {
			t.Fatalf("GetI8(%d,0): %v", i, err)
		}
		if got != 9 {
			t.Errorf("Sum dim1[%d,0] = %d, want 9", i, got)
		}
	}
}

func TestSumDimOutOfBounds(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	a := Int(ctx, Shape{2, 3}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for out-of-bounds dim, got nil")
		}
	}()
	a.Sum(5)
}
