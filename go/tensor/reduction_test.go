package tensor

import (
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestSumAlongDim0(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	// 2x3 tensor filled with 2s, sum along dim 0 → 1x3 of 4s
	a := Int(ctx, Shape{2, 3}, 2)

	result, err := a.Sum(ctx, 0)
	if err != nil {
		t.Fatalf("Sum: %v", err)
	}

	for j := uint32(0); j < 3; j++ {
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
	ctx := shapes.New(nil)
	defer ctx.Close()

	// 2x3 tensor filled with 3s, sum along dim 1 → 2x1 of 9s
	a := Int(ctx, Shape{2, 3}, 3)

	result, err := a.Sum(ctx, 1)
	if err != nil {
		t.Fatalf("Sum: %v", err)
	}

	for i := uint32(0); i < 2; i++ {
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
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{2, 3}, 1)

	_, err := a.Sum(ctx, 5)
	if err == nil {
		t.Fatal("expected error for out-of-bounds dim, got nil")
	}
}
