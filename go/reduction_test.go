package shapes

import (
	"context"
	"math"
	"testing"
)

func TestSumAlongDim0(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 2x3 tensor filled with 2s, sum along dim 0 → 1x3 of 4s
	a := Int8(ctx, Shape{2, 3}, 2)

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
	a := Int8(ctx, Shape{2, 3}, 3)

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

	a := Int8(ctx, Shape{2, 3}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for out-of-bounds dim, got nil")
		}
	}()
	a.Sum(ctx, 5)
}

func TestStdBasic(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := FromFloat32(ctx, Shape{4}, []float32{1, 2, 3, 4})
	result := a.Std(ctx)

	got := result.Item().(float32)
	want := float32(1.2909944) // sqrt(5/3)
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Errorf("Std = %f, want %f", got, want)
	}
}

func TestStdNonFloatPanics(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int8(ctx, Shape{3}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for non-float std, got nil")
		}
	}()
	a.Std(ctx)
}

func TestStdRequiresAtLeastTwoValuesPanics(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Float(ctx, Shape{1}, 42.0)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for std with fewer than 2 values, got nil")
		}
	}()
	a.Std(ctx)
}
