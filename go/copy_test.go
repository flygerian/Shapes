package shapes

import (
	"context"
	"testing"
)

func TestValuesI8(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromInt8(ctx, []int8{1, -2, 3, 4})
	got, ok := src.Values().([]int8)
	if !ok {
		t.Fatalf("Values() has type %T, want []int8", src.Values())
	}
	want := []int8{1, -2, 3, 4}

	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestValuesF32(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromFloat32(ctx, Shape{3}, []float32{1.5, -2.25, 3.75})
	got, ok := src.Values().([]float32)
	if !ok {
		t.Fatalf("Values() has type %T, want []float32", src.Values())
	}
	want := []float32{1.5, -2.25, 3.75}

	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !approxEq(got[i], want[i], 1e-6) {
			t.Fatalf("got[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestValuesBool(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := FromInt8(ctx, []int8{1, 4, 3})
	b := FromInt8(ctx, []int8{2, 4, 1})
	gt := a.GreaterThan(ctx, b)

	got := gt.Values()
	gotBools, ok := got.([]bool)
	if !ok {
		t.Fatalf("Values() has type %T, want []bool", got)
	}
	want := []bool{false, false, true}

	if len(gotBools) != len(want) {
		t.Fatalf("len = %d, want %d", len(gotBools), len(want))
	}
	for i := range want {
		if gotBools[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v", i, gotBools[i], want[i])
		}
	}
}

func TestValuesPanicsForView(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromInt8(ctx, [][]int8{{1, 2, 3}, {4, 5, 6}})
	row := src.Get(ctx, uint32(1))

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for non-contiguous view tensor")
		}
	}()

	_ = row.Values()
}
