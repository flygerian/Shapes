package shapes

import (
	"context"
	"testing"
)

func TestValuesI8(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromInt8(ctx, []int8{1, -2, 3, 4})
	got := src.Values()
	want := []int8{1, -2, 3, 4}

	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		v, ok := got[i].(int8)
		if !ok {
			t.Fatalf("got[%d] has type %T, want int8", i, got[i])
		}
		if v != want[i] {
			t.Fatalf("got[%d] = %d, want %d", i, v, want[i])
		}
	}
}

func TestValuesF32(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	src := FromFloat32(ctx, Shape{3}, []float32{1.5, -2.25, 3.75})
	got := src.Values()
	want := []float32{1.5, -2.25, 3.75}

	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		v, ok := got[i].(float32)
		if !ok {
			t.Fatalf("got[%d] has type %T, want float32", i, got[i])
		}
		if !approxEq(v, want[i], 1e-6) {
			t.Fatalf("got[%d] = %f, want %f", i, v, want[i])
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
	want := []bool{false, false, true}

	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		v, ok := got[i].(bool)
		if !ok {
			t.Fatalf("got[%d] has type %T, want bool", i, got[i])
		}
		if v != want[i] {
			t.Fatalf("got[%d] = %v, want %v", i, v, want[i])
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
