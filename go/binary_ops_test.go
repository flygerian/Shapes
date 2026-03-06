package shapes

import (
	"context"
	"testing"
)

func TestPlus(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int(ctx, Shape{2, 2}, 3)
	b := Int(ctx, Shape{2, 2}, 5)

	result := a.Plus(ctx, b)

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(ctx, i, j).Item().(int8)
			if got != 8 {
				t.Errorf("Plus[%d,%d] = %d, want 8", i, j, got)
			}
		}
	}
}

func TestPlusScalar(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int(ctx, Shape{2, 2}, 3)
	result := a.Plus(ctx, 1)

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(ctx, i, j).Item().(int8)
			if got != 4 {
				t.Errorf("PlusScalar[%d,%d] = %d, want 4", i, j, got)
			}
		}
	}
}

func TestMinus(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int(ctx, Shape{2, 3}, 10)
	b := Int(ctx, Shape{2, 3}, 4)

	result := a.Minus(ctx, b)

	for i := range uint32(2) {
		for j := range uint32(3) {
			got := result.Get(ctx, i, j).Item().(int8)
			if got != 6 {
				t.Errorf("Minus[%d,%d] = %d, want 6", i, j, got)
			}
		}
	}
}

func TestTimes(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int(ctx, Shape{2, 2}, 3)
	b := Int(ctx, Shape{2, 2}, 7)

	result := a.Times(ctx, b)

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(ctx, i, j).Item().(int8)
			if got != 21 {
				t.Errorf("Times[%d,%d] = %d, want 21", i, j, got)
			}
		}
	}
}

func TestDivide(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Float(ctx, Shape{2, 2}, 10.0)
	b := Float(ctx, Shape{2, 2}, 4.0)

	result := a.Divide(ctx, b)

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(ctx, i, j).Item().(float32)
			if got < 2.49 || got > 2.51 {
				t.Errorf("Divide[%d,%d] = %f, want 2.5", i, j, got)
			}
		}
	}
}

func TestBinaryOpChain(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int(ctx, Shape{2}, 2)
	b := Int(ctx, Shape{2}, 3)
	c := Int(ctx, Shape{2}, 4)

	// (a + b) * c = (2 + 3) * 4 = 20
	sum := a.Plus(ctx, b)
	result := sum.Times(ctx, c)

	for i := range uint32(2) {
		got := result.Get(ctx, i).Item().(int8)
		if got != 20 {
			t.Errorf("Chain[%d] = %d, want 20", i, got)
		}
	}
}

func TestBinaryOpBroadcast(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int(ctx, Shape{2, 3}, 5)
	b := Int(ctx, Shape{1, 3}, 2)

	result := a.Plus(ctx, b)

	for i := range uint32(2) {
		for j := range uint32(3) {
			got := result.Get(ctx, i, j).Item().(int8)
			if got != 7 {
				t.Errorf("Broadcast[%d,%d] = %d, want 7", i, j, got)
			}
		}
	}
}

func TestBinaryOpDtypeMismatch(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int(ctx, Shape{2}, 1)
	b := Float(ctx, Shape{2}, 1.0)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for dtype mismatch, got nil")
		}
	}()
	a.Plus(ctx, b)
}

func TestComparisonOps(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := FromInt8(ctx, []int8{1, 4, 3})
	b := FromInt8(ctx, []int8{2, 4, 1})

	gt := a.GreaterThan(ctx, b)
	if gt.Dtype() != DtypeBool {
		t.Fatalf("expected GreaterThan dtype bool, got %s", gt.Dtype())
	}
	if got := gt.Get(ctx, 0).Item().(bool); got != false {
		t.Errorf("GreaterThan[0] = %v, want false", got)
	}
	if got := gt.Get(ctx, 1).Item().(bool); got != false {
		t.Errorf("GreaterThan[1] = %v, want false", got)
	}
	if got := gt.Get(ctx, 2).Item().(bool); got != true {
		t.Errorf("GreaterThan[2] = %v, want true", got)
	}

	ge := a.GreaterThanOrEqual(ctx, b)
	if ge.Dtype() != DtypeBool {
		t.Fatalf("expected GreaterThanOrEqual dtype bool, got %s", ge.Dtype())
	}
	if got := ge.Get(ctx, 1).Item().(bool); got != true {
		t.Errorf("GreaterThanOrEqual[1] = %v, want true", got)
	}

	lt := a.LessThan(ctx, b)
	if lt.Dtype() != DtypeBool {
		t.Fatalf("expected LessThan dtype bool, got %s", lt.Dtype())
	}
	if got := lt.Get(ctx, 0).Item().(bool); got != true {
		t.Errorf("LessThan[0] = %v, want true", got)
	}

	le := a.LessThanOrEqual(ctx, b)
	if le.Dtype() != DtypeBool {
		t.Fatalf("expected LessThanOrEqual dtype bool, got %s", le.Dtype())
	}
	if got := le.Get(ctx, 1).Item().(bool); got != true {
		t.Errorf("LessThanOrEqual[1] = %v, want true", got)
	}
}

func TestComparisonWithScalar(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := FromInt8(ctx, []int8{1, 4, 3})
	gt := a.GreaterThan(ctx, 2)
	expected := []bool{false, true, true}
	for i, want := range expected {
		got := gt.Get(ctx, uint32(i)).Item().(bool)
		if got != want {
			t.Errorf("GreaterThanScalar[%d] = %v, want %v", i, got, want)
		}
	}
}
