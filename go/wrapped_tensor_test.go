package shapes

import (
	"context"
	"testing"
)

// --- Wrap / Unwrap ---

func TestWrap(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	tensor := Float(ctx, Shape{2, 2}, 1.0)
	wt := ctx.Wrap(tensor)

	if wt.Tensor() != tensor {
		t.Fatal("Wrap: Tensor() should return the original tensor")
	}
	if wt.Context() != ctx {
		t.Fatal("Wrap: Context() should return the original context")
	}
}

// --- Utility methods ---

func TestWrappedShape(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Float(Shape{3, 4}, 0.0)
	s := wt.Shape()
	if s[0] != 3 || s[1] != 4 {
		t.Fatalf("Shape() = %v, want [3 4]", s)
	}
}

func TestWrappedDtype(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Float(Shape{2}, 0.0)
	if wt.Dtype() != DtypeF32 {
		t.Fatalf("Dtype() = %v, want DtypeF32", wt.Dtype())
	}
}

// --- Binary ops ---

func TestWrappedPlus(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Int(Shape{2, 2}, 3)
	b := ctx.Int(Shape{2, 2}, 5)

	result := a.Plus(b)

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(i, j).Item().(int8)
			if got != 8 {
				t.Errorf("Plus[%d,%d] = %d, want 8", i, j, got)
			}
		}
	}
}

func TestWrappedMinus(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Int(Shape{2, 3}, 10)
	b := ctx.Int(Shape{2, 3}, 4)

	result := a.Minus(b)

	for i := range uint32(2) {
		for j := range uint32(3) {
			got := result.Get(i, j).Item().(int8)
			if got != 6 {
				t.Errorf("Minus[%d,%d] = %d, want 6", i, j, got)
			}
		}
	}
}

func TestWrappedTimes(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Int(Shape{2, 2}, 3)
	b := ctx.Int(Shape{2, 2}, 7)

	result := a.Times(b)

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(i, j).Item().(int8)
			if got != 21 {
				t.Errorf("Times[%d,%d] = %d, want 21", i, j, got)
			}
		}
	}
}

func TestWrappedDivide(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Float(Shape{2, 2}, 10.0)
	b := ctx.Float(Shape{2, 2}, 4.0)

	result := a.Divide(b)

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(i, j).Item().(float32)
			if got < 2.49 || got > 2.51 {
				t.Errorf("Divide[%d,%d] = %f, want 2.5", i, j, got)
			}
		}
	}
}

func TestWrappedAddInPlace(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Int(Shape{2}, 3)
	b := ctx.Int(Shape{2}, 5)

	a.AddInPlace(b)

	for i := range uint32(2) {
		got := a.Get(i).Item().(int8)
		if got != 8 {
			t.Errorf("AddInPlace[%d] = %d, want 8", i, got)
		}
	}
}

// --- Fluent chaining ---

func TestWrappedChaining(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Int(Shape{2}, 2)
	b := ctx.Int(Shape{2}, 3)
	c := ctx.Int(Shape{2}, 4)

	// (a + b) * c = (2 + 3) * 4 = 20
	result := a.Plus(b).Times(c)

	for i := range uint32(2) {
		got := result.Get(i).Item().(int8)
		if got != 20 {
			t.Errorf("Chain[%d] = %d, want 20", i, got)
		}
	}
}

// --- Context validation ---

func TestWrappedContextMismatchPanic(t *testing.T) {
	ctx1 := New(context.Background())
	defer ctx1.Finish()
	ctx2 := New(context.Background())
	defer ctx2.Finish()

	a := ctx1.Int(Shape{2}, 1)
	b := ctx2.Int(Shape{2}, 2)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for context mismatch, got nil")
		}
	}()
	a.Plus(b)
}

func TestWrappedContextMismatchMinus(t *testing.T) {
	ctx1 := New(context.Background())
	defer ctx1.Finish()
	ctx2 := New(context.Background())
	defer ctx2.Finish()

	a := ctx1.Int(Shape{2}, 10)
	b := ctx2.Int(Shape{2}, 5)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for context mismatch, got nil")
		}
	}()
	a.Minus(b)
}

func TestWrappedContextMismatchMul(t *testing.T) {
	ctx1 := New(context.Background())
	defer ctx1.Finish()
	ctx2 := New(context.Background())
	defer ctx2.Finish()

	a := ctx1.Float(Shape{2, 2}, 1.0)
	b := ctx2.Float(Shape{2, 2}, 1.0)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for context mismatch, got nil")
		}
	}()
	a.Mul(b)
}

func TestWrappedContextMismatchDot(t *testing.T) {
	ctx1 := New(context.Background())
	defer ctx1.Finish()
	ctx2 := New(context.Background())
	defer ctx2.Finish()

	a := ctx1.Float(Shape{3}, 1.0)
	b := ctx2.Float(Shape{3}, 2.0)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for context mismatch, got nil")
		}
	}()
	a.Dot(b)
}

func TestWrappedContextMismatchAddInPlace(t *testing.T) {
	ctx1 := New(context.Background())
	defer ctx1.Finish()
	ctx2 := New(context.Background())
	defer ctx2.Finish()

	a := ctx1.Int(Shape{2}, 1)
	b := ctx2.Int(Shape{2}, 2)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for context mismatch, got nil")
		}
	}()
	a.AddInPlace(b)
}

// --- Unary ops ---

func TestWrappedPow(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Float(Shape{2}, 3.0)
	result := wt.Pow(2.0)

	for i := range uint32(2) {
		got := result.Get(i).Item().(float32)
		if got < 8.99 || got > 9.01 {
			t.Errorf("Pow[%d] = %f, want 9.0", i, got)
		}
	}
}

func TestWrappedExp(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Float(Shape{2}, 0.0)
	result := wt.Exp()

	for i := range uint32(2) {
		got := result.Get(i).Item().(float32)
		if got < 0.99 || got > 1.01 {
			t.Errorf("Exp[%d] = %f, want 1.0", i, got)
		}
	}
}

func TestWrappedNegate(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Float(Shape{2}, 5.0)
	result := wt.Negate()

	for i := range uint32(2) {
		got := result.Get(i).Item().(float32)
		if got > -4.99 || got < -5.01 {
			t.Errorf("Negate[%d] = %f, want -5.0", i, got)
		}
	}
}

// --- Reduction ops ---

func TestWrappedSum(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Float(Shape{2, 3}, 1.0)
	result := wt.Sum(1)

	s := result.Shape()
	if s[0] != 2 || s[1] != 1 {
		t.Fatalf("Sum shape = %v, want [2 1]", s)
	}

	for i := range uint32(2) {
		got := result.Get(i, 0).Item().(float32)
		if got < 2.99 || got > 3.01 {
			t.Errorf("Sum[%d] = %f, want 3.0", i, got)
		}
	}
}

// --- Shape ops ---

func TestWrappedReshape(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Float(Shape{2, 3}, 1.0)
	result := wt.Reshape(3, 2)

	s := result.Shape()
	if s[0] != 3 || s[1] != 2 {
		t.Fatalf("Reshape shape = %v, want [3 2]", s)
	}
}

func TestWrappedTranspose(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Float(Shape{2, 3}, 1.0)
	result := wt.Transpose()

	s := result.Shape()
	if s[0] != 3 || s[1] != 2 {
		t.Fatalf("Transpose shape = %v, want [3 2]", s)
	}
}

func TestWrappedSqueeze(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Float(Shape{1, 3, 1}, 0.0)
	result := wt.Squeeze()

	s := result.Shape()
	if len(s) != 1 || s[0] != 3 {
		t.Fatalf("Squeeze shape = %v, want [3]", s)
	}
}

func TestWrappedSqueezeDim(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Float(Shape{1, 3}, 0.0)
	result := wt.SqueezeDim(0)

	s := result.Shape()
	if len(s) != 1 || s[0] != 3 {
		t.Fatalf("SqueezeDim shape = %v, want [3]", s)
	}
}

func TestWrappedUnSqueeze(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Float(Shape{3}, 0.0)
	result := wt.UnSqueeze(0)

	s := result.Shape()
	if len(s) != 2 || s[0] != 1 || s[1] != 3 {
		t.Fatalf("UnSqueeze shape = %v, want [1 3]", s)
	}
}

func TestWrappedSlice(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Float(Shape{4, 4}, 1.0)
	result := wt.Slice(Range{1, 3}, Range{0, 4})

	s := result.Shape()
	if s[0] != 2 || s[1] != 4 {
		t.Fatalf("Slice shape = %v, want [2 4]", s)
	}
}

// --- Matrix ops ---

func TestWrappedMul(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Float(Shape{2, 3}, 1.0)
	b := ctx.Float(Shape{3, 2}, 1.0)

	result := a.Mul(b)
	s := result.Shape()
	if s[0] != 2 || s[1] != 2 {
		t.Fatalf("Mul shape = %v, want [2 2]", s)
	}

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(i, j).Item().(float32)
			if got < 2.99 || got > 3.01 {
				t.Errorf("Mul[%d,%d] = %f, want 3.0", i, j, got)
			}
		}
	}
}

func TestWrappedDot(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Float(Shape{3}, 2.0)
	b := ctx.Float(Shape{3}, 3.0)

	// Dot returns a 1-element tensor, not a 0-dim scalar
	result := a.Dot(b)
	got := result.Get(0).Item().(float32)
	if got < 17.99 || got > 18.01 {
		t.Errorf("Dot = %f, want 18.0", got)
	}
}

// --- Access ---

func TestWrappedGet(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Int(Shape{3, 3}, 7)
	got := wt.Get(1, 2).Item().(int8)
	if got != 7 {
		t.Errorf("Get(1,2) = %d, want 7", got)
	}
}

func TestWrappedItem(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	wt := ctx.Int(Shape{2, 2}, 5)
	got := wt.Get(0, 0).Item().(int8)
	if got != 5 {
		t.Errorf("Item() = %d, want 5", got)
	}
}
