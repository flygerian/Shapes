package tensor

import (
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestPlus(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{2, 2}, 3)
	b := Int(ctx, Shape{2, 2}, 5)

	result := a.Plus(b)

	for i := uint32(0); i < 2; i++ {
		for j := uint32(0); j < 2; j++ {
			got, err := result.GetI8(i, j)
			if err != nil {
				t.Fatalf("GetI8(%d,%d): %v", i, j, err)
			}
			if got != 8 {
				t.Errorf("Plus[%d,%d] = %d, want 8", i, j, got)
			}
		}
	}
}

func TestMinus(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{2, 3}, 10)
	b := Int(ctx, Shape{2, 3}, 4)

	result := a.Minus(b)

	for i := uint32(0); i < 2; i++ {
		for j := uint32(0); j < 3; j++ {
			got, err := result.GetI8(i, j)
			if err != nil {
				t.Fatalf("GetI8(%d,%d): %v", i, j, err)
			}
			if got != 6 {
				t.Errorf("Minus[%d,%d] = %d, want 6", i, j, got)
			}
		}
	}
}

func TestTimes(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{2, 2}, 3)
	b := Int(ctx, Shape{2, 2}, 7)

	result := a.Times(b)

	for i := uint32(0); i < 2; i++ {
		for j := uint32(0); j < 2; j++ {
			got, err := result.GetI8(i, j)
			if err != nil {
				t.Fatalf("GetI8(%d,%d): %v", i, j, err)
			}
			if got != 21 {
				t.Errorf("Times[%d,%d] = %d, want 21", i, j, got)
			}
		}
	}
}

func TestDivide(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Float(ctx, Shape{2, 2}, 10.0)
	b := Float(ctx, Shape{2, 2}, 4.0)

	result := a.Divide(b)

	for i := uint32(0); i < 2; i++ {
		for j := uint32(0); j < 2; j++ {
			got, err := result.GetF32(i, j)
			if err != nil {
				t.Fatalf("GetF32(%d,%d): %v", i, j, err)
			}
			if got < 2.49 || got > 2.51 {
				t.Errorf("Divide[%d,%d] = %f, want 2.5", i, j, got)
			}
		}
	}
}

func TestBinaryOpChain(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{2}, 2)
	b := Int(ctx, Shape{2}, 3)
	c := Int(ctx, Shape{2}, 4)

	// (a + b) * c = (2 + 3) * 4 = 20
	sum := a.Plus(b)
	result := sum.Times(c)

	for i := uint32(0); i < 2; i++ {
		got, err := result.GetI8(i)
		if err != nil {
			t.Fatalf("GetI8(%d): %v", i, err)
		}
		if got != 20 {
			t.Errorf("Chain[%d] = %d, want 20", i, got)
		}
	}
}

func TestBinaryOpBroadcast(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{2, 3}, 5)
	b := Int(ctx, Shape{1, 3}, 2)

	result := a.Plus(b)

	for i := uint32(0); i < 2; i++ {
		for j := uint32(0); j < 3; j++ {
			got, err := result.GetI8(i, j)
			if err != nil {
				t.Fatalf("GetI8(%d,%d): %v", i, j, err)
			}
			if got != 7 {
				t.Errorf("Broadcast[%d,%d] = %d, want 7", i, j, got)
			}
		}
	}
}

func TestBinaryOpDtypeMismatch(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{2}, 1)
	b := Float(ctx, Shape{2}, 1.0)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for dtype mismatch, got nil")
		}
	}()
	a.Plus(b)
}
