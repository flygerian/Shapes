package tensor

import (
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestAdd(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{2, 2}, 3)
	b := Int(ctx, Shape{2, 2}, 5)

	result, err := a.Add(ctx, b)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}

	for i := uint32(0); i < 2; i++ {
		for j := uint32(0); j < 2; j++ {
			got, err := result.GetI8(i, j)
			if err != nil {
				t.Fatalf("GetI8(%d,%d): %v", i, j, err)
			}
			if got != 8 {
				t.Errorf("Add[%d,%d] = %d, want 8", i, j, got)
			}
		}
	}
}

func TestSub(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{2, 3}, 10)
	b := Int(ctx, Shape{2, 3}, 4)

	result, err := a.Sub(ctx, b)
	if err != nil {
		t.Fatalf("Sub: %v", err)
	}

	for i := uint32(0); i < 2; i++ {
		for j := uint32(0); j < 3; j++ {
			got, err := result.GetI8(i, j)
			if err != nil {
				t.Fatalf("GetI8(%d,%d): %v", i, j, err)
			}
			if got != 6 {
				t.Errorf("Sub[%d,%d] = %d, want 6", i, j, got)
			}
		}
	}
}

func TestMul(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{2, 2}, 3)
	b := Int(ctx, Shape{2, 2}, 7)

	result, err := a.Mul(ctx, b)
	if err != nil {
		t.Fatalf("Mul: %v", err)
	}

	for i := uint32(0); i < 2; i++ {
		for j := uint32(0); j < 2; j++ {
			got, err := result.GetI8(i, j)
			if err != nil {
				t.Fatalf("GetI8(%d,%d): %v", i, j, err)
			}
			if got != 21 {
				t.Errorf("Mul[%d,%d] = %d, want 21", i, j, got)
			}
		}
	}
}

func TestDiv(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Float(ctx, Shape{2, 2}, 10.0)
	b := Float(ctx, Shape{2, 2}, 4.0)

	result, err := a.Div(ctx, b)
	if err != nil {
		t.Fatalf("Div: %v", err)
	}

	for i := uint32(0); i < 2; i++ {
		for j := uint32(0); j < 2; j++ {
			got, err := result.GetF32(i, j)
			if err != nil {
				t.Fatalf("GetF32(%d,%d): %v", i, j, err)
			}
			if got < 2.49 || got > 2.51 {
				t.Errorf("Div[%d,%d] = %f, want 2.5", i, j, got)
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
	sum, err := a.Add(ctx, b)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	result, err := sum.Mul(ctx, c)
	if err != nil {
		t.Fatalf("Mul: %v", err)
	}

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

	result, err := a.Add(ctx, b)
	if err != nil {
		t.Fatalf("Add broadcast: %v", err)
	}

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

	_, err := a.Add(ctx, b)
	if err == nil {
		t.Fatal("expected error for dtype mismatch, got nil")
	}
}
