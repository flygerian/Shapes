package tensor

import (
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestSlice(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	// 3x4 tensor filled with 5s, slice to [0:2, 1:3] → 2x2
	a := Int(ctx, Shape{3, 4}, 5)

	sliced, err := a.Slice(ctx, Range{0, 2}, Range{1, 3})
	if err != nil {
		t.Fatalf("Slice: %v", err)
	}

	for i := uint32(0); i < 2; i++ {
		for j := uint32(0); j < 2; j++ {
			got, err := sliced.GetI8(i, j)
			if err != nil {
				t.Fatalf("GetI8(%d,%d): %v", i, j, err)
			}
			if got != 5 {
				t.Errorf("Slice[%d,%d] = %d, want 5", i, j, got)
			}
		}
	}
}

func TestSliceInvalidRange(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{3, 4}, 1)

	// end < start
	_, err := a.Slice(ctx, Range{2, 0}, Range{0, 4})
	if err == nil {
		t.Fatal("expected error for invalid range, got nil")
	}
}

func TestReshape(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	// 2x3 → 3x2
	a := Int(ctx, Shape{2, 3}, 7)

	reshaped, err := a.Reshape(ctx, Shape{3, 2})
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}

	for i := uint32(0); i < 3; i++ {
		for j := uint32(0); j < 2; j++ {
			got, err := reshaped.GetI8(i, j)
			if err != nil {
				t.Fatalf("GetI8(%d,%d): %v", i, j, err)
			}
			if got != 7 {
				t.Errorf("Reshape[%d,%d] = %d, want 7", i, j, got)
			}
		}
	}
}

func TestReshapeSizeMismatch(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{2, 3}, 1)

	_, err := a.Reshape(ctx, Shape{2, 2})
	if err == nil {
		t.Fatal("expected error for reshape size mismatch, got nil")
	}
}

func TestTranspose(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	// 2x3 filled with 4, transpose dims 0,1 → 3x2
	a := Int(ctx, Shape{2, 3}, 4)

	transposed, err := a.Transpose(ctx, 0, 1)
	if err != nil {
		t.Fatalf("Transpose: %v", err)
	}

	for i := uint32(0); i < 3; i++ {
		for j := uint32(0); j < 2; j++ {
			got, err := transposed.GetI8(i, j)
			if err != nil {
				t.Fatalf("GetI8(%d,%d): %v", i, j, err)
			}
			if got != 4 {
				t.Errorf("Transpose[%d,%d] = %d, want 4", i, j, got)
			}
		}
	}
}

func TestTransposeDimOutOfBounds(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{2, 3}, 1)

	_, err := a.Transpose(ctx, 0, 5)
	if err == nil {
		t.Fatal("expected error for dim out of bounds, got nil")
	}
}

func TestSqueeze(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	// 1x3x1 → 3
	a := Int(ctx, Shape{1, 3, 1}, 9)

	squeezed, err := a.Squeeze(ctx)
	if err != nil {
		t.Fatalf("Squeeze: %v", err)
	}

	for i := uint32(0); i < 3; i++ {
		got, err := squeezed.GetI8(i)
		if err != nil {
			t.Fatalf("GetI8(%d): %v", i, err)
		}
		if got != 9 {
			t.Errorf("Squeeze[%d] = %d, want 9", i, got)
		}
	}
}

func TestUnSqueeze(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	// shape [3] → unsqueeze at dim 0 → [1, 3]
	a := Int(ctx, Shape{3}, 6)

	unsqueezed, err := a.UnSqueeze(ctx, 0)
	if err != nil {
		t.Fatalf("UnSqueeze: %v", err)
	}

	for j := uint32(0); j < 3; j++ {
		got, err := unsqueezed.GetI8(0, j)
		if err != nil {
			t.Fatalf("GetI8(0,%d): %v", j, err)
		}
		if got != 6 {
			t.Errorf("UnSqueeze[0,%d] = %d, want 6", j, got)
		}
	}
}

func TestUnSqueezeDimOutOfBounds(t *testing.T) {
	ctx := shapes.New(nil)
	defer ctx.Close()

	a := Int(ctx, Shape{3}, 1)

	_, err := a.UnSqueeze(ctx, 5)
	if err == nil {
		t.Fatal("expected error for dim out of bounds, got nil")
	}
}
