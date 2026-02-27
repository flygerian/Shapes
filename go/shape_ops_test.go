package shapes

import (
	"context"
	"testing"
)

func TestSlice(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 3x4 tensor filled with 5s, slice to [0:2, 1:3] → 2x2
	a := ctx.Int(Shape{3, 4}, 5)

	sliced := a.Slice(Range{0, 2}, Range{1, 3})

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := sliced.Get(i, j).Item().(int8)
			if got != 5 {
				t.Errorf("Slice[%d,%d] = %d, want 5", i, j, got)
			}
		}
	}
}

func TestSliceInvalidRange(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Int(Shape{3, 4}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for invalid range, got nil")
		}
	}()
	a.Slice(Range{2, 0}, Range{0, 4})
}

func TestReshape(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 2x3 → 3x2
	a := ctx.Int(Shape{2, 3}, 7)

	reshaped := a.Reshape(3, 2)

	for i := range uint32(3) {
		for j := range uint32(2) {
			got := reshaped.Get(i, j).Item().(int8)
			if got != 7 {
				t.Errorf("Reshape[%d,%d] = %d, want 7", i, j, got)
			}
		}
	}
}

func TestReshapeSizeMismatch(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Int(Shape{2, 3}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for reshape size mismatch, got nil")
		}
	}()
	a.Reshape(2, 2)
}

func TestReshapeWithMinusOne(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 2x3x4 = 24 elements, reshape to (-1, 4) should infer 6
	a := ctx.Int(Shape{2, 3, 4}, 7)

	reshaped := a.Reshape(-1, 4)

	expectedShape := Shape{6, 4}
	actualShape := reshaped.Shape()
	if len(actualShape) != len(expectedShape) {
		t.Fatalf("Reshape shape length mismatch: got %v, want %v", actualShape, expectedShape)
	}
	for i := range expectedShape {
		if actualShape[i] != expectedShape[i] {
			t.Errorf("Reshape shape[%d] = %d, want %d", i, actualShape[i], expectedShape[i])
		}
	}

	// Verify data is preserved
	for i := range uint32(6) {
		for j := range uint32(4) {
			got := reshaped.Get(i, j).Item().(int8)
			if got != 7 {
				t.Errorf("Reshape[%d,%d] = %d, want 7", i, j, got)
			}
		}
	}
}

func TestReshapeWithMinusOneMiddle(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 2x3x4 = 24 elements, reshape to (2, -1, 2) should infer 6
	a := ctx.Int(Shape{2, 3, 4}, 5)

	reshaped := a.Reshape(2, -1, 2)

	expectedShape := Shape{2, 6, 2}
	actualShape := reshaped.Shape()
	if len(actualShape) != len(expectedShape) {
		t.Fatalf("Reshape shape length mismatch: got %v, want %v", actualShape, expectedShape)
	}
	for i := range expectedShape {
		if actualShape[i] != expectedShape[i] {
			t.Errorf("Reshape shape[%d] = %d, want %d", i, actualShape[i], expectedShape[i])
		}
	}
}

func TestReshapeMultipleMinusOnePanics(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Int(Shape{2, 3, 4}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for multiple -1 dimensions, got nil")
		}
	}()
	a.Reshape(-1, -1, 4)
}

func TestReshapeMinusOneSizeMismatch(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 2x3 = 6 elements, reshape to (-1, 4) would need 4 to divide 6 evenly
	a := ctx.Int(Shape{2, 3}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for -1 reshape size mismatch, got nil")
		}
	}()
	a.Reshape(-1, 4)
}

func TestTranspose(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 2x3 filled with 4, transpose dims 0,1 → 3x2
	a := ctx.Int(Shape{2, 3}, 4)

	transposed := a.Transpose(0, 1)

	for i := range uint32(3) {
		for j := range uint32(2) {
			got := transposed.Get(i, j).Item().(int8)
			if got != 4 {
				t.Errorf("Transpose[%d,%d] = %d, want 4", i, j, got)
			}
		}
	}
}

func TestTransposeDimOutOfBounds(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Int(Shape{2, 3}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for dim out of bounds, got nil")
		}
	}()
	a.Transpose(0, 5)
}

func TestSqueeze(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 1x3x1 → 3
	a := ctx.Int(Shape{1, 3, 1}, 9)

	squeezed := a.Squeeze()

	for i := range uint32(3) {
		got := squeezed.Get(i).Item().(int8)
		if got != 9 {
			t.Errorf("Squeeze[%d] = %d, want 9", i, got)
		}
	}
}

func TestUnSqueeze(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// shape [3] → unsqueeze at dim 0 → [1, 3]
	a := ctx.Int(Shape{3}, 6)

	unsqueezed := a.UnSqueeze(0)

	for j := range uint32(3) {
		got := unsqueezed.Get(0, j).Item().(int8)
		if got != 6 {
			t.Errorf("UnSqueeze[0,%d] = %d, want 6", j, got)
		}
	}
}

func TestUnSqueezeDimOutOfBounds(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Int(Shape{3}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for dim out of bounds, got nil")
		}
	}()
	a.UnSqueeze(5)
}
