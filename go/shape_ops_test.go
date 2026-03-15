package shapes

import (
	"context"
	"testing"
)

func TestSlice(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 3x4 tensor filled with 5s, slice to [0:2, 1:3] → 2x2
	a := Int8(ctx, Shape{3, 4}, 5)

	sliced := a.Slice(ctx, Range{0, 2}, Range{1, 3})

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := sliced.Get(ctx, i, j).Item().(int8)
			if got != 5 {
				t.Errorf("Slice[%d,%d] = %d, want 5", i, j, got)
			}
		}
	}
}

func TestSliceInvalidRange(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int8(ctx, Shape{3, 4}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for invalid range, got nil")
		}
	}()
	a.Slice(ctx, Range{2, 0}, Range{0, 4})
}

func TestReshape(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 2x3 → 3x2
	a := Int8(ctx, Shape{2, 3}, 7)

	reshaped := a.Reshape(ctx, 3, 2)

	for i := range uint32(3) {
		for j := range uint32(2) {
			got := reshaped.Get(ctx, i, j).Item().(int8)
			if got != 7 {
				t.Errorf("Reshape[%d,%d] = %d, want 7", i, j, got)
			}
		}
	}
}

func TestReshapeSizeMismatch(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int8(ctx, Shape{2, 3}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for reshape size mismatch, got nil")
		}
	}()
	a.Reshape(ctx, 2, 2)
}

func TestReshapeWithMinusOne(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 2x3x4 = 24 elements, reshape to (-1, 4) should infer 6
	a := Int8(ctx, Shape{2, 3, 4}, 7)

	reshaped := a.Reshape(ctx, -1, 4)

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
			got := reshaped.Get(ctx, i, j).Item().(int8)
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
	a := Int8(ctx, Shape{2, 3, 4}, 5)

	reshaped := a.Reshape(ctx, 2, -1, 2)

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

	a := Int8(ctx, Shape{2, 3, 4}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for multiple -1 dimensions, got nil")
		}
	}()
	a.Reshape(ctx, -1, -1, 4)
}

func TestReshapeMinusOneSizeMismatch(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 2x3 = 6 elements, reshape to (-1, 4) would need 4 to divide 6 evenly
	a := Int8(ctx, Shape{2, 3}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for -1 reshape size mismatch, got nil")
		}
	}()
	a.Reshape(ctx, -1, 4)
}

func TestTranspose(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 2x3 filled with 4, transpose dims 0,1 → 3x2
	a := Int8(ctx, Shape{2, 3}, 4)

	transposed := a.Transpose(ctx, 0, 1)

	for i := range uint32(3) {
		for j := range uint32(2) {
			got := transposed.Get(ctx, i, j).Item().(int8)
			if got != 4 {
				t.Errorf("Transpose[%d,%d] = %d, want 4", i, j, got)
			}
		}
	}
}

func TestTransposeDimOutOfBounds(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int8(ctx, Shape{2, 3}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for dim out of bounds, got nil")
		}
	}()
	a.Transpose(ctx, 0, 5)
}

func TestPermute(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := FromInt8(ctx, [][][]int8{
		{
			{1, 2, 3},
			{4, 5, 6},
		},
	})

	permuted := a.Permute(ctx, 0, 2, 1) // [1,2,3] -> [1,3,2]
	shape := permuted.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 3 || shape[2] != 2 {
		t.Fatalf("expected shape [1,3,2], got %v", shape)
	}

	tests := []struct {
		i, j, k uint32
		want    int8
	}{
		{0, 0, 0, 1},
		{0, 0, 1, 4},
		{0, 1, 0, 2},
		{0, 1, 1, 5},
		{0, 2, 0, 3},
		{0, 2, 1, 6},
	}
	for _, tt := range tests {
		got := permuted.Get(ctx, tt.i, tt.j, tt.k).Item().(int8)
		if got != tt.want {
			t.Fatalf("Permute[%d,%d,%d] = %d, want %d", tt.i, tt.j, tt.k, got, tt.want)
		}
	}
}

func TestPermutePanicsOnWrongRank(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int8(ctx, Shape{2, 3}, 1)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for wrong rank dims, got nil")
		}
	}()
	a.Permute(ctx, 1)
}

func TestPermutePanicsOnDuplicateDims(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int8(ctx, Shape{2, 3}, 1)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for duplicate dims, got nil")
		}
	}()
	a.Permute(ctx, 0, 0)
}

func TestSqueeze(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// 1x3x1 → 3
	a := Int8(ctx, Shape{1, 3, 1}, 9)

	squeezed := a.Squeeze(ctx)

	for i := range uint32(3) {
		got := squeezed.Get(ctx, i).Item().(int8)
		if got != 9 {
			t.Errorf("Squeeze[%d] = %d, want 9", i, got)
		}
	}
}

func TestUnSqueeze(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// shape [3] → unsqueeze at dim 0 → [1, 3]
	a := Int8(ctx, Shape{3}, 6)

	unsqueezed := a.UnSqueeze(ctx, 0)

	for j := range uint32(3) {
		got := unsqueezed.Get(ctx, 0, j).Item().(int8)
		if got != 6 {
			t.Errorf("UnSqueeze[0,%d] = %d, want 6", j, got)
		}
	}
}

func TestUnSqueezeDimOutOfBounds(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Int8(ctx, Shape{3}, 1)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for dim out of bounds, got nil")
		}
	}()
	a.UnSqueeze(ctx, 5)
}

func TestConcat2D(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// Two 2x3 tensors with different values
	a := FromFloat32(ctx, Shape{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	b := FromFloat32(ctx, Shape{2, 3}, []float32{
		7, 8, 9,
		10, 11, 12,
	})

	// Concat along dim 0 (rows) → 4x3
	concatenated := a.Concat(ctx, 0, b)

	shape := concatenated.Shape()
	if len(shape) != 2 || shape[0] != 4 || shape[1] != 3 {
		t.Fatalf("expected shape [4,3], got %v", shape)
	}

	// Verify values
	expected := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	for i := range uint32(4) {
		for j := range uint32(3) {
			got := concatenated.Get(ctx, i, j).Item().(float32)
			idx := i*3 + j
			if got != expected[idx] {
				t.Errorf("Concat[%d,%d] = %v, want %v", i, j, got, expected[idx])
			}
		}
	}
}

func TestConcatDim1(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// Two 2x2 tensors
	a := FromFloat32(ctx, Shape{2, 2}, []float32{
		1, 2,
		3, 4,
	})
	b := FromFloat32(ctx, Shape{2, 3}, []float32{
		5, 6, 7,
		8, 9, 10,
	})

	// Concat along dim 1 (cols) → 2x5
	concatenated := a.Concat(ctx, 1, b)

	shape := concatenated.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 5 {
		t.Fatalf("expected shape [2,5], got %v", shape)
	}

	// Verify values
	expected := []float32{1, 2, 5, 6, 7, 3, 4, 8, 9, 10}
	for i := range uint32(2) {
		for j := range uint32(5) {
			got := concatenated.Get(ctx, i, j).Item().(float32)
			idx := i*5 + j
			if got != expected[idx] {
				t.Errorf("Concat[%d,%d] = %v, want %v", i, j, got, expected[idx])
			}
		}
	}
}

func TestConcat1D(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// Two 1D tensors
	a := FromFloat32(ctx, Shape{3}, []float32{1, 2, 3})
	b := FromFloat32(ctx, Shape{2}, []float32{4, 5})

	// Concat → 5-element vector
	concatenated := a.Concat(ctx, 0, b)

	shape := concatenated.Shape()
	if len(shape) != 1 || shape[0] != 5 {
		t.Fatalf("expected shape [5], got %v", shape)
	}

	expected := []float32{1, 2, 3, 4, 5}
	for i := range uint32(5) {
		got := concatenated.Get(ctx, i).Item().(float32)
		if got != expected[i] {
			t.Errorf("Concat[%d] = %v, want %v", i, got, expected[i])
		}
	}
}

func TestConcatMultiple(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// Three 1x2 tensors
	a := FromFloat32(ctx, Shape{1, 2}, []float32{1, 2})
	b := FromFloat32(ctx, Shape{1, 2}, []float32{3, 4})
	c := FromFloat32(ctx, Shape{1, 2}, []float32{5, 6})

	// Concat all three along dim 0
	concatenated := a.Concat(ctx, 0, b, c)

	shape := concatenated.Shape()
	if len(shape) != 2 || shape[0] != 3 || shape[1] != 2 {
		t.Fatalf("expected shape [3,2], got %v", shape)
	}
}

func TestConcatEmpty(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Float(ctx, Shape{2, 3}, 5.0)

	// Concat with no tensors should clone
	cloned := a.Concat(ctx, 0)

	shape := cloned.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("expected shape [2,3], got %v", shape)
	}
}

func TestConcatShapeMismatch(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Float(ctx, Shape{2, 3}, 1.0)
	b := Float(ctx, Shape{2, 4}, 2.0) // Mismatched dim 1

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for shape mismatch, got nil")
		}
	}()
	a.Concat(ctx, 0, b)
}

func TestStack(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// Two 2x3 tensors
	a := FromFloat32(ctx, Shape{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	b := FromFloat32(ctx, Shape{2, 3}, []float32{
		7, 8, 9,
		10, 11, 12,
	})

	// Stack along dim 0 → 2x2x3
	stacked := Stack(ctx, 0, a, b)

	shape := stacked.Shape()
	if len(shape) != 3 || shape[0] != 2 || shape[1] != 2 || shape[2] != 3 {
		t.Fatalf("expected shape [2,2,3], got %v", shape)
	}

	// Verify values: stacked[0] should be a, stacked[1] should be b
	for i := range uint32(2) {
		for j := range uint32(3) {
			got0 := stacked.Get(ctx, 0, i, j).Item().(float32)
			got1 := stacked.Get(ctx, 1, i, j).Item().(float32)
			want0 := a.Get(ctx, i, j).Item().(float32)
			want1 := b.Get(ctx, i, j).Item().(float32)
			if got0 != want0 {
				t.Errorf("Stack[0,%d,%d] = %v, want %v", i, j, got0, want0)
			}
			if got1 != want1 {
				t.Errorf("Stack[1,%d,%d] = %v, want %v", i, j, got1, want1)
			}
		}
	}
}

func TestWrappedTensorConcat(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	// Create wrapped tensors using the Float creation function
	wa := &WrappedTensor{tensor: Float(ctx, Shape{1, 2}, 1.0), context: ctx}
	wb := &WrappedTensor{tensor: Float(ctx, Shape{1, 2}, 2.0), context: ctx}

	// Concat using wrapped tensor API
	concatenated := wa.Concat(0, wb)

	shape := concatenated.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 2 {
		t.Fatalf("expected shape [2,2], got %v", shape)
	}
}
