package shapes

import (
	"context"
	"testing"
)

// TestGetTensorAtBackward verifies that gradient from row selection lands only on that row.
// x shape [3, 4], y = x.Get(1), loss = sum(y), grad should be 1 at row 1, 0 elsewhere.
func TestGetTensorAtBackward(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	x := ctx.Float(Shape{3, 4}, 0.0)

	// y = x[1] -> shape [1, 4] after sum (we select row 1)
	y := x.Get(1)

	// loss: sum of y
	loss := y.Sum(0)
	loss.Backward()

	// x.grad row 1 should all be 1, rows 0 and 2 should be 0
	for j := range uint32(4) {
		g0 := x.Grad().Get(0, j).Item().(float32)
		if !approxEq(g0, 0.0, 1e-5) {
			t.Errorf("x.grad[0,%d] = %f, want 0.0", j, g0)
		}
		g1 := x.Grad().Get(1, j).Item().(float32)
		if !approxEq(g1, 1.0, 1e-5) {
			t.Errorf("x.grad[1,%d] = %f, want 1.0", j, g1)
		}
		g2 := x.Grad().Get(2, j).Item().(float32)
		if !approxEq(g2, 0.0, 1e-5) {
			t.Errorf("x.grad[2,%d] = %f, want 0.0", j, g2)
		}
	}
}

// TestSliceBackward verifies that gradient from a slice lands only in the sliced region.
// x shape [4, 4], s = x[1:3, 1:3], loss = sum(s), grad should be 1 only at [1:3, 1:3].
func TestSliceBackward(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	x := ctx.Float(Shape{4, 4}, 0.0)

	// s = x[1:3, 1:3] -> 2x2 view
	s := x.Slice(Range{1, 3}, Range{1, 3})

	// Backward from sum of s
	loss := s.Sum(0)
	loss.Backward()

	for i := range uint32(4) {
		for j := range uint32(4) {
			g := x.Grad().Get(i, j).Item().(float32)
			inSlice := i >= 1 && i < 3 && j >= 1 && j < 3
			if inSlice {
				if !approxEq(g, 1.0, 1e-5) {
					t.Errorf("x.grad[%d,%d] = %f, want 1.0 (in slice)", i, j, g)
				}
			} else {
				if !approxEq(g, 0.0, 1e-5) {
					t.Errorf("x.grad[%d,%d] = %f, want 0.0 (outside slice)", i, j, g)
				}
			}
		}
	}
}

// TestNestedViewBackward verifies backward through x[1:4, :].Get(0) i.e. Slice then GetTensorAt.
func TestNestedViewBackward(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	// x: 4x3
	x := ctx.Float(Shape{4, 3}, 0.0)

	// s = x[1:4, :] -> 3x3 view
	s := x.Slice(Range{1, 4}, Range{0, 3})

	// row = s[0] -> should be x[1, :] shape [3]
	row := s.Get(0)

	// loss = sum of row
	loss := row.Sum(0)
	loss.Backward()

	// grad should be 1.0 at row 1, 0 elsewhere
	for i := range uint32(4) {
		for j := range uint32(3) {
			g := x.Grad().Get(i, j).Item().(float32)
			if i == 1 {
				if !approxEq(g, 1.0, 1e-5) {
					t.Errorf("x.grad[%d,%d] = %f, want 1.0", i, j, g)
				}
			} else {
				if !approxEq(g, 0.0, 1e-5) {
					t.Errorf("x.grad[%d,%d] = %f, want 0.0", i, j, g)
				}
			}
		}
	}
}

// TestIndexWithTensorBackward verifies 1D advanced indexing backward with repeated indices.
// x shape [4, 3], indices = [0, 2, 0], each selected row contributes grad once.
// x.grad[0, :] should be 2 (selected twice), x.grad[2, :] should be 1.
func TestIndexWithTensorBackward(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	x := ctx.Float(Shape{4, 3}, 0.0)

	// indices = [0, 2, 0] (I8 dtype)
	indices := ctx.Wrap(FromInt8(ctx, []int8{0, 2, 0}))

	// y = x[indices] -> shape [3, 3]
	y := x.Get(indices)

	// backward from y sum
	loss := y.Sum(0)
	loss.Backward()

	// x.grad[0, :] = 2.0 (gathered twice)
	for j := range uint32(3) {
		g := x.Grad().Get(0, j).Item().(float32)
		if !approxEq(g, 2.0, 1e-5) {
			t.Errorf("x.grad[0,%d] = %f, want 2.0 (selected twice)", j, g)
		}
	}
	// x.grad[1, :] = 0.0
	for j := range uint32(3) {
		g := x.Grad().Get(1, j).Item().(float32)
		if !approxEq(g, 0.0, 1e-5) {
			t.Errorf("x.grad[1,%d] = %f, want 0.0", j, g)
		}
	}
	// x.grad[2, :] = 1.0
	for j := range uint32(3) {
		g := x.Grad().Get(2, j).Item().(float32)
		if !approxEq(g, 1.0, 1e-5) {
			t.Errorf("x.grad[2,%d] = %f, want 1.0", j, g)
		}
	}
	// x.grad[3, :] = 0.0
	for j := range uint32(3) {
		g := x.Grad().Get(3, j).Item().(float32)
		if !approxEq(g, 0.0, 1e-5) {
			t.Errorf("x.grad[3,%d] = %f, want 0.0", j, g)
		}
	}
}

// TestIndexWithTensor2dBackward verifies 2D advanced indexing backward with repeated pairs.
// x shape [3, 4], rowIdx = [0, 1, 0], colIdx = [2, 3, 2].
// x.grad[0, 2] should be 2.0 (selected twice), x.grad[1, 3] should be 1.0.
func TestIndexWithTensor2dBackward(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	x := ctx.Float(Shape{3, 4}, 0.0)

	rowIdx := ctx.Wrap(FromInt8(ctx, []int8{0, 1, 0}))
	colIdx := ctx.Wrap(FromInt8(ctx, []int8{2, 3, 2}))

	// y = x[rowIdx, colIdx] -> shape [3]
	y := x.Get(rowIdx, colIdx)

	// backward from y sum
	loss := y.Sum(0)
	loss.Backward()

	// x.grad[0,2] = 2.0 (selected at i=0 and i=2)
	g02 := x.Grad().Get(0, 2).Item().(float32)
	if !approxEq(g02, 2.0, 1e-5) {
		t.Errorf("x.grad[0,2] = %f, want 2.0", g02)
	}

	// x.grad[1,3] = 1.0
	g13 := x.Grad().Get(1, 3).Item().(float32)
	if !approxEq(g13, 1.0, 1e-5) {
		t.Errorf("x.grad[1,3] = %f, want 1.0", g13)
	}

	// All other grads should be 0
	for i := range uint32(3) {
		for j := range uint32(4) {
			if (i == 0 && j == 2) || (i == 1 && j == 3) {
				continue
			}
			g := x.Grad().Get(i, j).Item().(float32)
			if !approxEq(g, 0.0, 1e-5) {
				t.Errorf("x.grad[%d,%d] = %f, want 0.0", i, j, g)
			}
		}
	}
}

// TestSliceIndexSumBackward verifies the full chain: slice -> index -> sum -> backward.
func TestSliceIndexSumBackward(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	// x: 6x4
	x := ctx.Float(Shape{6, 4}, 1.0)

	// s = x[2:6, :] -> 4x4 view (rows 2,3,4,5)
	s := x.Slice(Range{2, 6}, Range{0, 4})

	// indices = [0, 2] -> selects s[0,:] = x[2,:] and s[2,:] = x[4,:]
	indices := ctx.Wrap(FromInt8(ctx, []int8{0, 2}))
	y := s.Get(indices) // shape [2, 4]

	// loss = sum over dim 0
	loss := y.Sum(0)
	loss.Backward()

	// x.grad[2, :] should be 1.0, x.grad[4, :] should be 1.0, rest 0.0
	for i := range uint32(6) {
		for j := range uint32(4) {
			g := x.Grad().Get(i, j).Item().(float32)
			if i == 2 || i == 4 {
				if !approxEq(g, 1.0, 1e-5) {
					t.Errorf("x.grad[%d,%d] = %f, want 1.0", i, j, g)
				}
			} else {
				if !approxEq(g, 0.0, 1e-5) {
					t.Errorf("x.grad[%d,%d] = %f, want 0.0", i, j, g)
				}
			}
		}
	}
}
