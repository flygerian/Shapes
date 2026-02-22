package shapes

import (
	"context"
	"math"
	"testing"
)

func TestFromFloat32(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	tensor := ctx.FromFloat32(Shape{2, 3}, data)

	if tensor == nil {
		t.Fatal("FromFloat32 returned nil")
	}

	expected := [][]float32{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}

	for i := range uint32(2) {
		for j := range uint32(3) {
			got := tensor.Get(i, j).Item().(float32)
			if float32(math.Abs(float64(got-expected[i][j]))) > 1e-5 {
				t.Errorf("FromFloat32[%d,%d] = %f, want %f", i, j, got, expected[i][j])
			}
		}
	}
}

func TestFromFloat321D(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	data := []float32{3.14, 2.72, 1.41}
	tensor := ctx.FromFloat32(Shape{3}, data)

	for i, want := range data {
		got := tensor.Get(uint32(i)).Item().(float32)
		if float32(math.Abs(float64(got-want))) > 1e-5 {
			t.Errorf("FromFloat32[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestFromFloat32SizeMismatch(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for data/shape size mismatch, got nil")
		}
	}()

	ctx.FromFloat32(Shape{2, 3}, []float32{1.0, 2.0})
}

func TestFromFloat32EmptyShape(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	result := ctx.FromFloat32(Shape{}, []float32{1.0})
	if result != nil {
		t.Error("expected nil for empty shape")
	}
}

func TestFloatRandom(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	tensor := ctx.FloatRandom(Shape{3, 4})
	if tensor == nil {
		t.Fatal("FloatRandom returned nil")
	}

	shape := tensor.Shape()
	if len(shape) != 2 || shape[0] != 3 || shape[1] != 4 {
		t.Fatalf("expected shape [3,4], got %v", shape)
	}

	for i := range uint32(3) {
		for j := range uint32(4) {
			v := tensor.Get(i, j).Item().(float32)
			if v < -1 || v > 1 {
				t.Errorf("FloatRandom[%d,%d] = %f, want in [-1, 1]", i, j, v)
			}
		}
	}
}

func TestFloatRandomNotAllSame(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	tensor := ctx.FloatRandom(Shape{100})
	first := tensor.Get(0).Item().(float32)
	allSame := true
	for i := range uint32(100) {
		v := tensor.Get(i).Item().(float32)
		if v != first {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("expected random values, but all elements are the same")
	}
}

func TestFromFloat32WithGrad(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	data := []float32{1.0, 2.0, 3.0}
	tensor := ctx.FromFloat32(Shape{3}, data)

	if !tensor.RequiresGrad() {
		t.Fatal("expected grad tracking when context has grad enabled")
	}

	// Values should still be correct
	for i, want := range data {
		got := tensor.Get(uint32(i)).Item().(float32)
		if float32(math.Abs(float64(got-want))) > 1e-5 {
			t.Errorf("FromFloat32[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestFromInt8(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	data := [][]int8{{1, 2, 3}, {4, 5, 6}}
	tensor := ctx.FromInt8(data)

	if tensor == nil {
		t.Fatal("FromInt8 returned nil")
	}

	expected := [][]int8{
		{1, 2, 3},
		{4, 5, 6},
	}

	for i := range uint32(2) {
		for j := range uint32(3) {
			got := tensor.Get(i, j).Item().(int8)
			if got != expected[i][j] {
				t.Errorf("FromInt8[%d,%d] = %d, want %d", i, j, got, expected[i][j])
			}
		}
	}
}

func TestFromInt81D(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	data := []int8{10, 20, 30}
	tensor := ctx.FromInt8(data)

	for i, want := range data {
		got := tensor.Get(uint32(i)).Item().(int8)
		if got != want {
			t.Errorf("FromInt8[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestFromInt8RaggedArray(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for ragged array, got nil")
		}
	}()

	ctx.FromInt8([][]int8{{1, 2}, {3}})
}

func TestFromInt8EmptyData(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	result := ctx.FromInt8([]int8{})
	if result != nil {
		t.Error("expected nil for empty data")
	}
}

func TestFromInt8WithGrad(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	data := []int8{1, 2, 3}
	tensor := ctx.FromInt8(data)

	if !tensor.RequiresGrad() {
		t.Fatal("expected grad tracking when context has grad enabled")
	}

	// Values should still be correct
	for i, want := range data {
		got := tensor.Get(uint32(i)).Item().(int8)
		if got != want {
			t.Errorf("FromInt8[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestFromInt8NegativeValues(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	data := []int8{-128, -50, 0, 50, 127}
	tensor := ctx.FromInt8(data)

	for i, want := range data {
		got := tensor.Get(uint32(i)).Item().(int8)
		if got != want {
			t.Errorf("FromInt8[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestFromInt83D(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	data := [][][]int8{
		{{1, 2}, {3, 4}},
		{{5, 6}, {7, 8}},
	}
	tensor := ctx.FromInt8(data)

	if tensor == nil {
		t.Fatal("FromInt8 returned nil")
	}

	shape := tensor.Shape()
	if len(shape) != 3 || shape[0] != 2 || shape[1] != 2 || shape[2] != 2 {
		t.Fatalf("expected shape [2,2,2], got %v", shape)
	}

	expected := [][][]int8{
		{{1, 2}, {3, 4}},
		{{5, 6}, {7, 8}},
	}

	for i := range uint32(2) {
		for j := range uint32(2) {
			for k := range uint32(2) {
				got := tensor.Get(i, j, k).Item().(int8)
				if got != expected[i][j][k] {
					t.Errorf("FromInt8[%d,%d,%d] = %d, want %d", i, j, k, got, expected[i][j][k])
				}
			}
		}
	}
}

func TestFromInt84D(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	data := [][][][]int8{
		{{{1, 2}}, {{3, 4}}},
		{{{5, 6}}, {{7, 8}}},
	}
	tensor := ctx.FromInt8(data)

	if tensor == nil {
		t.Fatal("FromInt8 returned nil")
	}

	shape := tensor.Shape()
	if len(shape) != 4 || shape[0] != 2 || shape[1] != 2 || shape[2] != 1 || shape[3] != 2 {
		t.Fatalf("expected shape [2,2,1,2], got %v", shape)
	}

	expected := [][][][]int8{
		{{{1, 2}}, {{3, 4}}},
		{{{5, 6}}, {{7, 8}}},
	}

	for i := range uint32(2) {
		for j := range uint32(2) {
			for k := range uint32(1) {
				for l := range uint32(2) {
					got := tensor.Get(i, j, k, l).Item().(int8)
					if got != expected[i][j][k][l] {
						t.Errorf("FromInt8[%d,%d,%d,%d] = %d, want %d", i, j, k, l, got, expected[i][j][k][l])
					}
				}
			}
		}
	}
}

func TestOneHotBasic(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	// Simple 1D case: indices [0, 2, 1] with numClasses=3
	// Should produce: [[1,0,0], [0,0,1], [0,1,0]]
	indices := ctx.FromInt8([]int8{0, 2, 1})
	oneHot := ctx.OneHot(indices, 3)

	if oneHot == nil {
		t.Fatal("OneHot returned nil")
	}

	shape := oneHot.Shape()
	if len(shape) != 2 || shape[0] != 3 || shape[1] != 3 {
		t.Fatalf("expected shape [3,3], got %v", shape)
	}

	expected := [][]float32{
		{1, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
	}

	for i := range uint32(3) {
		for j := range uint32(3) {
			got := oneHot.Get(i, j).Item().(float32)
			if got != expected[i][j] {
				t.Errorf("OneHot[%d,%d] = %f, want %f", i, j, got, expected[i][j])
			}
		}
	}
}

func TestOneHot2DIndices(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	// 2D indices: [[0, 2], [1, 0]] with numClasses=3
	// Should produce shape [2, 2, 3]
	indices := ctx.FromInt8([][]int8{{0, 2}, {1, 0}})
	oneHot := ctx.OneHot(indices, 3)

	if oneHot == nil {
		t.Fatal("OneHot returned nil")
	}

	shape := oneHot.Shape()
	if len(shape) != 3 || shape[0] != 2 || shape[1] != 2 || shape[2] != 3 {
		t.Fatalf("expected shape [2,2,3], got %v", shape)
	}

	// Check first row: [0, 2] -> [[1,0,0], [0,0,1]]
	if oneHot.Get(0, 0, 0).Item().(float32) != 1.0 {
		t.Error("Expected oneHot[0,0,0] = 1.0")
	}
	if oneHot.Get(0, 0, 1).Item().(float32) != 0.0 {
		t.Error("Expected oneHot[0,0,1] = 0.0")
	}
	if oneHot.Get(0, 0, 2).Item().(float32) != 0.0 {
		t.Error("Expected oneHot[0,0,2] = 0.0")
	}
	if oneHot.Get(0, 1, 2).Item().(float32) != 1.0 {
		t.Error("Expected oneHot[0,1,2] = 1.0")
	}

	// Check second row: [1, 0] -> [[0,1,0], [1,0,0]]
	if oneHot.Get(1, 0, 1).Item().(float32) != 1.0 {
		t.Error("Expected oneHot[1,0,1] = 1.0")
	}
	if oneHot.Get(1, 1, 0).Item().(float32) != 1.0 {
		t.Error("Expected oneHot[1,1,0] = 1.0")
	}
}

func TestOneHotWithGrad(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	indices := ctx.FromInt8([]int8{0, 1})
	oneHot := ctx.OneHot(indices, 2)

	if oneHot == nil {
		t.Fatal("OneHot returned nil")
	}

	if !oneHot.RequiresGrad() {
		t.Fatal("expected grad tracking when context has grad enabled")
	}

	// OneHot must be an intermediate node (with indices as input), not a leaf.
	// This ensures the backward pass can traverse through it.
	node := oneHot.Tensor().Computation
	if node.Backward == nil {
		t.Fatal("expected OneHot to register a backward function (not a leaf node)")
	}
	if len(node.Inputs) != 1 {
		t.Fatalf("expected 1 input on OneHot node, got %d", len(node.Inputs))
	}
	if node.Op != OpOneHot {
		t.Fatalf("expected Op == OpOneHot, got %v", node.Op)
	}
}

func TestOneHotBackwardNoopAndGraphTraversal(t *testing.T) {
	// Verifies that:
	// 1. Backward does not panic when traversing through a OneHot node.
	// 2. Gradients flow to tensors that used the OneHot output (downstream ops).
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	indices := ctx.FromInt8([]int8{0, 1}) // 2 samples: class 0 and class 1

	// numClasses=2 → shape [2, 2]
	oneHot := ctx.OneHot(indices, 2)

	// Downstream op: sum over all elements (scalar output for easy backward).
	// grad of sum w.r.t. oneHot is all-ones — but since OneHot backward is a no-op,
	// oneHot.Grad() stays all-zeros (no panic, no accumulation).
	result := oneHot.Tensor().Sum(ctx, 0).Sum(ctx, 0)

	// This must not panic.
	result.Backward(ctx)

	// oneHot.Grad() should exist and be populated by the downstream sum backward.
	grad := oneHot.Tensor().Grad()
	if grad == nil {
		t.Fatal("expected grad tensor on oneHot output")
	}
	// The no-op backward means gradient does NOT propagate to indices,
	// but oneHot.Grad() is set by the upstream sum backward (all-ones for sum).
	gradShape := ShapeOf(grad)
	for i := range gradShape[0] {
		for j := range gradShape[1] {
			v := grad.Get(ctx, i, j).Item().(float32)
			if v != 1.0 {
				t.Errorf("oneHot.Grad()[%d,%d] = %f, want 1.0 (sum backward)", i, j, v)
			}
		}
	}

	// indices.Grad() should remain zero — no gradient flows through discrete indices.
	indicesGrad := indices.Tensor().Grad()
	if indicesGrad == nil {
		t.Fatal("expected grad tensor on indices")
	}
	indicesShape := ShapeOf(indicesGrad)
	for i := range indicesShape[0] {
		v := indicesGrad.Get(ctx, i).Item().(float32)
		if v != 0.0 {
			t.Errorf("indices.Grad()[%d] = %f, want 0.0 (no gradient through discrete indices)", i, v)
		}
	}
}

func TestOneHotNilInput(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for nil input")
		}
	}()
	ctx.OneHot(nil, 3)
}

func TestOneHotZeroClasses(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	indices := ctx.FromInt8([]int8{0, 1})
	result := ctx.OneHot(indices, 0)
	if result != nil {
		t.Error("expected nil for zero classes")
	}
}

func TestArangeBasic(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	tensor := ctx.Arange(0.0, 5.0, 1.0)
	if tensor == nil {
		t.Fatal("Arange returned nil")
	}

	shape := tensor.Shape()
	if len(shape) != 1 || shape[0] != 5 {
		t.Fatalf("expected shape [5], got %v", shape)
	}

	expected := []float32{0.0, 1.0, 2.0, 3.0, 4.0}
	for i, want := range expected {
		got := tensor.Get(uint32(i)).Item().(float32)
		if float32(math.Abs(float64(got-want))) > 1e-5 {
			t.Errorf("Arange[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestArangeSingleArg(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	// Arange(5) should produce [0, 1, 2, 3, 4]
	tensor := ctx.Arange(5.0)
	if tensor == nil {
		t.Fatal("Arange returned nil")
	}

	shape := tensor.Shape()
	if len(shape) != 1 || shape[0] != 5 {
		t.Fatalf("expected shape [5], got %v", shape)
	}

	expected := []float32{0.0, 1.0, 2.0, 3.0, 4.0}
	for i, want := range expected {
		got := tensor.Get(uint32(i)).Item().(float32)
		if float32(math.Abs(float64(got-want))) > 1e-5 {
			t.Errorf("Arange[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestArangeTwoArgs(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	// Arange(2, 7) should produce [2, 3, 4, 5, 6]
	tensor := ctx.Arange(2.0, 7.0)
	if tensor == nil {
		t.Fatal("Arange returned nil")
	}

	shape := tensor.Shape()
	if len(shape) != 1 || shape[0] != 5 {
		t.Fatalf("expected shape [5], got %v", shape)
	}

	expected := []float32{2.0, 3.0, 4.0, 5.0, 6.0}
	for i, want := range expected {
		got := tensor.Get(uint32(i)).Item().(float32)
		if float32(math.Abs(float64(got-want))) > 1e-5 {
			t.Errorf("Arange[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestArangeNegativeStep(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	tensor := ctx.Arange(10.0, 0.0, -2.0)
	if tensor == nil {
		t.Fatal("Arange with negative step returned nil")
	}

	shape := tensor.Shape()
	if len(shape) != 1 || shape[0] != 5 {
		t.Fatalf("expected shape [5], got %v", shape)
	}

	expected := []float32{10.0, 8.0, 6.0, 4.0, 2.0}
	for i, want := range expected {
		got := tensor.Get(uint32(i)).Item().(float32)
		if float32(math.Abs(float64(got-want))) > 1e-5 {
			t.Errorf("Arange[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestArangeNonIntegerStep(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	tensor := ctx.Arange(1.0, 5.0, 0.5)
	if tensor == nil {
		t.Fatal("Arange with non-integer step returned nil")
	}

	shape := tensor.Shape()
	if len(shape) != 1 || shape[0] != 8 {
		t.Fatalf("expected shape [8], got %v", shape)
	}

	expected := []float32{1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5}
	for i, want := range expected {
		got := tensor.Get(uint32(i)).Item().(float32)
		if float32(math.Abs(float64(got-want))) > 1e-5 {
			t.Errorf("Arange[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestArangeDefaultStep(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	// Step of 0 should default to 1
	tensor := ctx.Arange(0.0, 3.0, 0.0)
	if tensor == nil {
		t.Fatal("Arange with step=0 returned nil")
	}

	shape := tensor.Shape()
	if len(shape) != 1 || shape[0] != 3 {
		t.Fatalf("expected shape [3], got %v", shape)
	}

	expected := []float32{0.0, 1.0, 2.0}
	for i, want := range expected {
		got := tensor.Get(uint32(i)).Item().(float32)
		if float32(math.Abs(float64(got-want))) > 1e-5 {
			t.Errorf("Arange[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestArangeEmptyRange(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	// start >= end with positive step should return nil
	result := ctx.Arange(5.0, 5.0, 1.0)
	if result != nil {
		t.Error("expected nil for start >= end with positive step")
	}

	// start > end with positive step should return nil
	result = ctx.Arange(10.0, 5.0, 1.0)
	if result != nil {
		t.Error("expected nil for start > end with positive step")
	}

	// start <= end with negative step should return nil
	result = ctx.Arange(5.0, 5.0, -1.0)
	if result != nil {
		t.Error("expected nil for start <= end with negative step")
	}

	// start < end with negative step should return nil
	result = ctx.Arange(0.0, 5.0, -1.0)
	if result != nil {
		t.Error("expected nil for start < end with negative step")
	}
}

func TestArangeWithGrad(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Close()

	tensor := ctx.Arange(0.0, 3.0, 1.0)
	if tensor == nil {
		t.Fatal("Arange returned nil")
	}

	if !tensor.RequiresGrad() {
		t.Fatal("expected grad tracking when context has grad enabled")
	}
}

func TestArangeStandalone(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	// Test the standalone Arange function (not the Context method)
	tensor := Arange(ctx, 1.0, 5.0, 1.0)
	if tensor == nil {
		t.Fatal("Arange returned nil")
	}

	shape := tensor.Shape()
	if len(shape) != 1 || shape[0] != 4 {
		t.Fatalf("expected shape [4], got %v", shape)
	}

	expected := []float32{1.0, 2.0, 3.0, 4.0}
	for i, want := range expected {
		got := tensor.Get(ctx, uint32(i)).Item().(float32)
		if float32(math.Abs(float64(got-want))) > 1e-5 {
			t.Errorf("Arange[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestArangeInvalidArgs(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	// Test with 0 arguments (should panic)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for 0 arguments")
		}
	}()
	ctx.Arange()
}
