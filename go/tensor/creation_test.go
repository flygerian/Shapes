package tensor

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestFromFloat32(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	tensor := FromFloat32(ctx, Shape{2, 3}, data)

	if tensor == nil {
		t.Fatal("FromFloat32 returned nil")
	}

	expected := [][]float32{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}

	for i := range uint32(2) {
		for j := range uint32(3) {
			got := tensor.GetF32(i, j)
			if float32(math.Abs(float64(got-expected[i][j]))) > 1e-5 {
				t.Errorf("FromFloat32[%d,%d] = %f, want %f", i, j, got, expected[i][j])
			}
		}
	}
}

func TestFromFloat321D(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	data := []float32{3.14, 2.72, 1.41}
	tensor := FromFloat32(ctx, Shape{3}, data)

	for i, want := range data {
		got := tensor.GetF32(uint32(i))
		if float32(math.Abs(float64(got-want))) > 1e-5 {
			t.Errorf("FromFloat32[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestFromFloat32SizeMismatch(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for data/shape size mismatch, got nil")
		}
	}()

	FromFloat32(ctx, Shape{2, 3}, []float32{1.0, 2.0})
}

func TestFromFloat32EmptyShape(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	result := FromFloat32(ctx, Shape{}, []float32{1.0})
	if result != nil {
		t.Error("expected nil for empty shape")
	}
}

func TestFloatRandom(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	tensor := FloatRandom(ctx, Shape{3, 4})
	if tensor == nil {
		t.Fatal("FloatRandom returned nil")
	}

	shape := ShapeOf(tensor)
	if len(shape) != 2 || shape[0] != 3 || shape[1] != 4 {
		t.Fatalf("expected shape [3,4], got %v", shape)
	}

	for i := range uint32(3) {
		for j := range uint32(4) {
			v := tensor.GetF32(i, j)
			if v < -1 || v > 1 {
				t.Errorf("FloatRandom[%d,%d] = %f, want in [-1, 1]", i, j, v)
			}
		}
	}
}

func TestFloatRandomNotAllSame(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	tensor := FloatRandom(ctx, Shape{100})
	first := tensor.GetF32(0)
	allSame := true
	for i := range uint32(100) {
		v := tensor.GetF32(i)
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
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	data := []float32{1.0, 2.0, 3.0}
	tensor := FromFloat32(ctx, Shape{3}, data)

	if !tensor.RequiresGrad() {
		t.Fatal("expected grad tracking when context has grad enabled")
	}

	// Values should still be correct
	for i, want := range data {
		got := tensor.GetF32(uint32(i))
		if float32(math.Abs(float64(got-want))) > 1e-5 {
			t.Errorf("FromFloat32[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestFromInt8(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	data := [][]int8{{1, 2, 3}, {4, 5, 6}}
	tensor := FromInt8(ctx, data)

	if tensor == nil {
		t.Fatal("FromInt8 returned nil")
	}

	expected := [][]int8{
		{1, 2, 3},
		{4, 5, 6},
	}

	for i := range uint32(2) {
		for j := range uint32(3) {
			got := tensor.GetI8(i, j)
			if got != expected[i][j] {
				t.Errorf("FromInt8[%d,%d] = %d, want %d", i, j, got, expected[i][j])
			}
		}
	}
}

func TestFromInt81D(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	data := []int8{10, 20, 30}
	tensor := FromInt8(ctx, data)

	for i, want := range data {
		got := tensor.GetI8(uint32(i))
		if got != want {
			t.Errorf("FromInt8[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestFromInt8RaggedArray(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for ragged array, got nil")
		}
	}()

	FromInt8(ctx, [][]int8{{1, 2}, {3}})
}

func TestFromInt8EmptyData(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	result := FromInt8(ctx, []int8{})
	if result != nil {
		t.Error("expected nil for empty data")
	}
}

func TestFromInt8WithGrad(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	data := []int8{1, 2, 3}
	tensor := FromInt8(ctx, data)

	if !tensor.RequiresGrad() {
		t.Fatal("expected grad tracking when context has grad enabled")
	}

	// Values should still be correct
	for i, want := range data {
		got := tensor.GetI8(uint32(i))
		if got != want {
			t.Errorf("FromInt8[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestFromInt8NegativeValues(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	data := []int8{-128, -50, 0, 50, 127}
	tensor := FromInt8(ctx, data)

	for i, want := range data {
		got := tensor.GetI8(uint32(i))
		if got != want {
			t.Errorf("FromInt8[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestFromInt83D(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	data := [][][]int8{
		{{1, 2}, {3, 4}},
		{{5, 6}, {7, 8}},
	}
	tensor := FromInt8(ctx, data)

	if tensor == nil {
		t.Fatal("FromInt8 returned nil")
	}

	shape := ShapeOf(tensor)
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
				got := tensor.GetI8(i, j, k)
				if got != expected[i][j][k] {
					t.Errorf("FromInt8[%d,%d,%d] = %d, want %d", i, j, k, got, expected[i][j][k])
				}
			}
		}
	}
}

func TestFromInt84D(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	data := [][][][]int8{
		{{{1, 2}}, {{3, 4}}},
		{{{5, 6}}, {{7, 8}}},
	}
	tensor := FromInt8(ctx, data)

	if tensor == nil {
		t.Fatal("FromInt8 returned nil")
	}

	shape := ShapeOf(tensor)
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
					got := tensor.GetI8(i, j, k, l)
					if got != expected[i][j][k][l] {
						t.Errorf("FromInt8[%d,%d,%d,%d] = %d, want %d", i, j, k, l, got, expected[i][j][k][l])
					}
				}
			}
		}
	}
}
