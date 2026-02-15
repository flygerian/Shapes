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
			got, err := tensor.GetF32(i, j)
			if err != nil {
				t.Fatalf("GetF32(%d,%d): %v", i, j, err)
			}
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
		got, err := tensor.GetF32(uint32(i))
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
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
		got, err := tensor.GetF32(uint32(i))
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if float32(math.Abs(float64(got-want))) > 1e-5 {
			t.Errorf("FromFloat32[%d] = %f, want %f", i, got, want)
		}
	}
}
