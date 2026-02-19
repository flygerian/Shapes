package shapes

import (
	"context"
	"math"
	"testing"
)

func TestGetWithCoords(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}
	source := FromFloat32(ctx, Shape{3, 4}, data)

	t.Run("single coordinate", func(t *testing.T) {
		row := source.Get(ctx, 1)
		if row.Shape()[0] != 4 {
			t.Errorf("Expected shape [4], got %v", row.Shape())
		}
		expected := []float32{5.0, 6.0, 7.0, 8.0}
		for i, want := range expected {
			got := row.Get(ctx, uint32(i)).Item().(float32)
			if float32(math.Abs(float64(got-want))) > 1e-5 {
				t.Errorf("Row[1][%d] = %f, want %f", i, got, want)
			}
		}
	})

	t.Run("multiple coordinates", func(t *testing.T) {
		val := source.Get(ctx, 2, 3).Item().(float32)
		expected := float32(12.0)
		if float32(math.Abs(float64(val-expected))) > 1e-5 {
			t.Errorf("Get(2, 3) = %f, want %f", val, expected)
		}
	})

	t.Run("different integer types", func(t *testing.T) {
		val := source.Get(ctx, int(0), uint32(1)).Item().(float32)
		expected := float32(2.0)
		if float32(math.Abs(float64(val-expected))) > 1e-5 {
			t.Errorf("Get(int(0), uint32(1)) = %f, want %f", val, expected)
		}
	})

	t.Run("3D tensor", func(t *testing.T) {
		data3d := make([]float32, 24)
		for i := range data3d {
			data3d[i] = float32(i + 1)
		}
		t3d := FromFloat32(ctx, Shape{2, 3, 4}, data3d)

		val := t3d.Get(ctx, 1, 2, 3).Item().(float32)
		expected := float32(24.0)
		if float32(math.Abs(float64(val-expected))) > 1e-5 {
			t.Errorf("Get(1, 2, 3) = %f, want %f", val, expected)
		}
	})
}

func TestGetWithTensor(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	t.Run("1D indices on 2D tensor", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}
		source := FromFloat32(ctx, Shape{3, 4}, data)

		indices := FromInt8(ctx, []int8{0, 2})
		result := source.Get(ctx, indices)

		expectedShape := Shape{2, 4}
		if len(result.Shape()) != len(expectedShape) {
			t.Fatalf("Expected shape %v, got %v", expectedShape, result.Shape())
		}
		for i := range expectedShape {
			if result.Shape()[i] != expectedShape[i] {
				t.Errorf("Shape[%d] = %d, want %d", i, result.Shape()[i], expectedShape[i])
			}
		}

		expected := [][]float32{
			{1.0, 2.0, 3.0, 4.0},
			{9.0, 10.0, 11.0, 12.0},
		}
		for i := range uint32(2) {
			for j := range uint32(4) {
				got := result.Get(ctx, i, j).Item().(float32)
				if float32(math.Abs(float64(got-expected[i][j]))) > 1e-5 {
					t.Errorf("Result[%d,%d] = %f, want %f", i, j, got, expected[i][j])
				}
			}
		}
	})

	t.Run("2D indices on 2D tensor", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}
		source := FromFloat32(ctx, Shape{3, 4}, data)

		indices := FromInt8(ctx, [][]int8{{0, 1}, {1, 2}})
		result := source.Get(ctx, indices)

		expectedShape := Shape{2, 2, 4}
		if len(result.Shape()) != len(expectedShape) {
			t.Fatalf("Expected shape %v, got %v", expectedShape, result.Shape())
		}
		for i := range expectedShape {
			if result.Shape()[i] != expectedShape[i] {
				t.Errorf("Shape[%d] = %d, want %d", i, result.Shape()[i], expectedShape[i])
			}
		}
	})

	t.Run("single index", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
		source := FromFloat32(ctx, Shape{3, 2}, data)

		indices := FromInt8(ctx, []int8{1})
		result := source.Get(ctx, indices)

		expectedShape := Shape{1, 2}
		if len(result.Shape()) != len(expectedShape) {
			t.Fatalf("Expected shape %v, got %v", expectedShape, result.Shape())
		}

		expected := []float32{3.0, 4.0}
		for i := range uint32(2) {
			got := result.Get(ctx, uint32(0), i).Item().(float32)
			if float32(math.Abs(float64(got-expected[i]))) > 1e-5 {
				t.Errorf("Result[0,%d] = %f, want %f", i, got, expected[i])
			}
		}
	})

	t.Run("3D source tensor", func(t *testing.T) {
		data3d := make([]float32, 24)
		for i := range data3d {
			data3d[i] = float32(i + 1)
		}
		source := FromFloat32(ctx, Shape{2, 3, 4}, data3d)

		indices := FromInt8(ctx, []int8{0, 1})
		result := source.Get(ctx, indices)

		expectedShape := Shape{2, 3, 4}
		if len(result.Shape()) != len(expectedShape) {
			t.Fatalf("Expected shape %v, got %v", expectedShape, result.Shape())
		}

		expected := []float32{1.0, 2.0, 3.0, 4.0}
		for i := range uint32(4) {
			got := result.Get(ctx, uint32(0), uint32(0), i).Item().(float32)
			if float32(math.Abs(float64(got-expected[i]))) > 1e-5 {
				t.Errorf("Result[0,0,%d] = %f, want %f", i, got, expected[i])
			}
		}
	})
}

func TestGetErrors(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Close()

	t.Run("no arguments", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0}
		source := FromFloat32(ctx, Shape{3}, data)

		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for no arguments")
			}
		}()
		source.Get(ctx)
	})

	t.Run("negative index", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0}
		source := FromFloat32(ctx, Shape{3}, data)

		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for negative index")
			}
		}()
		source.Get(ctx, int(-1))
	})

	t.Run("out of bounds coordinate", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0}
		source := FromFloat32(ctx, Shape{3}, data)

		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for out of bounds")
			}
		}()
		source.Get(ctx, uint32(10))
	})

	t.Run("out of bounds tensor index", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0, 4.0}
		source := FromFloat32(ctx, Shape{2, 2}, data)

		indices := FromInt8(ctx, []int8{5})

		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for out of bounds tensor index")
			}
		}()
		source.Get(ctx, indices)
	})

	t.Run("invalid type", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0}
		source := FromFloat32(ctx, Shape{3}, data)

		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for invalid type")
			}
		}()
		source.Get(ctx, "invalid")
	})
}
