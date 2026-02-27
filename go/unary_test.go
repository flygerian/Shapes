package shapes

import (
	"context"
	"math"
	"testing"
)

func TestPow(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Float(Shape{2, 2}, 3.0)
	result := a.Pow(2.0)

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(i, j).Item().(float32)
			if math.Abs(float64(got)-9.0) > 1e-4 {
				t.Errorf("Pow[%d,%d] = %f, want 9.0", i, j, got)
			}
		}
	}
}

func TestPowFractional(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Float(Shape{2}, 4.0)
	result := a.Pow(0.5)

	for i := range uint32(2) {
		got := result.Get(i).Item().(float32)
		if math.Abs(float64(got)-2.0) > 1e-4 {
			t.Errorf("Pow(0.5)[%d] = %f, want 2.0", i, got)
		}
	}
}

func TestPowBackwardSquare(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Finish()

	// x=3, x^2=9, d(x^2)/dx = 2x = 6
	x := ctx.Float(Shape{1}, 3.0)
	y := x.Pow(2.0)
	y.Backward()

	got := x.Grad().(*Tensor).Get(ctx, 0).Item().(float32)
	if !approxEq(got, 6.0, 1e-4) {
		t.Errorf("d(x^2)/dx at x=3 = %f, want 6.0", got)
	}
}

func TestPowBackwardCube(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Finish()

	// x=2, x^3=8, d(x^3)/dx = 3x^2 = 12
	x := ctx.Float(Shape{1}, 2.0)
	y := x.Pow(3.0)
	y.Backward()

	got := x.Grad().(*Tensor).Get(ctx, 0).Item().(float32)
	if !approxEq(got, 12.0, 1e-4) {
		t.Errorf("d(x^3)/dx at x=2 = %f, want 12.0", got)
	}
}

func TestPowBackwardSqrt(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Finish()

	// x=4, x^0.5=2, d(x^0.5)/dx = 0.5 * x^(-0.5) = 0.5/2 = 0.25
	x := ctx.Float(Shape{1}, 4.0)
	y := x.Pow(0.5)
	y.Backward()

	got := x.Grad().(*Tensor).Get(ctx, 0).Item().(float32)
	if !approxEq(got, 0.25, 1e-4) {
		t.Errorf("d(x^0.5)/dx at x=4 = %f, want 0.25", got)
	}
}

func TestPowBackwardMultiElement(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Finish()

	// x=[1,2,3], x^2=[1,4,9], d(x^2)/dx = 2x = [2,4,6]
	x := ctx.FromFloat32(Shape{3}, []float32{1.0, 2.0, 3.0})
	y := x.Pow(2.0)
	y.Backward()

	expected := []float32{2.0, 4.0, 6.0}
	for i, want := range expected {
		got := x.Grad().(*Tensor).Get(ctx, uint32(i)).Item().(float32)
		if !approxEq(got, want, 1e-4) {
			t.Errorf("d(x^2)/dx[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestPowBackwardIdentity(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Finish()

	// x^1, d(x^1)/dx = 1
	x := ctx.Float(Shape{2}, 5.0)
	y := x.Pow(1.0)
	y.Backward()

	for i := range uint32(2) {
		got := x.Grad().(*Tensor).Get(ctx, i).Item().(float32)
		if !approxEq(got, 1.0, 1e-4) {
			t.Errorf("d(x^1)/dx[%d] = %f, want 1.0", i, got)
		}
	}
}

func TestExp(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Float(Shape{2, 2}, 1.0)
	result := a.Exp()

	want := float32(math.E)
	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(i, j).Item().(float32)
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Errorf("Exp[%d,%d] = %f, want %f", i, j, got, want)
			}
		}
	}
}

func TestNegate(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.FromFloat32(Shape{3}, []float32{1.0, -2.0, 3.0})
	result := a.Negate()

	expected := []float32{-1.0, 2.0, -3.0}
	for i, want := range expected {
		got := result.Get(uint32(i)).Item().(float32)
		if !approxEq(got, want, 1e-5) {
			t.Errorf("Negate[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestNegateZeros(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Zeros(Shape{2, 2})
	result := a.Negate()

	for i := range uint32(2) {
		for j := range uint32(2) {
			got := result.Get(i, j).Item().(float32)
			if !approxEq(got, 0.0, 1e-5) {
				t.Errorf("Negate(0)[%d,%d] = %f, want 0.0", i, j, got)
			}
		}
	}
}

func TestNegateBackward(t *testing.T) {
	ctx := New(context.Background(), WithGrad(true))
	defer ctx.Finish()

	// -x, d(-x)/dx = -1
	x := ctx.FromFloat32(Shape{3}, []float32{1.0, -2.0, 3.0})
	y := x.Negate()
	y.Backward()

	for i := range uint32(3) {
		got := x.Grad().(*Tensor).Get(ctx, i).Item().(float32)
		if !approxEq(got, -1.0, 1e-5) {
			t.Errorf("grad[%d] = %f, want -1.0", i, got)
		}
	}
}

func TestExpZero(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Float(Shape{3}, 0.0)
	result := a.Exp()

	for i := range uint32(3) {
		got := result.Get(i).Item().(float32)
		if math.Abs(float64(got)-1.0) > 1e-4 {
			t.Errorf("Exp(0)[%d] = %f, want 1.0", i, got)
		}
	}
}

func TestMean(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.FromFloat32(Shape{2, 3}, []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	result := a.Mean()

	// Mean of [1,2,3,4,5,6] = 21/6 = 3.5
	got := result.Get(0).Item().(float32)
	want := float32(3.5)
	if !approxEq(got, want, 1e-5) {
		t.Errorf("Mean = %f, want %f", got, want)
	}
}

func TestMeanSingleElement(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Float(Shape{1}, 42.0)
	result := a.Mean()

	got := result.Get(0).Item().(float32)
	if !approxEq(got, 42.0, 1e-5) {
		t.Errorf("Mean = %f, want 42.0", got)
	}
}

func TestLog(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.FromFloat32(Shape{2, 2}, []float32{1.0, 2.71828, 10.0, 100.0})
	result := a.Log()

	// ln(1) = 0, ln(e) ≈ 1, ln(10) ≈ 2.303, ln(100) ≈ 4.605
	expected := [][]float32{{0.0, 1.0}, {2.302585, 4.605170}}
	for i := uint32(0); i < 2; i++ {
		for j := uint32(0); j < 2; j++ {
			got := result.Get(i, j).Item().(float32)
			want := expected[i][j]
			if !approxEq(got, want, 1e-4) {
				t.Errorf("Log[%d,%d] = %f, want %f", i, j, got, want)
			}
		}
	}
}

func TestLogOfOne(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.Float(Shape{3}, 1.0)
	result := a.Log()

	for i := range uint32(3) {
		got := result.Get(i).Item().(float32)
		if !approxEq(got, 0.0, 1e-5) {
			t.Errorf("Log(1)[%d] = %f, want 0.0", i, got)
		}
	}
}

// [[1,2,3],[4,5,6]] → max along dim0 → [[4,5,6]] (shape [1,3])
func TestMaxDim0(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := FromFloat32(ctx, Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	result := a.Max(ctx, 0)

	if got, want := shapeOf(result), (Shape{1, 3}); len(got) != len(want) {
		t.Fatalf("Max(dim=0) shape = %v, want %v", got, want)
	} else {
		for i, v := range want {
			if got[i] != v {
				t.Fatalf("Max(dim=0) shape = %v, want %v", got, want)
			}
		}
	}

	expected := []float32{4, 5, 6}
	for j, want := range expected {
		got := result.Get(ctx, 0, uint32(j)).Item().(float32)
		if !approxEq(got, want, 1e-5) {
			t.Errorf("Max(dim=0)[0,%d] = %f, want %f", j, got, want)
		}
	}
}

// [[1,2,3],[4,5,6]] → max along dim1 → [[3],[6]] (shape [2,1])
func TestMaxDim1(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := FromFloat32(ctx, Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	result := a.Max(ctx, 1)

	if got, want := shapeOf(result), (Shape{2, 1}); len(got) != len(want) {
		t.Fatalf("Max(dim=1) shape = %v, want %v", got, want)
	} else {
		for i, v := range want {
			if got[i] != v {
				t.Fatalf("Max(dim=1) shape = %v, want %v", got, want)
			}
		}
	}

	expected := []float32{3, 6}
	for i, want := range expected {
		got := result.Get(ctx, uint32(i), 0).Item().(float32)
		if !approxEq(got, want, 1e-5) {
			t.Errorf("Max(dim=1)[%d,0] = %f, want %f", i, got, want)
		}
	}
}

// Max with no dims reduces everything → scalar shape [1]
func TestMaxGlobal(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := FromFloat32(ctx, Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	result := a.Max(ctx)

	got := result.Get(ctx, 0).Item().(float32)
	if !approxEq(got, 6.0, 1e-5) {
		t.Errorf("Max(global) = %f, want 6.0", got)
	}
}

// WrappedTensor.Max delegates correctly
func TestMaxWrapped(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.FromFloat32(Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	result := a.Max(1)

	expected := []float32{3, 6}
	for i, want := range expected {
		got := result.Get(uint32(i), 0).Item().(float32)
		if !approxEq(got, want, 1e-5) {
			t.Errorf("WrappedTensor Max(dim=1)[%d,0] = %f, want %f", i, got, want)
		}
	}
}

// [[1,2,3],[4,5,6]] → mean along dim0 → [[2.5,3.5,4.5]] (shape [1,3])
func TestMeanDim0(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := FromFloat32(ctx, Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	result := a.Mean(ctx, 0)

	expected := []float32{2.5, 3.5, 4.5}
	for j, want := range expected {
		got := result.Get(ctx, 0, uint32(j)).Item().(float32)
		if !approxEq(got, want, 1e-5) {
			t.Errorf("Mean(dim=0)[0,%d] = %f, want %f", j, got, want)
		}
	}
}

// [[1,2,3],[4,5,6]] → mean along dim1 → [[2.0],[5.0]] (shape [2,1])
func TestMeanDim1(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := FromFloat32(ctx, Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	result := a.Mean(ctx, 1)

	expected := []float32{2.0, 5.0}
	for i, want := range expected {
		got := result.Get(ctx, uint32(i), 0).Item().(float32)
		if !approxEq(got, want, 1e-5) {
			t.Errorf("Mean(dim=1)[%d,0] = %f, want %f", i, got, want)
		}
	}
}

// WrappedTensor.Mean with dim delegates to MeanDim
func TestMeanDimWrapped(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := ctx.FromFloat32(Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	result := a.Mean(1)

	expected := []float32{2.0, 5.0}
	for i, want := range expected {
		got := result.Get(uint32(i), 0).Item().(float32)
		if !approxEq(got, want, 1e-5) {
			t.Errorf("WrappedTensor Mean(dim=1)[%d,0] = %f, want %f", i, got, want)
		}
	}
}
