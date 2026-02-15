package loss

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/tensor"
)

func approxEq(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) < tol
}

func TestMse(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	yGround := tensor.FromFloat32(ctx, tensor.Shape{4}, []float32{1.0, -1.0, -1.0, 1.0})
	yPred := tensor.FromFloat32(ctx, tensor.Shape{4}, []float32{0.5, -0.5, -0.8, 0.9})

	loss := Mse(ctx, yGround, yPred)

	shape := tensor.ShapeOf(loss)
	if len(shape) != 1 || shape[0] != 4 {
		t.Fatalf("expected shape [4], got %v", shape)
	}

	// MSE per element: (pred - ground)^2
	// (0.5-1)^2=0.25, (-0.5-(-1))^2=0.25, (-0.8-(-1))^2=0.04, (0.9-1)^2=0.01
	expected := []float32{0.25, 0.25, 0.04, 0.01}
	for i, want := range expected {
		got, err := loss.GetF32(uint32(i))
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if !approxEq(got, want, 1e-4) {
			t.Errorf("Mse[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestMsePerfectPrediction(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	y := tensor.FromFloat32(ctx, tensor.Shape{3}, []float32{1.0, 2.0, 3.0})

	loss := Mse(ctx, y, y)

	for i := range uint32(3) {
		got, err := loss.GetF32(i)
		if err != nil {
			t.Fatalf("GetF32(%d): %v", i, err)
		}
		if !approxEq(got, 0.0, 1e-6) {
			t.Errorf("Mse[%d] = %f, want 0.0", i, got)
		}
	}
}

func TestMse2D(t *testing.T) {
	ctx := shapes.New(context.Background())
	defer ctx.Close()

	yGround := tensor.FromFloat32(ctx, tensor.Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	yPred := tensor.FromFloat32(ctx, tensor.Shape{2, 3}, []float32{2, 2, 2, 2, 2, 2})

	loss := Mse(ctx, yGround, yPred)

	shape := tensor.ShapeOf(loss)
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("expected shape [2,3], got %v", shape)
	}

	// (pred - ground)^2: (2-1)^2=1, (2-2)^2=0, (2-3)^2=1, (2-4)^2=4, (2-5)^2=9, (2-6)^2=16
	expected := [][]float32{
		{1.0, 0.0, 1.0},
		{4.0, 9.0, 16.0},
	}
	for i := range uint32(2) {
		for j := range uint32(3) {
			got, err := loss.GetF32(i, j)
			if err != nil {
				t.Fatalf("GetF32(%d,%d): %v", i, j, err)
			}
			if !approxEq(got, expected[i][j], 1e-4) {
				t.Errorf("Mse[%d,%d] = %f, want %f", i, j, got, expected[i][j])
			}
		}
	}
}

func TestMseBackward(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	// Scalar case: pred=1.5, ground=1.0 => loss=(0.5)^2=0.25
	// d(loss)/d(pred) = 2*(pred-ground) = 1.0
	yGround := tensor.Float(ctx, tensor.Shape{1}, 1.0)
	yPred := tensor.Float(ctx, tensor.Shape{1}, 1.5)

	loss := Mse(ctx, yGround, yPred)
	loss.Backward(ctx)

	predGrad := yPred.Grad()
	if predGrad == nil {
		t.Fatal("expected gradient on yPred")
	}

	got, err := predGrad.GetF32(0)
	if err != nil {
		t.Fatalf("GetF32(0): %v", err)
	}
	if !approxEq(got, 1.0, 1e-4) {
		t.Errorf("d(loss)/d(yPred) = %f, want 1.0", got)
	}
}
