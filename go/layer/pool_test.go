package layer

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestMaxPoolingBackwardRoutesGradToMaxima(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	layer := MaxPooling(shapes.Shape{2, 2}, 2)
	x := shapes.FromFloat32(ctx, shapes.Shape{1, 1, 4, 4}, []float32{
		1, 3, 2, 1,
		4, 6, 5, 2,
		7, 8, 9, 3,
		0, 1, 2, 4,
	})

	out := layer.Forward(ctx, x)
	out.Backward(ctx)

	gotDX := x.Grad().(shapes.Tensor).Values().([]float32)
	wantDX := []float32{
		0, 0, 0, 0,
		0, 1, 1, 0,
		0, 1, 1, 0,
		0, 0, 0, 0,
	}
	for i := range wantDX {
		if math.Abs(float64(gotDX[i]-wantDX[i])) > 1e-5 {
			t.Fatalf("xGrad[%d]=%f want %f", i, gotDX[i], wantDX[i])
		}
	}
}

func TestAdaptiveAvgPoolBackwardDistributesGradEvenly(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	layer := AdaptiveAvgPool(shapes.Shape{2, 2})
	x := shapes.FromFloat32(ctx, shapes.Shape{1, 1, 4, 4}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	})

	out := layer.Forward(ctx, x)
	out.Backward(ctx)

	gotDX := x.Grad().(shapes.Tensor).Values().([]float32)
	wantDX := []float32{
		0.25, 0.25, 0.25, 0.25,
		0.25, 0.25, 0.25, 0.25,
		0.25, 0.25, 0.25, 0.25,
		0.25, 0.25, 0.25, 0.25,
	}
	for i := range wantDX {
		if math.Abs(float64(gotDX[i]-wantDX[i])) > 1e-5 {
			t.Fatalf("xGrad[%d]=%f want %f", i, gotDX[i], wantDX[i])
		}
	}
}
