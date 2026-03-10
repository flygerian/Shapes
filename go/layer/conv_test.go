package layer

import (
	"context"
	"math"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestConvBackwardAccumulatesInputKernelAndBiasGrads(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	layer := Conv2d(ctx, 1, 1, shapes.Shape{2, 2}, 1).(*conv)
	layer.kernels = shapes.FromFloat32(ctx, shapes.Shape{1, 1, 2, 2}, []float32{
		1, 0,
		0, 1,
	})
	layer.bias = shapes.Float(ctx, shapes.Shape{1}, 0)

	x := shapes.FromFloat32(ctx, shapes.Shape{1, 1, 3, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})

	out := layer.Forward(ctx, x)
	out.Backward(ctx)

	gotDX := x.Grad().(shapes.Tensor).Values().([]float32)
	wantDX := []float32{
		1, 1, 0,
		1, 2, 1,
		0, 1, 1,
	}
	for i := range wantDX {
		if math.Abs(float64(gotDX[i]-wantDX[i])) > 1e-5 {
			t.Fatalf("xGrad[%d]=%f want %f", i, gotDX[i], wantDX[i])
		}
	}

	gotDK := layer.kernels.Grad().(shapes.Tensor).Values().([]float32)
	wantDK := []float32{12, 16, 24, 28}
	for i := range wantDK {
		if math.Abs(float64(gotDK[i]-wantDK[i])) > 1e-5 {
			t.Fatalf("kernelGrad[%d]=%f want %f", i, gotDK[i], wantDK[i])
		}
	}

	gotDB := layer.bias.Grad().(shapes.Tensor).Values().([]float32)
	if len(gotDB) != 1 || math.Abs(float64(gotDB[0]-4.0)) > 1e-5 {
		t.Fatalf("biasGrad=%v want [4]", gotDB)
	}
}
