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

	layer := MaxPooling(ctx, shapes.Shape{2, 2}, 2)
	x := shapes.FromFloat32(ctx, shapes.Shape{1, 4, 4, 1}, []float32{
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

func TestMaxPoolingBackwardMixedCpuTargetAndCudaForward(t *testing.T) {
	cpuCtx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer cpuCtx.Finish()
	cudaCtx := shapes.New(context.Background(), shapes.WithGrad(true), shapes.WithCuda())
	defer cudaCtx.Finish()

	layer := MaxPooling(cudaCtx, shapes.Shape{2, 2}, 2)
	x := shapes.Float(cudaCtx, shapes.Shape{8, 4, 4, 256}, 1.0)

	out := layer.Forward(cudaCtx, x)
	out.Backward(cpuCtx)

	gotDX := x.Grad().(shapes.Tensor).Values().([]float32)
	if len(gotDX) != 8*256*4*4 {
		t.Fatalf("unexpected grad length %d", len(gotDX))
	}

	for batch := range uint(8) {
		for channel := range uint(256) {
			base := int(batch*4*4*256 + channel)
			want := []float32{
				1, 0, 1, 0,
				0, 0, 0, 0,
				1, 0, 1, 0,
				0, 0, 0, 0,
			}
			for i, expected := range want {
				valueIdx := base + (i/4)*4*256 + (i%4)*256
				if math.Abs(float64(gotDX[valueIdx]-expected)) > 1e-5 {
					t.Fatalf("grad[%d,%d,%d]=%f want %f", batch, channel, i, gotDX[valueIdx], expected)
				}
			}
		}
	}
}

func TestAdaptiveAvgPoolBackwardDistributesGradEvenly(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	layer := AdaptiveAvgPool(ctx, shapes.Shape{2, 2})
	x := shapes.FromFloat32(ctx, shapes.Shape{1, 4, 4, 1}, []float32{
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
