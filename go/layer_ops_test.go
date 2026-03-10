package shapes

import (
	stdctx "context"
	"math"
	"testing"
)

func TestConv2dForwardSingleBatchSingleChannel(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 1, 3, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	kernels := FromFloat32(ctx, Shape{1, 1, 2, 2}, []float32{
		1, 0,
		0, 1,
	})

	out := Conv2d(ctx, x, kernels, 1)
	wantShape := Shape{1, 1, 2, 2}
	if got := out.Shape(); len(got) != len(wantShape) ||
		got[0] != wantShape[0] || got[1] != wantShape[1] || got[2] != wantShape[2] || got[3] != wantShape[3] {
		t.Fatalf("shape mismatch: got %v want %v", got, wantShape)
	}

	values := out.Values().([]float32)
	want := []float32{6, 8, 12, 14}
	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 1e-5 {
			t.Fatalf("out[%d]=%f want %f", i, values[i], want[i])
		}
	}
}

func TestConv2dForwardBatchDimension(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{2, 1, 3, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		13, 14, 15,
		16, 17, 18,
	})
	kernels := FromFloat32(ctx, Shape{1, 1, 2, 2}, []float32{
		1, 0,
		0, 1,
	})

	out := Conv2d(ctx, x, kernels, 1)
	values := out.Values().([]float32)
	want := []float32{6, 8, 12, 14, 24, 26, 30, 32}
	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 1e-5 {
			t.Fatalf("out[%d]=%f want %f", i, values[i], want[i])
		}
	}
}

func TestConv2dPanicsOnInvalidKernelShape(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 1, 3, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})

	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for invalid kernel shape")
		}
	}()
	invalidKernels := FromFloat32(ctx, Shape{1, 2, 2}, []float32{
		1, 0,
		0, 1,
	})
	_ = Conv2d(ctx, x, invalidKernels, 1)
}

func TestConv2dBackwardSingleChannel(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 1, 3, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	kernels := FromFloat32(ctx, Shape{1, 1, 2, 2}, []float32{
		1, 0,
		0, 1,
	})
	gradOut := FromFloat32(ctx, Shape{1, 1, 2, 2}, []float32{
		1, 1,
		1, 1,
	})

	dX, dKernels := Conv2dBackward(ctx, x, kernels, gradOut, 1)

	wantDX := []float32{
		1, 1, 0,
		1, 2, 1,
		0, 1, 1,
	}
	gotDX := dX.Values().([]float32)
	for i := range wantDX {
		if math.Abs(float64(gotDX[i]-wantDX[i])) > 1e-5 {
			t.Fatalf("dX[%d]=%f want %f", i, gotDX[i], wantDX[i])
		}
	}

	wantDK := []float32{12, 16, 24, 28}
	gotDK := dKernels.Values().([]float32)
	for i := range wantDK {
		if math.Abs(float64(gotDK[i]-wantDK[i])) > 1e-5 {
			t.Fatalf("dKernels[%d]=%f want %f", i, gotDK[i], wantDK[i])
		}
	}
}
