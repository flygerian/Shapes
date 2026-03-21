package shapes

import (
	stdctx "context"
	"math"
	"testing"
)

func TestConv2dForwardSingleBatchSingleChannel(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 3, 3, 1}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	kernels := FromFloat32(ctx, Shape{1, 1, 2, 2}, []float32{
		1, 0,
		0, 1,
	})

	out, _ := Conv2d(ctx, x, kernels, 1)
	wantShape := Shape{1, 2, 2, 1}
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
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{2, 3, 3, 1}, []float32{
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

	out, _ := Conv2d(ctx, x, kernels, 1)
	values := out.Values().([]float32)
	want := []float32{6, 8, 12, 14, 24, 26, 30, 32}
	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 1e-5 {
			t.Fatalf("out[%d]=%f want %f", i, values[i], want[i])
		}
	}
}

func TestConv2dPanicsOnInvalidKernelShape(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 3, 3, 1}, []float32{
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
	_, _ = Conv2d(ctx, x, invalidKernels, 1)
}

func TestConv2dBackwardSingleChannel(t *testing.T) {
	ctx := New(stdctx.Background(), WithGrad(true))
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 3, 3, 1}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	kernels := FromFloat32(ctx, Shape{1, 1, 2, 2}, []float32{
		1, 0,
		0, 1,
	})
	gradOut := FromFloat32(ctx, Shape{1, 2, 2, 1}, []float32{
		1, 1,
		1, 1,
	})

	_, colBuffer := Conv2d(ctx, x, kernels, 1)
	Conv2dBackward(ctx, x, kernels, gradOut, colBuffer, 1)

	wantDX := []float32{
		1, 1, 0,
		1, 2, 1,
		0, 1, 1,
	}
	gotDX := x.Grad().(Tensor).Values().([]float32)
	for i := range wantDX {
		if math.Abs(float64(gotDX[i]-wantDX[i])) > 1e-5 {
			t.Fatalf("dX[%d]=%f want %f", i, gotDX[i], wantDX[i])
		}
	}

	wantDK := []float32{12, 16, 24, 28}
	gotDK := kernels.Grad().(Tensor).Values().([]float32)
	for i := range wantDK {
		if math.Abs(float64(gotDK[i]-wantDK[i])) > 1e-5 {
			t.Fatalf("dKernels[%d]=%f want %f", i, gotDK[i], wantDK[i])
		}
	}
}

func TestConvTranspose2dForwardSingleBatchSingleChannel(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 2, 2, 1}, []float32{
		1, 2,
		3, 4,
	})
	kernels := FromFloat32(ctx, Shape{1, 1, 2, 2}, []float32{
		1, 0,
		0, 1,
	})

	out := ConvTranspose2d(ctx, x, kernels, 1)
	values := out.Values().([]float32)
	want := []float32{
		1, 2, 0,
		3, 5, 2,
		0, 3, 4,
	}
	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 1e-5 {
			t.Fatalf("out[%d]=%f want %f", i, values[i], want[i])
		}
	}
}

func TestConvTranspose2dBackwardSingleChannel(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 2, 2, 1}, []float32{
		1, 2,
		3, 4,
	})
	kernels := FromFloat32(ctx, Shape{1, 1, 2, 2}, []float32{
		1, 0,
		0, 1,
	})
	gradOut := FromFloat32(ctx, Shape{1, 3, 3, 1}, []float32{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	})

	dX, dKernels := ConvTranspose2dBackward(ctx, x, kernels, gradOut, 1)
	wantDX := []float32{2, 2, 2, 2}
	gotDX := dX.Values().([]float32)
	for i := range wantDX {
		if math.Abs(float64(gotDX[i]-wantDX[i])) > 1e-5 {
			t.Fatalf("dX[%d]=%f want %f", i, gotDX[i], wantDX[i])
		}
	}

	wantDK := []float32{10, 10, 10, 10}
	gotDK := dKernels.Values().([]float32)
	for i := range wantDK {
		if math.Abs(float64(gotDK[i]-wantDK[i])) > 1e-5 {
			t.Fatalf("dKernels[%d]=%f want %f", i, gotDK[i], wantDK[i])
		}
	}
}

func TestMaxPool2dForward(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 4, 4, 1}, []float32{
		1, 3, 2, 1,
		4, 6, 5, 2,
		7, 8, 9, 3,
		0, 1, 2, 4,
	})

	out := MaxPool2d(ctx, x, Shape{2, 2}, 2)
	values := out.Values().([]float32)
	want := []float32{6, 5, 8, 9}
	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 1e-5 {
			t.Fatalf("out[%d]=%f want %f", i, values[i], want[i])
		}
	}
}

func TestMaxPool2dBackward(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 4, 4, 1}, []float32{
		1, 3, 2, 1,
		4, 6, 5, 2,
		7, 8, 9, 3,
		0, 1, 2, 4,
	})
	gradOut := FromFloat32(ctx, Shape{1, 2, 2, 1}, []float32{
		1, 2,
		3, 4,
	})

	dX := MaxPool2dBackward(ctx, x, gradOut, Shape{2, 2}, 2)
	gotDX := dX.Values().([]float32)
	wantDX := []float32{
		0, 0, 0, 0,
		0, 1, 2, 0,
		0, 3, 4, 0,
		0, 0, 0, 0,
	}
	for i := range wantDX {
		if math.Abs(float64(gotDX[i]-wantDX[i])) > 1e-5 {
			t.Fatalf("dX[%d]=%f want %f", i, gotDX[i], wantDX[i])
		}
	}
}

func TestAdaptiveAvgPool2dForward(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 4, 4, 1}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	})

	out := AdaptiveAvgPool2d(ctx, x, Shape{2, 2})
	values := out.Values().([]float32)
	want := []float32{3.5, 5.5, 11.5, 13.5}
	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 1e-5 {
			t.Fatalf("out[%d]=%f want %f", i, values[i], want[i])
		}
	}
}

func TestAdaptiveAvgPool2dBackward(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 4, 4, 1}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	})
	gradOut := FromFloat32(ctx, Shape{1, 2, 2, 1}, []float32{
		1, 2,
		3, 4,
	})

	dX := AdaptiveAvgPool2dBackward(ctx, x, gradOut, Shape{2, 2})
	gotDX := dX.Values().([]float32)
	wantDX := []float32{
		0.25, 0.25, 0.5, 0.5,
		0.25, 0.25, 0.5, 0.5,
		0.75, 0.75, 1.0, 1.0,
		0.75, 0.75, 1.0, 1.0,
	}
	for i := range wantDX {
		if math.Abs(float64(gotDX[i]-wantDX[i])) > 1e-5 {
			t.Fatalf("dX[%d]=%f want %f", i, gotDX[i], wantDX[i])
		}
	}
}

func TestAdaptiveAvgPool2dBackwardMaterializesNonContiguousGrad(t *testing.T) {
	ctx := New(stdctx.Background())
	defer ctx.Finish()

	x := FromFloat32(ctx, Shape{1, 4, 4, 1}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	})
	gradBase := FromFloat32(ctx, Shape{1, 2, 2, 1}, []float32{
		1, 3,
		2, 4,
	})
	gradOut := gradBase.Transpose(ctx, 1, 2)

	dX := AdaptiveAvgPool2dBackward(ctx, x, gradOut, Shape{2, 2})
	gotDX := dX.Values().([]float32)
	wantDX := []float32{
		0.25, 0.25, 0.5, 0.5,
		0.25, 0.25, 0.5, 0.5,
		0.75, 0.75, 1.0, 1.0,
		0.75, 0.75, 1.0, 1.0,
	}
	for i := range wantDX {
		if math.Abs(float64(gotDX[i]-wantDX[i])) > 1e-5 {
			t.Fatalf("dX[%d]=%f want %f", i, gotDX[i], wantDX[i])
		}
	}
}
