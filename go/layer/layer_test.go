package layer

import (
	"context"
	"testing"

	shapes "github.com/flygerian/shapes"
)

type captureLayer struct {
	input shapes.Tensor
}

func (l *captureLayer) Forward(_ shapes.Context, x shapes.Tensor) shapes.Tensor {
	l.input = x
	return x
}

func TestSequentialPassesOriginalInputToFirstLayer(t *testing.T) {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Finish()

	x := shapes.Float(ctx, shapes.Shape{2, 3}, 1.0)
	first := &captureLayer{}
	seq := NewSequential(ctx, first)

	out := seq.Forward(ctx, x)

	if first.input == nil {
		t.Fatal("expected first layer to receive input")
	}
	if first.input.UnsafeCTensor() != x.UnsafeCTensor() {
		t.Fatal("expected sequential to pass original tensor to first layer")
	}
	if out.UnsafeCTensor() != x.UnsafeCTensor() {
		t.Fatal("expected sequential to return original tensor when layer is identity")
	}
}
