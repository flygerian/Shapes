package shapes

import (
	"context"
	"testing"
)

func TestWrappedTensorShape(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	tensor := Float(ctx, Shape{2, 3}, 1.0)
	wt := &WrappedTensor{tensor: tensor, context: ctx}

	shape := wt.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("expected shape [2,3], got %v", shape)
	}
}

func TestWrappedTensorAddInPlace(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Float(ctx, Shape{2}, 1.0)
	b := Float(ctx, Shape{2}, 2.0)
	wa := &WrappedTensor{tensor: a, context: ctx}
	wb := &WrappedTensor{tensor: b, context: ctx}

	wa.AddInPlace(wb)

	for i := range uint32(2) {
		got := a.Get(ctx, i).Item().(float32)
		if got != 3.0 {
			t.Fatalf("a[%d] = %f, want 3.0", i, got)
		}
	}
}

func TestWrappedTensorPlusScalar(t *testing.T) {
	ctx := New(context.Background())
	defer ctx.Finish()

	a := Float(ctx, Shape{2}, 1.5)
	wa := &WrappedTensor{tensor: a, context: ctx}
	result := wa.Plus(1.0)

	for i := range uint32(2) {
		got := result.tensor.Get(ctx, i).Item().(float32)
		if got != 2.5 {
			t.Fatalf("result[%d] = %f, want 2.5", i, got)
		}
	}
}
