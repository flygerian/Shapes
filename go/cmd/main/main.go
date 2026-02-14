package main

import (
	"context"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/activation"
	"github.com/flygerian/shapes/tensor"
	"github.com/flygerian/shapes/visual"
)

func main() {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	x1 := tensor.Float(ctx, []uint32{1}, 2.0)
	x2 := tensor.Float(ctx, []uint32{1}, 0.0)

	w1 := tensor.Float(ctx, []uint32{1}, -3.0)
	w2 := tensor.Float(ctx, []uint32{1}, 1.0)

	b := tensor.Float(ctx, []uint32{1}, 6.8813735870195432)

	x1w1 := x1.Times(ctx, w1)
	x2w2 := x2.Times(ctx, w2)

	n := x1w1.Plus(ctx, x2w2).Plus(ctx, b)
	o := activation.Tanh(ctx, n)

	o.Backward(ctx)

	visual.Visualize(o)
}
