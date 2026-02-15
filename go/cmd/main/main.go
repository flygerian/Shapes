package main

import (
	"context"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/layer"
	"github.com/flygerian/shapes/tensor"
	"github.com/flygerian/shapes/visual"
)

func main() {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	x := tensor.FromFloat32(ctx, tensor.Shape{3}, []float32{2.0, 3.0, -1.0})
	x.Label = "x"
	dense := layer.Dense(3, 10)
	denseOutput := dense(ctx, x)
	denseOutput.Label = "dense"

	denseOutput.Backward(ctx)

	visual.Visualize(denseOutput)
}
