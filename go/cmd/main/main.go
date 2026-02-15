package main

import (
	"context"
	"fmt"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/layer"
	"github.com/flygerian/shapes/loss"
	"github.com/flygerian/shapes/tensor"
	"github.com/flygerian/shapes/visual"
)

func main() {
	ctx := shapes.New(context.Background(), shapes.WithGrad(true))
	defer ctx.Close()

	xs := tensor.FromFloat32(
		ctx,
		tensor.Shape{4, 3},
		[]float32{2.0, 3.0, -1.0, 3.0, -1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, -1.0},
	)

	ys := tensor.FromFloat32(ctx, tensor.Shape{4}, []float32{1.0, -1.0, -1.0, 1.0})

	dense := layer.Dense(3, 10)
	dense2 := layer.Dense(10, 120)
	dens3 := layer.Dense(120, 1)
	denseOutput := dense(ctx, xs)
	denseOutput.Label = "dense1"

	denseOutput2 := dense2(ctx, denseOutput)
	denseOutput2.Label = "dense2"

	logits := dens3(ctx, denseOutput2)
	logits.Label = "logits"

	l := loss.Mse(ctx, ys, logits.Squeeze(ctx))
	l.Backward(ctx)

	visual.Visualize(l)

	fmt.Printf("Output: \n")
	visual.Print(logits)

	fmt.Printf("Expected \n")
	visual.Print(ys)

	fmt.Printf("Loss \n")
	visual.Print(l)

}
