package main

import (
	"context"
	"fmt"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/extract"
	"github.com/flygerian/shapes/layer"
	"github.com/flygerian/shapes/loss"
	"github.com/flygerian/shapes/optimizer"
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
	dens3 := layer.Dense(10, 1)

	noGraph := ctx.NoGraph()
	sgd := optimizer.SGD(noGraph, 0.01)

	var l *tensor.Tensor
	var computationGraph *tensor.ComputationGraph
	var logits *tensor.Tensor

	for range 20 {
		denseOutput := dense(ctx, xs)
		denseOutput.Label = "dense1"

		logits = dens3(ctx, denseOutput)
		logits.Label = "logits"

		l = loss.Mse(ctx, ys, logits.Squeeze(ctx))
		computationGraph = l.Backward(ctx)

		fmt.Printf("Loss \n")
		visual.Print(l)

		sgd(computationGraph)
		optimizer.ZeroGrad(noGraph, computationGraph)
	}

	// visual.Visualize(l)

	parameters := extract.Parameters(computationGraph)
	fmt.Println("Model parameters: ", len(parameters))

	fmt.Printf("Output: \n")
	visual.Print(logits)

	fmt.Printf("Expected \n")
	visual.Print(ys)

}
