package examples

import (
	"github.com/flygerian/shapes"
)

func MakeMore_1(ctx *shapes.MainContext) {

	// xs := ctx.FromFloat32(
	// 	shapes.Shape{4, 3},
	// 	[]float32{2.0, 3.0, -1.0, 3.0, -1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, -1.0},
	// )
	//
	// ys := ctx.FromFloat32(shapes.Shape{4}, []float32{1.0, -1.0, -1.0, 1.0})
	//
	// dense := layer.Dense(ctx, 3, 10)
	// dens3 := layer.Dense(ctx, 10, 1)
	//
	// noGraph := ctx.NoGraph()
	// sgd := optimizer.SGD(noGraph, 0.01)
	//
	// var l *shapes.Tensor
	// var computationGraph *shapes.ComputationGraph
	// var logits *shapes.WrappedTensor
	//
	// for range 20 {
	// 	denseOutput := dense(xs)
	// 	denseOutput.Tensor().Label = "dense1"
	//
	// 	logits = dens3(denseOutput)
	// 	logits.Tensor().Label = "logits"
	//
	// 	lossTensor := loss.Mse(ys, logits.Squeeze())
	// 	l = lossTensor.Tensor()
	// 	computationGraph = l.Backward(ctx)
	//
	// 	fmt.Printf("Loss \n")
	// 	// visual.Print(l)
	//
	// 	sgd(computationGraph)
	// 	optimizer.ZeroGrad(noGraph, computationGraph)
	// }
	//
	// // visual.Visualize(l)
	//
	// parameters := extract.Parameters(computationGraph)
	// fmt.Println("Model parameters: ", len(parameters))
	//
	// fmt.Printf("Output: \n")
	// // visual.Print(ctx, logits.Tensor())
	//
	// fmt.Printf("Expected \n")
	// visual.Print(ctx, ys.Tensor())
}
