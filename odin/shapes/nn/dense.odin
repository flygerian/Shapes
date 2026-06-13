package shapesnn

import "../"
import "core:fmt"
import "core:math"

denseLayer :: struct {}

Dense :: proc(inputSize: uint, outputSize: uint, withBias: bool) -> LayerWithState {
	initVal := (5.0 / 3.0) / math.pow(f32(inputSize), 0.5)
	weights := new(shapes.Tensor)
	weights^ = shapes.MakeRandomTensor(-initVal, initVal, shapes.Shape2D(outputSize, inputSize))

	layer := LayerWithState {
		weights = weights,
	}

	if withBias {
		layer.bias = new(shapes.Tensor)
		layer.bias^ = shapes.MakeRandomTensor(-0.1, 0.1, shapes.Shape1D(outputSize))
	}

	return layer
}

denseForward :: proc(layer: ^LayerWithState, x: ^shapes.Tensor) -> shapes.Tensor {
	if layer == nil {
		return shapes.Tensor{}
	}

	out := shapes.DenseForward(x, layer.weights, layer.bias, layer.withBias)

	arrSize := 4 if layer.withBias else 3
	out.inputs = make([]^shapes.Tensor, arrSize)

	out.inputs[0] = x
	out.inputs[1] = layer.weights

	if layer.withBias != true {
		out.inputs[1] = layer.bias
	}

	out.opMetadata = layer
	out.opType = .OP_DENSE
	return out
}

denseBackward :: proc(tensor: ^shapes.Tensor) {
	ctx := cast(^shapes.Context)context.user_ptr

	if ctx == nil || tensor.inputs == nil || tensor.grad == nil {
		panic("Null tensors provided")
	}

	if tensor.opType != .OP_DENSE {
		panic("Node passed to backward is not from a dense layer")
	}

	input := tensor.inputs[0]
	weights := tensor.inputs[1]
	bias: ^shapes.Tensor

	layerData := tensor.opMetadata.(^LayerWithState)

	if layerData.withBias && len(tensor.inputs) == 3 {
		bias = tensor.inputs[2]
	}

	biasGrad := layerData.withBias ? bias.grad : nil
	result := shapes.DenseBackward(input, weights, tensor.grad, input.grad, weights.grad, biasGrad)
	if result != .OK {
		panic("DenseBackward failed")
	}
}

