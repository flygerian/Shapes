package shapesnn

import "../"
import "../olib/"
import "core:fmt"
import "core:math"

denseLayer :: struct {}

Dense :: proc(inputSize: uint, outputSize: uint, withBias: bool) -> LayerWithState {
	ctx := cast(^shapes.Context)context.user_ptr
	fmt.println("%v", ctx.memory)

	initVal := (5.0 / 3.0) / math.pow(f32(inputSize), 0.5)
	layer := LayerWithState {
		weights = shapes.MakeRandomTensor(
			ctx,
			shapes.Shape2D(outputSize, inputSize),
			-initVal,
			initVal,
			.F32,
		),
	}

	if withBias {
		layer.bias = shapes.MakeRandomTensor(ctx, shapes.Shape1D(outputSize), -0.1, 0.1, .F32)
	}

	return layer
}

denseForward :: proc(layer: ^LayerWithState, x: ^shapes.Tensor) -> (shapes.Tensor, shapes.Result) {
	ctx := cast(^shapes.Context)context.user_ptr
	if layer == nil || x == nil {
		return shapes.Tensor{}, .ERR_NULL_TENSOR_PROVIDED
	}

	out := shapes.DenseLinear(ctx, x, &layer.weights, &layer.bias, layer.withBias)

	arrSize := 4 if layer.withBias else 3
	out.inputs = shapes.MakeTensorArray(ctx.memory, uint(arrSize))

	shapes.ArrayAppendTensor(out.inputs, x)
	shapes.ArrayAppendTensor(out.inputs, &layer.weights)

	if layer.withBias != true {
		shapes.ArrayAppendTensor(out.inputs, &layer.bias)
	}

	out.opMetadata = rawptr(layer)
	out.opType = .OP_DENSE
	return out, .OK
}

denseBackward :: proc() {
	ctx := cast(^shapes.Context)context.user_ptr
}

