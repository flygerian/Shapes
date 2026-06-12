package shapesnn
import "../"

Tensor :: struct {
	using tensor: shapes.Tensor,
	inputs:       []^Tensor,
	opType:       shapes.Optype,
	opMetadata:   Layer,
}

LayerWithState :: struct {
	weights:   Tensor,
	bias:      Tensor,
	withBias:  bool,
	layerData: any,
}

LayerWithNoState :: struct {}

Layer :: union {
	LayerWithNoState,
	LayerWithState,
}

FromRawTensor :: proc(rawTensor: shapes.Raw_Tensor) -> Tensor {
	t := shapes.FromRawTensor(rawTensor)
	return Tensor{tensor = t}
}

ToRawTensor :: proc(t: Tensor) -> shapes.Raw_Tensor {
	return shapes.ToRawTensor(t.tensor)
}

doDenseOp :: proc(x: Tensor, w: Tensor, bias: Tensor, withBias: bool) -> Tensor {
	ctx := cast(^shapes.Context)context.user_ptr
	_x := ToRawTensor(x)
	_w := ToRawTensor(w)
	_bias := ToRawTensor(bias)
	result := shapes._DenseLinear(ctx, &_x, &_w, &_bias, withBias)
	return FromRawTensor(result)
}

Forward :: proc {
	denseForward,
}

DenseBackward :: proc(
	x: ^Tensor,
	w: ^Tensor,
	gradOut: ^Tensor,
	dX: ^Tensor,
	dW: ^Tensor,
	dB: ^Tensor,
) -> shapes.Result {
	ctx := cast(^shapes.Context)context.user_ptr
	_x := shapes.ToRawTensor(x^)
	_w := shapes.ToRawTensor(w^)
	_gradOut := shapes.ToRawTensor(gradOut^)
	_dX := shapes.ToRawTensor(dX^)
	_dW := shapes.ToRawTensor(dW^)
	_dB := shapes.ToRawTensor(dB^)

	result := shapes._DenseBackward(ctx, &_x, &_w, &_gradOut, &_dX, &_dW, &_dB)
	return result
}

