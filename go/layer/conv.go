package layer

import (
	"math"

	"github.com/flygerian/shapes"
)

type conv struct {
	kernelShape shapes.Shape
	stride      uint8
	inChannels  uint
	outChannels uint

	kernels       shapes.Tensor
	bias          shapes.Tensor
	isBiasApplied bool
}

type convMetadata struct {
	stride    uint8
	colBuffer shapes.Tensor
}

func Conv2d(
	ctx shapes.Context,
	inChannels uint,
	outChannels uint,
	kernel shapes.Shape,
	stride uint8,
	options ...layerOption,
) Layer {
	if inChannels == 0 || outChannels == 0 {
		panic("Cannot have in/out channnels as 0")
	}
	if len(kernel) != 2 {
		panic("Conv2d kernel shape should be 2D")
	}

	for _, s := range kernel {
		if s == 0 {
			panic("No dim of the kernel should be zero")
		}
	}

	initialization := (5 / 3) / (math.Pow(float64(inChannels), 0.5))
	kernels := shapes.FloatRandom(
		ctx,
		shapes.Shape{uint(outChannels), uint(inChannels), kernel[0], kernel[1]},
		-float32(initialization), float32(initialization),
	)
	bias := shapes.Float(ctx, shapes.Shape{1, uint(outChannels), 1, 1}, 0)

	state := conv{
		kernelShape: kernel,
		inChannels:  inChannels,
		outChannels: outChannels,

		kernels: kernels,
		stride:  stride,
		bias:    bias,
	}

	for _, opt := range options {
		opt(&state)
	}

	return &state
}

func (c *conv) SetBiasEnabled(isBiasEnabled bool) {
	c.isBiasApplied = isBiasEnabled
}

func (c *conv) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	forwardCtx := ctx.Forward(
		shapes.WithInputs(x),
		shapes.WithHiddenState(c.kernels, c.bias),
		shapes.WithOpType(shapes.OpConv),
	)

	out, colBuffer := shapes.Conv2d(forwardCtx, x, c.kernels, c.stride)

	out = out.Plus(forwardCtx, c.bias)

	forwardCtx.Finish(
		shapes.WithResult(out),
		shapes.WithBackward(convBackward),
		shapes.WithMetadata(convMetadata{stride: c.stride, colBuffer: colBuffer}),
	)

	return out
}

func convBackward(ctx shapes.Context, out shapes.ComputationGraphNode) {
	backwardCtx := ctx.Backward()
	defer backwardCtx.Finish()

	x := out.Inputs()[0]
	hidden := out.HiddenState()
	kernels := hidden[0]
	bias := hidden[1]
	meta := out.Metadata().(convMetadata)

	dOutput := out.Grad() // (B, C, oH, oW)
	outputShape := dOutput.Shape()
	B := outputShape[0]
	C := outputShape[1]
	oh := outputShape[2]
	ow := outputShape[3]

	// Collapse spatial dims first so bias reduction is stable even when Sum squeezes singleton dims.
	dBias := dOutput.
		Reshape(backwardCtx, int(B), int(C), int(oh*ow)).
		Sum(backwardCtx, 2).
		Sum(backwardCtx, 0).
		Squeeze(backwardCtx)

	bias.Grad().Accumulate(backwardCtx, dBias)

	shapes.Conv2dBackward(backwardCtx, x, kernels, dOutput.(shapes.Tensor), meta.colBuffer, meta.stride)
}
