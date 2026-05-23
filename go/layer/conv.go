package layer

import (
	"math"

	"github.com/flygerian/shapes"
)

type conv struct {
	ctx         shapes.Context
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
	withBias  bool
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
	bias := shapes.Float(ctx, shapes.Shape{1, 1, 1, uint(outChannels)}, 0)

	state := conv{
		ctx:           ctx,
		kernelShape:   kernel,
		inChannels:    inChannels,
		outChannels:   outChannels,
		isBiasApplied: true,

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

	out, colBuffer := shapes.Conv2d(forwardCtx, x, c.kernels, c.bias, c.isBiasApplied, c.stride)

	forwardCtx.Finish(
		shapes.WithResult(out),
		shapes.WithBackward(convBackward),
		shapes.WithMetadata(convMetadata{stride: c.stride, colBuffer: colBuffer, withBias: c.isBiasApplied}),
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

	ctx.Mark(meta.colBuffer)

	dOutput := out.Grad() // (B, oH, oW, C)
	var dBias shapes.Tensor
	if meta.withBias {
		dBias = bias.Grad().(shapes.Tensor)
	}

	shapes.Conv2dBackward(backwardCtx, x, kernels, dOutput.(shapes.Tensor), meta.colBuffer, dBias, meta.withBias, meta.stride)
}
