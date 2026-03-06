package layer

import (
	"fmt"

	shapes "github.com/flygerian/shapes"
)

const defaultBatchNormEpsilon = 1e-5

type batchNorm struct {
	gamma, beta, o shapes.Tensor
	numFeatures    int
	epsilon        float32
}

func BatchNorm(outerCtx shapes.Context, numFeatures int) Layer {
	gamma := shapes.Float(outerCtx, shapes.Shape{uint(numFeatures)}, 1.0)
	beta := shapes.Float(outerCtx, shapes.Shape{uint(numFeatures)}, 0.0)

	return &batchNorm{
		gamma:       gamma,
		beta:        beta,
		numFeatures: numFeatures,
		epsilon:     defaultBatchNormEpsilon,
	}
}

func (bn *batchNorm) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	fusedCtx := ctx.Forward(
		shapes.WithInputs(x),
		shapes.WithHiddenState(bn.gamma, bn.beta),
		shapes.WithOpType(shapes.OpBatchNorm),
	)

	x2d, originalShape := reshapeToBatchFeature2D(fusedCtx, x, bn.numFeatures)

	mean := x2d.Mean(fusedCtx, 0)
	centered := x2d.Minus(fusedCtx, mean)

	variance := centered.Pow(fusedCtx, 2).Mean(fusedCtx, 0)
	eps := shapes.Float(fusedCtx, variance.Shape(), bn.epsilon)
	invStd := variance.Plus(fusedCtx, eps).Pow(fusedCtx, -0.5)
	xHat := centered.Times(fusedCtx, invStd)

	bnOut2d := xHat.Times(fusedCtx, bn.gamma).Plus(fusedCtx, bn.beta)

	if len(originalShape) == 1 {
		bn.o = bnOut2d.Squeeze(fusedCtx)
	} else {
		bn.o = bnOut2d.Reshape(fusedCtx, shapeToIntDims(originalShape)...)
	}

	fusedCtx.Finish(
		shapes.WithResult(bn.o),
		shapes.WithBackward(constructBatchNormBackwardPass),
		shapes.WithMetadata(bn.epsilon),
	)

	return bn.o
}

func constructBatchNormBackwardPass(ctx shapes.Context, node shapes.ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	epsilon := node.Metadata().(float32)

	hiddenState := node.HiddenState()
	gamma := hiddenState[0]
	beta := hiddenState[1]

	featureShape := gamma.Shape()
	numFeatures := int(featureShape[0])

	x2d, originalShape := reshapeToBatchFeature2D(noGraphCtx, x, numFeatures)
	grad2d, _ := reshapeToBatchFeature2D(noGraphCtx, node.Grad().(shapes.Tensor), numFeatures)

	mean := x2d.Mean(noGraphCtx, 0)
	centered := x2d.Minus(noGraphCtx, mean)
	variance := centered.Pow(noGraphCtx, 2).Mean(noGraphCtx, 0)
	eps := shapes.Float(noGraphCtx, variance.Shape(), epsilon)
	invStd := variance.Plus(noGraphCtx, eps).Pow(noGraphCtx, -0.5)
	xHat := centered.Times(noGraphCtx, invStd)

	// dBeta = sum(dY, dim=0)
	dBeta2d := grad2d.Sum(noGraphCtx, 0)
	dBeta := dBeta2d.Squeeze(noGraphCtx)
	beta.Grad().Accumulate(noGraphCtx, dBeta)

	// dGamma = sum(dY * xHat, dim=0)
	dGamma2d := grad2d.Times(noGraphCtx, xHat).Sum(noGraphCtx, 0)
	dGamma := dGamma2d.Squeeze(noGraphCtx)
	gamma.Grad().Accumulate(noGraphCtx, dGamma)

	// dX = (1/m) * invStd * (m*dY*gamma - sum(dY*gamma) - xHat*sum((dY*gamma)*xHat))
	dXHat := grad2d.Times(noGraphCtx, gamma)
	sumDXHat := dXHat.Sum(noGraphCtx, 0)
	sumDXHatXHat := dXHat.Times(noGraphCtx, xHat).Sum(noGraphCtx, 0)

	m := float32(getBatchSizeFor2D(x2d.Shape()))
	mTensor := shapes.Float(noGraphCtx, xHat.Shape(), m)

	numerator := dXHat.Times(noGraphCtx, mTensor).
		Minus(noGraphCtx, sumDXHat).
		Minus(noGraphCtx, xHat.Times(noGraphCtx, sumDXHatXHat))
	dX2d := invStd.Times(noGraphCtx, numerator).Divide(noGraphCtx, mTensor)

	var dX shapes.Tensor
	if len(originalShape) == 1 {
		dX = dX2d.Squeeze(noGraphCtx)
	} else {
		dX = dX2d.Reshape(noGraphCtx, shapeToIntDims(originalShape)...)
	}

	gradX := shapes.ReduceBroadcast(noGraphCtx, x, dX.(shapes.GradTensor))
	x.Grad().Accumulate(noGraphCtx, gradX)
}

func reshapeToBatchFeature2D(ctx shapes.Context, x shapes.Tensor, numFeatures int) (shapes.Tensor, shapes.Shape) {
	shape := x.Shape()
	if len(shape) == 0 {
		panic("shapes: BatchNorm expects tensor with at least 1 dimension")
	}

	lastDim := shape[len(shape)-1]
	if int(lastDim) != numFeatures {
		err := fmt.Errorf("shapes: BatchNorm expecting last dim size %d, got %d", numFeatures, lastDim)
		panic(err)
	}

	if len(shape) == 1 {
		return x.UnSqueeze(ctx, 0), shape
	}

	batch := getBatchSizeForShape(shape)
	return x.Reshape(ctx, batch, numFeatures), shape
}

func getBatchSizeForShape(shape shapes.Shape) int {
	batch := 1
	for i := 0; i < len(shape)-1; i++ {
		batch *= int(shape[i])
	}
	return batch
}

func getBatchSizeFor2D(shape shapes.Shape) int {
	return int(shape[0])
}

func shapeToIntDims(shape shapes.Shape) []int {
	dims := make([]int, len(shape))
	for i := range shape {
		dims[i] = int(shape[i])
	}
	return dims
}
