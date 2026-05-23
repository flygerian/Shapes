package layer

import (
	"fmt"

	shapes "github.com/flygerian/shapes"
)

const defaultBatchNormEpsilon = 1e-5
const defaultBatchNormMomentum = 0.1

type batchNorm struct {
	ctx                     shapes.Context
	gamma, beta, o          shapes.Tensor
	runningMean, runningVar []float32
	runningStatsInitialized bool
	numFeatures             int
	epsilon                 float32
	momentum                float32
	dims                    int8 // Decide if it's 1, 2,...d batch norm
}

type batchNormMetadata struct {
	epsilon float32
	dims    int8
}

func BatchNorm(outerCtx shapes.Context, numFeatures int) *batchNorm {
	gamma := shapes.Float(outerCtx, shapes.Shape{uint(numFeatures)}, 1.0)
	beta := shapes.Float(outerCtx, shapes.Shape{uint(numFeatures)}, 0.0)

	return &batchNorm{
		ctx:         outerCtx,
		gamma:       gamma,
		beta:        beta,
		numFeatures: numFeatures,
		epsilon:     defaultBatchNormEpsilon,
		momentum:    defaultBatchNormMomentum,
		dims:        1,
	}
}

func BatchNorm2d(outerCtx shapes.Context, numFeatures int) *batchNorm {
	gamma := shapes.Float(outerCtx, shapes.Shape{uint(numFeatures)}, 1.0)
	beta := shapes.Float(outerCtx, shapes.Shape{uint(numFeatures)}, 0.0)

	return &batchNorm{
		ctx:         outerCtx,
		gamma:       gamma,
		beta:        beta,
		numFeatures: numFeatures,
		epsilon:     defaultBatchNormEpsilon,
		momentum:    defaultBatchNormMomentum,
		dims:        2,
	}
}

func (bn *batchNorm) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	fusedCtx := ctx.Forward(
		shapes.WithInputs(x),
		shapes.WithHiddenState(bn.gamma, bn.beta),
		shapes.WithOpType(shapes.OpBatchNorm),
	)

	var x2d shapes.Tensor
	var originalShape shapes.Shape

	switch bn.dims {
	case 1:
		x2d, originalShape = reshapeToBatchFeature2D(fusedCtx, x, bn.numFeatures)
	case 2:
		x2d, originalShape = reshapeNHWCToBatchFeature2D(fusedCtx, x, bn.numFeatures)
	default:
		panic(fmt.Sprintf("shapes: unsupported BatchNorm dims %d", bn.dims))
	}

	var bnOut2d shapes.Tensor
	if ctx.IsTraining() || !bn.runningStatsInitialized {
		// Fused training path computes output + batch stats in one C call.
		var mean, variance shapes.Tensor
		bnOut2d, mean, variance = shapes.BatchNormForwardTraining(fusedCtx, x2d, bn.gamma, bn.beta, bn.epsilon)
		if ctx.IsTraining() {
			bn.updateRunningStats(mean, variance)
		}
	} else {
		statsShape := shapes.Shape{uint(bn.numFeatures)}
		mean := shapes.FromFloat32(fusedCtx, statsShape, bn.runningMean)
		variance := shapes.FromFloat32(fusedCtx, statsShape, bn.runningVar)
		centered := x2d.Minus(fusedCtx, mean)
		eps := shapes.Float(fusedCtx, variance.Shape(), bn.epsilon)
		invStd := variance.Plus(fusedCtx, eps).Pow(fusedCtx, -0.5)
		xHat := centered.Times(fusedCtx, invStd)
		bnOut2d = xHat.Times(fusedCtx, bn.gamma).Plus(fusedCtx, bn.beta)
	}

	if len(originalShape) == 1 {
		bn.o = bnOut2d.Squeeze(fusedCtx)
	} else if bn.dims == 2 {
		bn.o = restoreBatchNorm2DOutput(fusedCtx, bnOut2d, originalShape)
	} else {
		bn.o = bnOut2d.Reshape(fusedCtx, shapeToIntDims(originalShape)...)
	}

	fusedCtx.Finish(
		shapes.WithResult(bn.o),
		shapes.WithBackward(constructBatchNormBackwardPass),
		shapes.WithMetadata(batchNormMetadata{
			epsilon: bn.epsilon,
			dims:    bn.dims,
		}),
	)

	return bn.o
}

func (bn *batchNorm) updateRunningStats(mean, variance shapes.Tensor) {
	meanVals, ok := mean.Values().([]float32)
	if !ok {
		panic("shapes: BatchNorm running stats require float32 mean")
	}
	varVals, ok := variance.Values().([]float32)
	if !ok {
		panic("shapes: BatchNorm running stats require float32 variance")
	}

	if !bn.runningStatsInitialized {
		bn.runningMean = append([]float32(nil), meanVals...)
		bn.runningVar = append([]float32(nil), varVals...)
		bn.runningStatsInitialized = true
		return
	}

	if len(meanVals) != len(bn.runningMean) || len(varVals) != len(bn.runningVar) {
		panic("shapes: BatchNorm running stats size mismatch")
	}

	keep := float32(1.0) - bn.momentum
	for i := range meanVals {
		bn.runningMean[i] = bn.runningMean[i]*keep + meanVals[i]*bn.momentum
		bn.runningVar[i] = bn.runningVar[i]*keep + varVals[i]*bn.momentum
	}
}

func constructBatchNormBackwardPass(ctx shapes.Context, node shapes.ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	meta := node.Metadata().(batchNormMetadata)
	epsilon := meta.epsilon

	hiddenState := node.HiddenState()
	gamma := hiddenState[0]
	beta := hiddenState[1]

	featureShape := gamma.Shape()
	numFeatures := int(featureShape[0])

	var x2d shapes.Tensor
	var grad2d shapes.Tensor
	var originalShape shapes.Shape
	switch meta.dims {
	case 1:
		x2d, originalShape = reshapeToBatchFeature2D(noGraphCtx, x, numFeatures)
		grad2d, _ = reshapeToBatchFeature2D(noGraphCtx, node.Grad().(shapes.Tensor), numFeatures)
	case 2:
		x2d, originalShape = reshapeNHWCToBatchFeature2D(noGraphCtx, x, numFeatures)
		grad2d, _ = reshapeNHWCToBatchFeature2D(noGraphCtx, node.Grad().(shapes.Tensor), numFeatures)
	default:
		panic(fmt.Sprintf("shapes: unsupported BatchNorm dims %d", meta.dims))
	}

	dX2d, dGamma, dBeta := shapes.BatchNormBackward(noGraphCtx, x2d, grad2d, gamma, epsilon)
	beta.Grad().Accumulate(noGraphCtx, dBeta)
	gamma.Grad().Accumulate(noGraphCtx, dGamma)

	var dX shapes.Tensor
	if meta.dims == 2 {
		dX = restoreBatchNorm2DOutput(noGraphCtx, dX2d, originalShape)
	} else if len(originalShape) == 1 {
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

	return x.Reshape(ctx, -1, numFeatures), shape
}

func reshapeNHWCToBatchFeature2D(ctx shapes.Context, x shapes.Tensor, numFeatures int) (shapes.Tensor, shapes.Shape) {
	shape := x.Shape()
	if len(shape) != 4 {
		err := fmt.Errorf("shapes: BatchNorm2d expects NHWC tensor with 4 dims, got %d dims", len(shape))
		panic(err)
	}

	channelDim := shape[3]
	if int(channelDim) != numFeatures {
		err := fmt.Errorf("shapes: BatchNorm2d expects channel dim size %d, got %d", numFeatures, channelDim)
		panic(err)
	}

	x2d, _ := reshapeToBatchFeature2D(ctx, x, numFeatures)
	return x2d, shape
}

func restoreBatchNorm2DOutput(ctx shapes.Context, x2d shapes.Tensor, originalShape shapes.Shape) shapes.Tensor {
	if len(originalShape) != 4 {
		panic("shapes: BatchNorm2d restore expects 4D original shape")
	}

	n := int(originalShape[0])
	h := int(originalShape[1])
	w := int(originalShape[2])
	c := int(originalShape[3])

	return x2d.Reshape(ctx, n, h, w, c)
}

func shapeToIntDims(shape shapes.Shape) []int {
	dims := make([]int, len(shape))
	for i := range shape {
		dims[i] = int(shape[i])
	}
	return dims
}
