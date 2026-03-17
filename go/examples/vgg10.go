package examples

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strings"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/activation"
	"github.com/flygerian/shapes/layer"
	"github.com/flygerian/shapes/loss_fns"
	"github.com/flygerian/shapes/optimizer"
	"github.com/flygerian/shapes/visual"
)

type imageLabelPair struct {
	label     uint8
	imageData shapes.Tensor
}

type trainingParams struct {
	inputShape   shapes.Shape
	batchSize    int
	learningRate float32
	epochs       int
}

func getTrainingParams() trainingParams {
	return trainingParams{
		batchSize:    8,
		learningRate: 1e-5,
		inputShape:   shapes.Shape{3, 32, 32},
		epochs:       10,
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func getTensors(ctx shapes.Context, reader io.Reader) []imageLabelPair {
	imageTensors := make([]imageLabelPair, 0)

	idx := 0
	for {

		label := make([]byte, 1)
		image := make([]byte, 3072)

		_, err := io.ReadFull(reader, label)
		if err != nil {
			break
		}

		_, err = io.ReadFull(reader, image)
		if err != nil {
			break
		}

		imageTensors = append(imageTensors, imageLabelPair{label: label[0], imageData: shapes.TensorFromImagebyes(ctx, shapes.Shape{3, 32, 32}, image)})
		idx++
	}

	return imageTensors
}

func getDataSet(ctx shapes.Context) ([]imageLabelPair, []imageLabelPair, []imageLabelPair, []string) {
	batchFiles, err := filepath.Glob("examples/datasets/cifar-10-binary/cifar-10-batches-bin/data_batch_*.bin")
	if err != nil {
		err := fmt.Errorf("could not list cifar_10 batch files: %w", err)
		panic(err)
	}
	if len(batchFiles) == 0 {
		panic("could not find cifar_10 batch files")
	}

	labelsFile, err := os.Open("examples/datasets/cifar-10-binary/cifar-10-batches-bin/batches.meta.txt")
	if err != nil {
		err := fmt.Errorf("Could not open laberls files")
		panic(err)
	}
	defer labelsFile.Close()

	testFile, err := os.Open("examples/datasets/cifar-10-binary/cifar-10-batches-bin/test_batch.bin")
	if err != nil {
		err := fmt.Errorf("Could not open laberls files")
		panic(err)
	}
	defer testFile.Close()

	allTrain := make([]imageLabelPair, 0)
	for _, batchPath := range batchFiles[:2] {
		batchFile, err := os.Open(batchPath)
		if err != nil {
			err := fmt.Errorf("could not open cifar_10 file %q: %w", batchPath, err)
			panic(err)
		}

		allTrain = append(allTrain, getTensors(ctx, batchFile)...)

		if err := batchFile.Close(); err != nil {
			err := fmt.Errorf("could not close cifar_10 file %q: %w", batchPath, err)
			panic(err)
		}
	}
	test := getTensors(ctx, testFile)

	// 80/20 train/validation split
	trainSize := int(float64(len(allTrain)) * 0.8)
	train := allTrain[:trainSize]
	validation := allTrain[trainSize:]

	scanner := bufio.NewScanner(labelsFile)

	labels := make([]string, 0)
	for scanner.Scan() {
		text := scanner.Text()

		if strings.TrimSpace(text) == "" {
			continue
		}
		labels = append(labels, text)
	}

	return train, validation, test, labels
}

// Model

func convBlock(ctx shapes.Context) *layer.Sequential {
	convBlock := layer.Sequential{
		Layers: []layer.HasForward{
			layer.Conv2d(ctx, 3, 64, shapes.Kernel{2, 2}, 1), // (B, 64, 31, 31)
			activation.Relu(),
			layer.Conv2d(ctx, 64, 64, shapes.Kernel{2, 2}, 1), // (B, 64, 30, 30)
			activation.Relu(),

			layer.MaxPool2d(shapes.Kernel{2, 2}, 2), // (B, 64, 15, 15)

			layer.Conv2d(ctx, 64, 128, shapes.Kernel{2, 2}, 1), // (B, 128, 14, 14)
			activation.Relu(),
			layer.Conv2d(ctx, 128, 128, shapes.Kernel{2, 2}, 1), // (B, 128, 13, 13)
			activation.Relu(),

			layer.MaxPool2d(shapes.Kernel{2, 2}, 2), // (B, 128, 6, 6)

			layer.Conv2d(ctx, 128, 256, shapes.Kernel{2, 2}, 1), // (B, 256, 5, 5)
			activation.Relu(),
			layer.Conv2d(ctx, 256, 256, shapes.Kernel{2, 2}, 1), // (B, 256, 4, 4)
			activation.Relu(),
			layer.MaxPool2d(shapes.Kernel{2, 2}, 2), // (B, 256, 2, 2)

			layer.Conv2d(ctx, 256, 512, shapes.Kernel{2, 2}, 1), // (B, 512, 1, 1)
			activation.Relu(),
			layer.AdaptiveAvgPool2d(shapes.Shape{1, 1}),
		},
	}

	return &convBlock
}

func linearBlock(ctx shapes.Context, numLabels int) *layer.Sequential {
	denseBlock := layer.Sequential{
		Layers: []layer.HasForward{
			layer.Flatten(),
			layer.Dense(ctx, 512, 256),
			activation.Relu(),
			layer.Dense(ctx, 256, numLabels),
		},
	}

	return &denseBlock
}

func makeBatchIndexTensor(ctx shapes.Context, indices []int) shapes.Tensor {
	batchIndices := make([]float32, len(indices))
	for i, idx := range indices {
		batchIndices[i] = float32(idx)
	}

	return shapes.FromFloat32(ctx, shapes.Shape{uint(len(indices))}, batchIndices).I64(ctx)
}

func modelForward(
	ctx shapes.Context,
	model layer.Sequential,
	x shapes.Tensor) shapes.Tensor {
	// x = blocks[0].Forward(ctx, x) // Forward convblock
	// x = blocks[1].Forward(ctx, x)

	return model.Forward(ctx, x)
}

func computeAndSetValidationMetrics(
	epochCtx shapes.EpochContext,
	hyperParams trainingParams,
	XVal shapes.Tensor,
	YVal shapes.Tensor,
	labels []string,
	model layer.Sequential,
	crossEnthropy func(shapes.Context, shapes.Tensor, shapes.Tensor) shapes.Tensor,
) {
	totalValidationLoss := 0.0
	totalCorrect := 0
	totalSeen := 0

	for batchStart := 0; batchStart < int(XVal.Shape()[0]); batchStart += hyperParams.batchSize {
		batchEnd := min(batchStart+hyperParams.batchSize, int(XVal.Shape()[0]))
		testCtx := epochCtx.Test()

		valIndices := make([]int, batchEnd-batchStart)
		for i := range valIndices {
			valIndices[i] = batchStart + i
		}

		valIx := makeBatchIndexTensor(testCtx, valIndices)
		valBatch := XVal.Get(testCtx, valIx)
		valLogits := modelForward(testCtx, model, valBatch.F32(testCtx))
		valYBatch := YVal.Get(testCtx, valIx)
		valYOneHot := shapes.OneHot(testCtx, valYBatch, uint(len(labels))).Squeeze(testCtx)
		valLoss := crossEnthropy(testCtx, valYOneHot, valLogits)
		valLossScalar := valLoss.Get(testCtx, 0).Item().(float32)
		totalValidationLoss += float64(valLossScalar) * float64(len(valIndices))

		predictions := valLogits.ArgMax(testCtx, 1)
		actualLabels := valYBatch.Squeeze(testCtx)

		predValues := predictions.Values().([]int64)
		actualValues := actualLabels.Values().([]uint8)

		for i := range predValues {
			if predValues[i] == int64(actualValues[i]) {
				totalCorrect++
			}
		}
		totalSeen += len(valIndices)

		testCtx.Finish()
	}

	fmt.Printf("Val loss: %v\n", totalValidationLoss)
	epochCtx.SetValidationLoss(totalValidationLoss / float64(totalSeen))
	fmt.Printf("Accuracy: %v\n", float64(totalCorrect)/float64(totalSeen))
	epochCtx.SetAccuracy(float64(totalCorrect) / float64(totalSeen))
}

func runTraining(
	trainingCtx shapes.MainContext,
	trainSize int,
	hyperParams trainingParams,
	crossEnthropy func(shapesCtx shapes.Context, yGround shapes.Tensor, logits shapes.Tensor) shapes.Tensor,
	X shapes.Tensor,
	Y shapes.Tensor,
	optimizerStep func(cg shapes.ComputationGraph),
	model layer.Sequential,
	XVal, YVal shapes.Tensor,
	labels []string,
) {

	defer trainingCtx.Finish()

	for epochNum := range hyperParams.epochs {
		perm := rand.Perm(trainSize)
		epochLoss := 0.0
		epochCtx := trainingCtx.Epoch(epochNum + 1)

		for batchStart := 0; batchStart < trainSize; batchStart += hyperParams.batchSize {
			batchEnd := min(batchStart+hyperParams.batchSize, trainSize)
			stepCtx := epochCtx.Step()

			ix := makeBatchIndexTensor(stepCtx, perm[batchStart:batchEnd])
			batch := X.Get(stepCtx, ix)
			logits := modelForward(stepCtx, model, batch.F32(stepCtx))

			yBatch := Y.Get(stepCtx, ix)
			yOneHot := shapes.OneHot(stepCtx, yBatch, uint(len(labels))).Squeeze(stepCtx)
			loss := crossEnthropy(stepCtx, yOneHot, logits)

			graph := loss.Backward(stepCtx)
			optimizerStep(graph)
			optimizer.ZeroGrad(stepCtx, graph)

			lossScalar := loss.Get(stepCtx, 0).Item().(float32)
			epochLoss += float64(lossScalar) * float64(batchEnd-batchStart)
			stepCtx.SetStepLoss(float64(lossScalar))

			// fmt.Printf("Loss: %v\n", lossScalar)

			stepCtx.Finish()
		}

		epochCtx.SetLoss(epochLoss / float64(trainSize))
		epochCtx.Finish()
		// computeAndSetValidationMetrics(epochCtx, hyperParams, XVal, YVal, labels, model, crossEnthropy)
	}
}

func Vgg_cifar10() {
	shapeCtx := shapes.New(context.Background(), shapes.WithGrad(true), shapes.WithArenaSize(10000*Mb))
	defer shapeCtx.Finish()

	train, _, test, labels := getDataSet(shapeCtx)
	hyperParams := getTrainingParams()

	fmt.Printf("There are %d training images\n", len(train))
	fmt.Printf("There are %d test images\n", len(test))

	imageArtefacts := make([]visual.Artefact, 0)
	rows := make([]visual.Artefact, 0)

	for idx, ilp := range train {
		img := visual.Flex(
			visual.FlexOptions{
				Children: []visual.Artefact{
					visual.Image(visual.ImageOptions{
						Tensor:   ilp.imageData,
						Context:  shapeCtx,
						MaxWidth: 32,
					}),
					visual.Text(labels[ilp.label]),
				},
			},
		)

		imageArtefacts = append(imageArtefacts, img)

		if idx%1 == 0 {
			rows = append(
				rows,
				visual.Flex(visual.FlexOptions{
					Direction: visual.DirectionRow,
					Children:  imageArtefacts,
				}),
			)
			imageArtefacts = make([]visual.Artefact, 0)
		}
	}

	flex := visual.Flex(visual.FlexOptions{
		Direction: visual.DirectionColumn,
		Children:  rows[5:10],
	})

	visual.RenderAt(os.Stdout, flex, visual.Point{}, visual.Area{Width: 40, Height: 80})

	trainImgs := make([]shapes.Tensor, 0)
	trainLabels := make([]shapes.Tensor, 0)
	for i, pair := range train {
		if pair.imageData == nil {
			err := fmt.Sprintf("Imagedata %d is nil", i)
			panic(err)
		}
		trainImgs = append(trainImgs, pair.imageData)
		trainLabels = append(trainLabels, shapes.UInt8(shapeCtx, shapes.Shape{1}, uint8(pair.label)))
	}

	valImgs := make([]shapes.Tensor, 0)
	valLabels := make([]shapes.Tensor, 0)
	for i, pair := range test {
		if pair.imageData == nil {
			err := fmt.Sprintf("Test imagedata %d is nil", i)
			panic(err)
		}
		valImgs = append(valImgs, pair.imageData)
		valLabels = append(valLabels, shapes.UInt8(shapeCtx, shapes.Shape{1}, uint8(pair.label)))
	}

	XVal := shapes.Stack(shapeCtx, 0, valImgs...)
	YVal := shapes.Stack(shapeCtx, 0, valLabels...)

	X := shapes.Stack(shapeCtx, 0, trainImgs...)
	Y := shapes.Stack(shapeCtx, 0, trainLabels...)

	optimerStep := optimizer.SGD(shapeCtx, hyperParams.learningRate)
	crossEnthropy := loss_fns.CrossEntropy()

	model := layer.Sequential{
		Layers: []layer.HasForward{
			convBlock(shapeCtx),
			linearBlock(shapeCtx, len(labels)),
		},
	}
	fmt.Printf("Training start...\n")
	stepsPerEpoch := (len(train) + hyperParams.batchSize - 1) / hyperParams.batchSize

	trainingCtx := shapeCtx.Training(
		hyperParams.epochs,
		shapes.WithNumSteps(stepsPerEpoch),
		shapes.WithTrainingStatsRenderer(&visual.TrainingStatsRenderer{}),
	)

	runTraining(
		trainingCtx,
		len(train),
		hyperParams,
		crossEnthropy,
		X, Y,
		optimerStep,
		model,
		XVal,
		YVal,
		labels,
	)

}
