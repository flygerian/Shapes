package examples

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
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
		batchSize:    32,
		learningRate: 1e-5,
		inputShape:   shapes.Shape{3, 32, 32},
		epochs:       20,
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

		_, err := reader.Read(label)
		if err != nil {
			break
		}

		_, err = reader.Read(image)
		if err != nil {
			break
		}

		imageTensors = append(imageTensors, imageLabelPair{label: label[0], imageData: shapes.TensorFromImagebyes(ctx, shapes.Shape{3, 32, 32}, image)})
		idx++
	}

	return imageTensors
}

func getDataSet(ctx shapes.Context) ([]imageLabelPair, []imageLabelPair, []imageLabelPair, []string) {

	batchFile, err := os.Open("examples/datasets/cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin")
	if err != nil {
		err := fmt.Errorf("could not open cifar_10 file: %e", err)
		panic(err)
	}
	defer batchFile.Close()

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

	allTrain := getTensors(ctx, batchFile)
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

func convBlock(ctx shapes.Context) layer.Sequential {
	convBlock := layer.Sequential{
		Layers: []layer.HasForward{
			layer.Conv2d(ctx, 3, 64, shapes.Kernel{2, 2}, 1),  // (B, 64, 31, 31)
			layer.Conv2d(ctx, 64, 64, shapes.Kernel{2, 2}, 1), // (B, 64, 30, 30)
			layer.MaxPool2d(shapes.Kernel{2, 2}, 1),           // (B, 64, 29, 29)
			layer.BatchNorm2d(ctx, 64),                        // (B, 64, 29, 29)

			layer.Conv2d(ctx, 64, 128, shapes.Kernel{2, 2}, 1), // (B, 128, 28, 28)
		},
	}

	return convBlock
}

func linearBlock(ctx shapes.Context, numLabels int) layer.Sequential {
	denseBlock := layer.Sequential{
		Layers: []layer.HasForward{
			layer.Dense(ctx, 128*28*28, 4096),
			layer.Dense(ctx, 4096, 4096),
			layer.Dense(ctx, 4096, numLabels),
		},
	}

	return denseBlock
}

func modelForward(
	ctx shapes.Context,
	params trainingParams,
	blocks []layer.Sequential,
	x shapes.Tensor) shapes.Tensor {
	x = blocks[0].Forward(ctx, x) // Forward convblock

	x = x.Reshape(ctx, params.batchSize, -1)
	x = activation.Tanh(ctx, x)

	x = blocks[1].Forward(ctx, x)

	return x
}

func computeAndSetValidationLoss(
	epochCtx shapes.EpochContext,
	shapeCtx shapes.Context,
	hyperParams trainingParams,
	validation []imageLabelPair,
	XVal shapes.Tensor,
	YVal shapes.Tensor,
	labels []string,
	blocks []layer.Sequential,
	crossEnthropy func(shapes.Context, shapes.Tensor, shapes.Tensor) shapes.Tensor,
) {
	testCtx := epochCtx.Test()
	defer testCtx.Finish()

	valBatchSize := min(hyperParams.batchSize, len(validation))
	valIx := shapes.FloatRandom(shapeCtx, shapes.Shape{uint(valBatchSize)}, 0, float32(XVal.Shape()[0])).I64(testCtx)
	valBatch := XVal.Get(testCtx, valIx)
	valLogits := modelForward(testCtx, hyperParams, blocks, valBatch.F32(testCtx))
	valYBatch := YVal.Get(testCtx, valIx)
	valYOneHot := shapes.OneHot(testCtx, valYBatch, uint(len(labels))).Squeeze(testCtx)
	valLoss := crossEnthropy(testCtx, valYOneHot, valLogits)
	valLossScalar := valLoss.Get(testCtx, 0).Item().(float32)

	epochCtx.SetValidationLoss(float64(valLossScalar))
}

func Vgg_cifar10() {
	shapeCtx := shapes.New(context.Background(), shapes.WithGrad(true), shapes.WithArenaSize(16000*Mb))
	defer shapeCtx.Finish()

	train, validation, test, labels := getDataSet(shapeCtx)
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
	for i, pair := range validation {
		if pair.imageData == nil {
			err := fmt.Sprintf("Validation imagedata %d is nil", i)
			panic(err)
		}
		valImgs = append(valImgs, pair.imageData)
		valLabels = append(valLabels, shapes.UInt8(shapeCtx, shapes.Shape{1}, uint8(pair.label)))
	}

	XVal := shapes.Stack(shapeCtx, 0, valImgs...)
	YVal := shapes.Stack(shapeCtx, 0, valLabels...)

	trainingCtx := shapeCtx.Training(
		hyperParams.epochs,
		shapes.WithTrainingStatsRenderer(&visual.TrainingStatsRenderer{}),
	)

	X := shapes.Stack(shapeCtx, 0, trainImgs...)
	Y := shapes.Stack(shapeCtx, 0, trainLabels...)

	optimerStep := optimizer.Adam(shapeCtx, optimizer.WithLearningRate(hyperParams.learningRate))
	crossEnthropy := loss_fns.CrossEntropy()

	blocks := []layer.Sequential{convBlock(shapeCtx), linearBlock(shapeCtx, len(labels))}

	for epoch := range hyperParams.epochs {
		epochCtx := trainingCtx.Epoch(epoch + 1)
		ix := shapes.FloatRandom(shapeCtx, shapes.Shape{uint(hyperParams.batchSize)}, 0, float32(X.Shape()[0])).I64(epochCtx)
		batch := X.Get(epochCtx, ix)

		logits := modelForward(epochCtx, hyperParams, blocks, batch.F32(epochCtx))

		yBatch := Y.Get(epochCtx, ix)
		yOneHot := shapes.OneHot(epochCtx, yBatch, uint(len(labels))).Squeeze(shapeCtx)
		loss := crossEnthropy(epochCtx, yOneHot, logits)

		graph := loss.Backward(epochCtx)

		optimerStep(graph)

		optimizer.ZeroGrad(shapeCtx, graph)

		// Compute validation loss
		computeAndSetValidationLoss(epochCtx, shapeCtx, hyperParams, validation, XVal, YVal, labels, blocks, crossEnthropy)

		lossScalar := loss.Get(epochCtx, 0).Item().(float32)
		epochCtx.Finish(shapes.WithLoss(lossScalar))
	}

	trainingCtx.Finish()
}
