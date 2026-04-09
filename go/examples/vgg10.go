package examples

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
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

type dataset struct {
	Xtrain []shapes.Tensor
	Ytrain []shapes.Tensor

	Xtest []shapes.Tensor
	Ytest []shapes.Tensor
}

type datasetRaw struct {
	Xtrain [][]float32
	Ytrain []int8

	Xtest [][]float32
	Ytest []int8
}

type trainingParams struct {
	inputShape   shapes.Shape
	batchSize    int
	learningRate float32
	epochs       int
}

func getTrainingParams() trainingParams {
	return trainingParams{
		batchSize:    128,
		learningRate: 1e-2,
		inputShape:   shapes.Shape{32, 32, 3},
		epochs:       50,
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func shouldLogTrainStep() bool {
	value := os.Getenv("SHAPES_LOG_TRAIN_STEP")
	return value != "" && value != "0"
}

func getImagesAndlabels(ctx shapes.Context, reader io.Reader) ([][]float32, []int8) {
	var images [][]float32
	var labels []int8

	idx := 0
	for {

		label := make([]byte, 1)
		imageData := make([]byte, 3072)

		_, err := io.ReadFull(reader, label)
		if err != nil {
			break
		}

		_, err = io.ReadFull(reader, imageData)
		if err != nil {
			break
		}

		imageNHWC := make([]float32, 32*32*3)
		channelPlane := 32 * 32
		for y := range 32 {
			for x := range 32 {
				pixelIdx := y*32 + x
				base := pixelIdx * 3
				imageNHWC[base+0] = float32(imageData[pixelIdx]) / 255.0
				imageNHWC[base+1] = float32(imageData[channelPlane+pixelIdx]) / 255.0
				imageNHWC[base+2] = float32(imageData[2*channelPlane+pixelIdx]) / 255.0
			}
		}

		images = append(images, imageNHWC)
		labels = append(labels, int8(label[0]))

		idx++
	}

	return images, labels
}

func getDataSet(ctx shapes.Context) (datasetRaw, []string) {
	batchFiles, err := filepath.Glob("examples/datasets/cifar-10-binary/cifar-10-batches-bin/data_batch_*.bin")
	if err != nil {
		err := fmt.Errorf("could not list cifar_10 batch files: %w", err)
		panic(err)
	}
	if len(batchFiles) == 0 {
		panic("could not find cifar_10 batch files")
	}

	labelsDataFile, err := os.Open("examples/datasets/cifar-10-binary/cifar-10-batches-bin/batches.meta.txt")
	if err != nil {
		err := fmt.Errorf("Could not open laberls files")
		panic(err)
	}
	defer labelsDataFile.Close()

	testBatchFile, err := os.Open("examples/datasets/cifar-10-binary/cifar-10-batches-bin/test_batch.bin")
	if err != nil {
		err := fmt.Errorf("Could not open laberls files")
		panic(err)
	}
	defer testBatchFile.Close()

	Xs := make([][]float32, 0)
	Ys := make([]int8, 0)

	for _, batchPath := range batchFiles {
		batchFile, err := os.Open(batchPath)
		if err != nil {
			err := fmt.Errorf("could not open cifar_10 file %q: %w", batchPath, err)
			panic(err)
		}

		imagesinBatchFile, labelsInBatchFile := getImagesAndlabels(ctx, batchFile)
		Xs = append(Xs, imagesinBatchFile...)
		Ys = append(Ys, labelsInBatchFile...)

		if err := batchFile.Close(); err != nil {
			err := fmt.Errorf("could not close cifar_10 file %q: %w", batchPath, err)
			panic(err)
		}
	}

	testImages, testLabels := getImagesAndlabels(ctx, testBatchFile)

	scanner := bufio.NewScanner(labelsDataFile)
	labelData := make([]string, 0)
	for scanner.Scan() {
		text := scanner.Text()

		if strings.TrimSpace(text) == "" {
			continue
		}

		labelData = append(labelData, text)
	}

	ds := datasetRaw{Xtrain: Xs, Ytrain: Ys, Xtest: testImages, Ytest: testLabels}
	return ds, labelData
}

func shuffleAndBatchDataSet(ctx shapes.Context, dsRaw datasetRaw, params trainingParams) dataset {
	// Shuffle train and test
	fmt.Printf("Shuffling dataset...\n")
	shuffledDs := dsRaw

	ctx = ctx.BackwardDisabled()

	rand.Shuffle(len(shuffledDs.Xtrain), func(i, j int) {
		shuffledDs.Xtrain[i], shuffledDs.Xtrain[j] = shuffledDs.Xtrain[i], shuffledDs.Xtrain[j]
		shuffledDs.Ytrain[i], shuffledDs.Ytrain[j] = shuffledDs.Ytrain[i], shuffledDs.Ytrain[j]
	})

	fmt.Printf("Chunking dataset...\n")
	XtrainChunked, YtrainChunked := slices.Chunk(shuffledDs.Xtrain, params.batchSize), slices.Chunk(shuffledDs.Ytrain, params.batchSize)
	_, _ = slices.Chunk(shuffledDs.Xtest, params.batchSize), slices.Chunk(shuffledDs.Ytest, params.batchSize)

	Xtrain := make([]shapes.Tensor, 0)
	Ytrain := make([]shapes.Tensor, 0)

	Xtest := make([]shapes.Tensor, 0)
	Ytest := make([]shapes.Tensor, 0)

	fmt.Printf("Building batches... Xtrain\n")
	for chunk := range XtrainChunked {
		Xtrain = append(Xtrain, shapes.FromFloat32(ctx, shapes.Shape{uint(len(chunk)), 32, 32, 3}, slices.Concat(chunk...)))
	}

	fmt.Printf("Building batches... Ytrain\n")
	for chunk := range YtrainChunked {
		Ytrain = append(Ytrain, shapes.FromInt8(ctx, chunk))
	}

	// fmt.Printf("Building batches... Xtest\n")
	// for chunk := range XtestChunked {
	// 	Xtest = append(Xtest, shapes.Stack(ctx, 0, chunk...))
	// }
	//
	// fmt.Printf("Building batches... Ytest\n")
	// for chunk := range YtestChunked {
	// 	Ytest = append(Ytest, shapes.Stack(ctx, 0, chunk...))
	// }

	fmt.Printf("Train batches: %d, Test batches %d\n", len(Xtrain), len(Xtest))

	return dataset{Xtrain: Xtrain, Ytrain: Ytrain, Xtest: Xtest, Ytest: Ytest}
}

// Model

func convBlock(ctx shapes.Context) *layer.Sequential {
	return layer.NewSequential(
		ctx,
		layer.Conv2d(ctx, 3, 64, shapes.Kernel{2, 2}, 1), // (B, 31, 31, 64)
		activation.Relu(),
		layer.Conv2d(ctx, 64, 64, shapes.Kernel{2, 2}, 1), // (B, 30, 30, 64)
		activation.Relu(),

		layer.MaxPool2d(ctx, shapes.Kernel{2, 2}, 2), // (B, 15, 15, 64)

		layer.Conv2d(ctx, 64, 128, shapes.Kernel{2, 2}, 1), // (B, 14, 14, 128)
		activation.Relu(),
		layer.Conv2d(ctx, 128, 128, shapes.Kernel{2, 2}, 1), // (B, 13, 13, 128)
		activation.Relu(),

		layer.MaxPool2d(ctx, shapes.Kernel{2, 2}, 2), // (B, 6, 6, 128)

		layer.Conv2d(ctx, 128, 256, shapes.Kernel{2, 2}, 1), // (B, 5, 5, 256)
		activation.Relu(),
		layer.Conv2d(ctx, 256, 256, shapes.Kernel{2, 2}, 1), // (B, 4, 4, 256)
		activation.Relu(),
		layer.MaxPool2d(ctx, shapes.Kernel{2, 2}, 2), // (B, 2, 2, 256)

		layer.Conv2d(ctx, 256, 512, shapes.Kernel{2, 2}, 1), // (B, 1, 1, 512)
		activation.Relu(),
		layer.AdaptiveAvgPool2d(ctx, shapes.Shape{1, 1}),
	)
}

func linearBlock(ctx shapes.Context, numLabels int) *layer.Sequential {
	return layer.NewSequential(
		ctx,
		layer.Flatten(ctx),
		layer.Dense(ctx, 512, 256),
		activation.Relu(),
		layer.Dense(ctx, 256, numLabels),
	)
}

func modelForward(
	ctx shapes.Context,
	model *layer.Sequential,
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
	model *layer.Sequential,
	crossEnthropy func(yGround shapes.Tensor, logits shapes.Tensor) shapes.Tensor,
) {
	totalValidationLoss := 0.0
	totalCorrect := 0
	totalSeen := 0
	valOrder := shapes.Arange(epochCtx, float32(XVal.Shape()[0])).I64(epochCtx)

	for batchStart := 0; batchStart < int(XVal.Shape()[0]); batchStart += hyperParams.batchSize {
		batchEnd := min(batchStart+hyperParams.batchSize, int(XVal.Shape()[0]))
		testCtx := epochCtx.Test()
		valIx := valOrder.Slice(testCtx, shapes.Range{uint(batchStart), uint(batchEnd)})
		valBatch := XVal.Get(testCtx, valIx)
		valLogits := modelForward(testCtx, model, valBatch.F32(testCtx))
		valYBatch := YVal.Get(testCtx, valIx)
		valYOneHot := shapes.OneHot(testCtx, valYBatch, uint(len(labels))).Squeeze(testCtx)
		valLoss := crossEnthropy(valYOneHot, valLogits)
		valLossScalar := valLoss.Get(testCtx, 0).Item().(float32)
		totalValidationLoss += float64(valLossScalar) * float64(batchEnd-batchStart)

		predictions := valLogits.ArgMax(testCtx, 1)
		actualLabels := valYBatch.Squeeze(testCtx)

		predValues := predictions.Values().([]int64)
		actualValues := actualLabels.Values().([]uint8)

		for i := range predValues {
			if predValues[i] == int64(actualValues[i]) {
				totalCorrect++
			}
		}
		totalSeen += batchEnd - batchStart

		testCtx.Finish()
	}

	fmt.Printf("Val loss: %v\n", totalValidationLoss)
	epochCtx.SetValidationLoss(totalValidationLoss / float64(totalSeen))
	fmt.Printf("Accuracy: %v\n", float64(totalCorrect)/float64(totalSeen))
	epochCtx.SetAccuracy(float64(totalCorrect) / float64(totalSeen))
}

func runTraining(
	trainingCtx shapes.SubContext,
	model *layer.Sequential,
	crossEnthropy func(ctx shapes.Context, yGround shapes.Tensor, logits shapes.Tensor) shapes.Tensor,
	optimizerStep func(ctx shapes.Context, cg shapes.ComputationGraph),
	hyperParams trainingParams,

	ds dataset,
	labels []string,
) {

	defer trainingCtx.Finish()

	for epochNum := range hyperParams.epochs {
		totalEpochLoss := 0.0
		epochCtx := trainingCtx.Epoch(epochNum + 1)
		// epochStart := time.Now()

		for batchNum := range len(ds.Xtrain) {
			stepCtx := epochCtx.(shapes.EpochContext).Step()

			if stepCtx.CurrentStep() > 10 {
			}

			Xbatch := ds.Xtrain[batchNum]

			logits := modelForward(stepCtx, model, Xbatch)

			yBatch := ds.Ytrain[batchNum]
			yOneHot := shapes.OneHot(stepCtx, yBatch, uint(len(labels))).Squeeze(stepCtx)
			loss := crossEnthropy(stepCtx, yOneHot, logits)

			graph := loss.Backward(stepCtx)
			optimizerStep(stepCtx, graph)
			optimizer.ZeroGrad(stepCtx, graph)

			if stepCtx.CurrentStep()%300 == 0 {
				lossScalar := loss.Get(stepCtx, 0).Item().(float32)
				totalEpochLoss += float64(lossScalar)
				// epochDuration := time.Since(epochStart)
				// avgStepDuration := epochDuration.Milliseconds() / int64(stepCtx.CurrentStep())

				// header := fmt.Sprintf("Avg Step Duration: %d ms", avgStepDuration)
				// fmt.Printf("Loss: %v, Step: %d | %s \n", lossScalar, stepCtx.CurrentStep(), header)
				stepCtx.SetStepLoss(float64(lossScalar))
			}

			stepCtx.Finish()
		}

		epochCtx.SetLoss(totalEpochLoss / (float64(hyperParams.epochs) / 100))
		epochCtx.Finish()
		// computeAndSetValidationMetrics(epochCtx, hyperParams, XVal, YVal, labels, model, crossEnthropy)
	}
}

func Vgg_cifar10() {
	cpuCtx := shapes.New(context.Background(), shapes.WithGrad(true), shapes.WithArenaSize(10000*Mb))
	gpuCtx := shapes.New(context.Background(), shapes.WithGrad(true), shapes.WithCuda())
	// defer cpuCtx.Finish()

	datasetRaw, labelData := getDataSet(cpuCtx)
	hyperParams := getTrainingParams()
	ds := shuffleAndBatchDataSet(gpuCtx, datasetRaw, hyperParams)

	// imageArtefacts := make([]visual.Artefact, 0)
	// rows := make([]visual.Artefact, 0)

	// for idx, ilp := range dataset.Xtrain {
	// 	img := visual.Flex(
	// 		visual.FlexOptions{
	// 			Children: []visual.Artefact{
	// 				visual.Image(visual.ImageOptions{
	// 					Tensor:   ilp.imageData,
	// 					Context:  cpuCtx,
	// 					MaxWidth: 32,
	// 				}),
	// 				visual.Text(labels[ilp.label]),
	// 			},
	// 		},
	// 	)
	//
	// 	imageArtefacts = append(imageArtefacts, img)
	//
	// 	if idx%1 == 0 {
	// 		rows = append(
	// 			rows,
	// 			visual.Flex(visual.FlexOptions{
	// 				Direction: visual.DirectionRow,
	// 				Children:  imageArtefacts,
	// 			}),
	// 		)
	// 		imageArtefacts = make([]visual.Artefact, 0)
	// 	}
	// }
	//
	// _ = visual.Flex(visual.FlexOptions{
	// 	Direction: visual.DirectionColumn,
	// 	Children:  rows[5:10],
	// })

	// visual.RenderAt(os.Stdout, flex, visual.Point{}, visual.Area{Width: 40, Height: 80})

	optimerStep := optimizer.SGD(hyperParams.learningRate)
	crossEnthropy := loss_fns.CrossEntropy()

	model := layer.NewSequential(
		cpuCtx,
		convBlock(gpuCtx),
		linearBlock(gpuCtx, len(labelData)),
	)

	fmt.Printf("\nTraining start...\n")

	trainingCtx := gpuCtx.Training(
		hyperParams.epochs,
		shapes.WithNumSteps(len(ds.Xtrain)),
		shapes.WithTrainingStatsRenderer(&visual.TrainingStatsRenderer{}),
	)

	runTraining(
		trainingCtx,
		model,
		crossEnthropy,
		optimerStep,
		hyperParams,

		ds,
		labelData,
	)

}
