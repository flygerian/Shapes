package examples

import (
	"bufio"
	"context"
	"fmt"
	"math/rand"
	"os"
	"slices"
	"strings"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/activation"
	"github.com/flygerian/shapes/layer"
	"github.com/flygerian/shapes/loss_fns"
	"github.com/flygerian/shapes/optimizer"
	"github.com/flygerian/shapes/visual"
)

func MakeMore_5() {
	shapesCtx := shapes.New(context.Background(), shapes.WithGrad(true), shapes.WithArenaSize(4096*Mb))
	defer shapesCtx.Finish()

	file, err := os.Open("names.txt")
	if err != nil {
		panic("Could not open file")
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var words []string
	for scanner.Scan() {
		words = append(words, scanner.Text())
	}

	newSection()

	fmt.Printf("Words: %v\n", words[:8])
	fmt.Printf("Number of words: %d\n\n", len(words))

	chars := makeSet(strings.Join(words, ""))
	slices.Sort(chars)

	fmt.Printf("Num chars: %d\n", len(chars))
	fmt.Printf("%s\n", string(chars))

	_ = make(map[rune]rune)
	stoi := make(map[rune]int8)
	itos := make(map[int8]rune)

	for _, s := range chars {
		stoi[s] = toAlphaPos(s)
		itos[toAlphaPos(s)] = s
	}

	stoi['.'] = 0
	itos[0] = '.'

	newSection()

	fmt.Printf("stoi: %v", stoi)
	fmt.Printf("itos: %v", itos)

	const blockSize = 8

	var x [][]int8
	var y []int8

	newSection()

	for _, word := range words {
		context := make([]int8, blockSize)

		runes := []rune(word + ".")
		for _, ch := range runes {
			x = append(x, context)
			ix := stoi[ch]
			y = append(y, ix)

			context = append(append([]int8{}, context[1:]...), ix)
		}
	}

	X := shapes.FromInt8(shapesCtx, x)
	Y := shapes.FromInt8(shapesCtx, y)

	newSection()

	embLayer := layer.Embedding(shapesCtx, 27, 10)
	l1 := layer.Dense(shapesCtx, 80, 100, layer.WithBias(false))
	l2 := layer.Dense(shapesCtx, 100, 100, layer.WithBias(false))
	l3 := layer.Dense(shapesCtx, 100, 27)
	bn1 := layer.BatchNorm(shapesCtx, 100)
	bn2 := layer.BatchNorm(shapesCtx, 100)
	optimizerStep := optimizer.Adam(shapesCtx, optimizer.WithLearningRate(0.0001))

	crossEnthropy := loss_fns.CrossEntropy()

	forward := func(ctx shapes.EpochContext, xBatch shapes.Tensor) shapes.Tensor {
		h := l1.Forward(ctx, xBatch.Reshape(ctx, int(xBatch.Shape()[0]), -1))
		h = bn1.Forward(ctx, h)
		h = activation.Tanh(ctx, h)

		if ctx.CurrentEpochNum()%100 == 0 {
			ctx.SampleTensor("layer-1-preact", h.Abs(ctx).GreaterThan(ctx, 0.99))
		}

		h = l2.Forward(ctx, h)
		h = bn2.Forward(ctx, h)
		h = activation.Tanh(ctx, h)
		logits := l3.Forward(ctx, h)

		return logits
	}

	numEpochs := 5000

	trainingCtx := shapesCtx.Training(
		numEpochs,
		shapes.WithTrainingStatsRenderer(&visual.TrainingStatsRenderer{}),
	)

	for i := range numEpochs {
		epochCtx := trainingCtx.Epoch(i + 1)
		ix := shapes.FloatRandom(epochCtx, shapes.Shape{32}, 0, float32(X.Shape()[0])).I64(epochCtx)

		// Forward pass
		emb := embLayer.Forward(epochCtx, X.Get(epochCtx, ix))
		yBatch := Y.Get(epochCtx, ix)

		yOneHot := shapes.OneHot(epochCtx, yBatch, 27)
		logits := forward(epochCtx, emb)

		lossValue := crossEnthropy(epochCtx, yOneHot, logits)

		lossScalar := lossValue.Get(epochCtx, 0).Item().(float32)

		// Backward pass
		graph := lossValue.Backward(epochCtx)

		// Update parameters
		optimizerStep(graph)

		optimizer.ZeroGrad(shapesCtx, graph)

		if i%100 == 0 {
			epochCtx.Finish(shapes.WithLoss(lossScalar))
		} else {
			epochCtx.Finish()
		}
	}

	newSection()

	fmt.Printf("Generated names:\n")
	const numSamples = 10
	const maxNameLen = 20
	const vocabSize = 27

	for sample := range numSamples {
		testCtx := shapesCtx.Forward()

		context := make([]int8, blockSize)
		generated := make([]rune, 0, maxNameLen)

		for range maxNameLen {
			xInf := shapes.FromInt8(testCtx, [][]int8{context})
			emb := embLayer.Forward(testCtx, xInf)
			logits := forward(testCtx.(shapes.EpochContext), emb)
			probs := activation.Softmax(testCtx, logits)

			// Sample from the probability distribution instead of greedy argmax
			// to avoid collapsing to the same output every time.
			r := rand.Float32()
			cumulative := float32(0.0)
			nextIdx := int8(0)
			for classIdx := range vocabSize {
				p := probs.Get(testCtx, 0, classIdx).Item().(float32)
				cumulative += p
				if r <= cumulative {
					nextIdx = int8(classIdx)
					break
				}
			}

			if nextIdx == 0 {
				break
			}

			generated = append(generated, itos[nextIdx])
			context = append(context[1:], nextIdx)
		}

		fmt.Printf("%2d. %s\n", sample+1, string(generated))
		testCtx.Finish()
	}

}
