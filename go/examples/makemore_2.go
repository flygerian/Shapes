package examples

/*
#cgo CFLAGS: -I../../base
#include "memory.h"
*/
import "C"

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"slices"
	"strings"
	"unicode"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/activation"
	"github.com/flygerian/shapes/layer"
	"github.com/flygerian/shapes/loss"
	"github.com/flygerian/shapes/optimizer"
	"github.com/flygerian/shapes/visual"
)

func memoryStats(ctx shapes.Context) (allocated uint64, capacity uint64, blocks int, freeBlocks int, usedBlocks int) {
	mem := (*C.Memory)(ctx.UnsafeMemory())
	if mem == nil {
		return 0, 0, 0, 0, 0
	}

	blocks = int(mem.numBlocks)
	freeBlocks = int(mem.numFreeBlocks)
	usedBlocks = blocks - freeBlocks
	return uint64(mem.allocated), uint64(mem.capacity), blocks, freeBlocks, usedBlocks
}

func makeSet(input string) []rune {
	lookup := make(map[rune]bool)
	var result []rune

	for _, s := range input {

		s = unicode.ToLower(s)
		if lookup[s] {
			continue
		}

		lookup[s] = true
		result = append(result, s)
	}

	return result
}

func toAlphaPos(ch rune) int8 {
	return int8(ch - 'a' + 1)
}

func printFromAphabetPos(positions []int8, itos map[int8]rune) string {
	chars := make([]rune, len(positions))

	for _, pos := range positions {
		chars = append(chars, itos[pos])
	}

	return string(chars)
}

func newSection() {
	fmt.Printf("\n\n................................................................................\n\n")
}

func MakeMore_2(shapesCtx shapes.Context) {
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

	const blockSize = 3

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

			// fmt.Printf("%s ---> %s\n", printFromAphabetPos(context, itos), printFromAphabetPos([]int8{ix}, itos))

			context = append(append([]int8{}, context[1:]...), ix)
		}
	}

	X := shapes.FromInt8(shapesCtx, x)
	Y := shapes.FromInt8(shapesCtx, y)

	// newSection()

	// fmt.Printf("%v, %v, %v, %v\n", X.Shape(), X.Dtype(), Y.Shape(), Y.Dtype())

	// newSection()

	newSection()

	embLayer := layer.Embedding(shapesCtx, 27, 30)
	l1 := layer.Dense(shapesCtx, 90, 100)
	l2 := layer.Dense(shapesCtx, 100, 200)
	l3 := layer.Dense(shapesCtx, 200, 27)

	sgd := optimizer.SGD(shapesCtx, 0.00001)

	crossEnthropy := loss.CrossEntropy()

	forward := func(ctx shapes.Context, xBatch shapes.Tensor) shapes.Tensor {

		h := l1.Forward(ctx, xBatch.Reshape(ctx, -1, 90))
		h = activation.Tanh(ctx, h)
		h = l2.Forward(ctx, h)
		h = activation.Tanh(ctx, h)
		logits := l3.Forward(ctx, h)

		// fmt.Printf("yoneHot.shape: %v\n", yOneHot.Shape())
		// fmt.Printf("logits.shape: %v\n", logits.Shape())

		return logits
	}

	numEpochs := 10000
	for i := range numEpochs {
		epochCtx := shapesCtx.Epoch()
		beforeAllocated, capacity, beforeBlocks, beforeFree, beforeUsed := memoryStats(shapesCtx)
		ix := shapes.FloatRandom(epochCtx, shapes.Shape{32}, 0, float32(X.Shape()[0])).I64(epochCtx)

		// Forward pass
		emb := embLayer.Forward(epochCtx, X.Get(epochCtx, ix))
		yBatch := Y.Get(epochCtx, ix)

		// fmt.Printf("Emb shape: %v\n", emb.Shape())

		yOneHot := shapes.OneHot(epochCtx, yBatch, 27)
		logits := forward(epochCtx, emb)

		lossValue := crossEnthropy(epochCtx, yOneHot, logits)

		if i%100 == 0 {

			visual.ClearScreen(os.Stdout)
			fmt.Printf("Epoch %d, Loss: %f, \n", i, lossValue.Get(epochCtx, 0).Item())

			afterAllocated, _, afterBlocks, afterFree, afterUsed := memoryStats(shapesCtx)
			fmt.Printf(
				"[epoch %04d] used=%d->%d (%+d) blocks=%d->%d free=%d->%d allocated=%d->%d (%+d) capacity=%d\n",
				i,
				beforeUsed,
				afterUsed,
				afterUsed-beforeUsed,
				beforeBlocks,
				afterBlocks,
				beforeFree,
				afterFree,
				beforeAllocated,
				afterAllocated,
				int64(afterAllocated)-int64(beforeAllocated),
				capacity,
			)
		}

		// Backward pass
		graph := lossValue.Backward(epochCtx)

		// Update parameters
		sgd(graph)
		optimizer.ZeroGrad(shapesCtx, graph)
		epochCtx.Finish()

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
			logits := forward(testCtx, emb)
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
