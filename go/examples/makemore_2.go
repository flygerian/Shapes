package examples

import (
	"bufio"
	"fmt"
	"os"
	"slices"
	"strings"
	"unicode"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/layer"
	"github.com/flygerian/shapes/loss"
	"github.com/flygerian/shapes/optimizer"
)

func set(input string) []rune {
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

func fromAlphaPos(chIdx uint8) rune {
	return rune(chIdx) + 'a' - 1
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

func MakeMore_2(shapesCtx *shapes.Context) {
	file, err := os.Open("names.txt")
	if err != nil {
		panic("Could not open file")
	}

	scanner := bufio.NewScanner(file)

	var words []string
	for scanner.Scan() {
		words = append(words, scanner.Text())
	}

	newSection()

	fmt.Printf("Words: %v\n", words[:8])
	fmt.Printf("Number of words: %d\n\n", len(words))

	chars := set(strings.Join(words, ""))
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

	X := shapesCtx.FromInt8(x)
	Y := shapesCtx.FromInt8(y)

	// newSection()

	// fmt.Printf("%v, %v, %v, %v\n", X.Shape(), X.Dtype(), Y.Shape(), Y.Dtype())

	// newSection()

	C := shapesCtx.FloatRandom(shapes.Shape{27, 2})

	newSection()

	l1 := layer.Dense(6, 100)
	l2 := layer.Dense(100, 27)

	sgd := optimizer.SGD(shapesCtx, 0.1)

	shapesCtx.PrintMemoryFragmentationChart()

	for i := range 100 {
		// Forward pass
		emb := C.Get(X)
		emb.Tensor().Label = "emb"

		h := l1(emb.Reshape(-1, 6))
		h.Tensor().Label = "h"
		logits := l2(h)
		logits.Tensor().Label = "logits"

		yOneHot := shapesCtx.OneHot(Y, 27)
		yOneHot.Tensor().Label = "one_hot"

		lossValue := loss.CrossEntropy(yOneHot, logits)

		fmt.Printf("Epoch %d, Loss: %f\n", i, lossValue.Get(0).Item())

		// Backward pass
		graph := lossValue.Backward()

		// Update parameters
		sgd(graph)
		optimizer.ZeroGrad(shapesCtx, graph)
	}

}
