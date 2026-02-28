package examples

import (
	"bufio"
	"fmt"
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

	X := shapes.FromInt8(shapesCtx, x)
	Y := shapes.FromInt8(shapesCtx, y)

	// newSection()

	// fmt.Printf("%v, %v, %v, %v\n", X.Shape(), X.Dtype(), Y.Shape(), Y.Dtype())

	// newSection()

	newSection()

	embLayer := layer.Embedding(shapesCtx, 27, 2)
	l1 := layer.Dense(shapesCtx, 6, 100)
	l2 := layer.Dense(shapesCtx, 100, 27)

	sgd := optimizer.SGD(shapesCtx, 0.01)

	crossEnthropy := loss.CrossEntropy(shapesCtx)

	forward := func(ctx shapes.Context, xBatch shapes.Tensor) shapes.Tensor {

		h := l1.Forward(ctx, xBatch.Reshape(ctx, -1, 6))
		logits := l2.Forward(ctx, h)

		// fmt.Printf("yoneHot.shape: %v\n", yOneHot.Shape())
		// fmt.Printf("logits.shape: %v\n", logits.Shape())

		return logits
	}

	for i := range 500 {
		epochCtx := shapesCtx.Epoch()
		ix := shapes.FloatRandom(epochCtx, shapes.Shape{32}, 0, float32(X.Shape()[0])).I64(epochCtx)

		// Forward pass
		emb := embLayer.Forward(epochCtx, X.Get(epochCtx, ix))
		yBatch := Y.Get(epochCtx, ix)

		// fmt.Printf("Emb shape: %v\n", emb.Shape())

		yOneHot := shapes.OneHot(epochCtx, yBatch, 27)
		logits := forward(epochCtx, emb)

		lossValue := crossEnthropy(yOneHot, logits)
		fmt.Printf("Epoch %d, Loss: %f, \n", i, lossValue.Get(epochCtx, 0).Item())

		// Backward pass
		graph := lossValue.Backward(epochCtx)

		// Update parameters
		sgd(graph)
		optimizer.ZeroGrad(shapesCtx, graph)
		epochCtx.Finish()
	}

	newSection()

	testCtx := shapesCtx.Forward()
	x_inf := shapes.Float(testCtx, shapes.Shape{1, 3}, 0)

	emb := embLayer.Forward(testCtx, x_inf.I64(shapesCtx))
	logits := forward(testCtx, emb)

	probs := activation.Softmax(testCtx, logits)

	fmt.Printf("Probs: ")
	visual.Print(testCtx, probs)
	predicted := probs.Squeeze(testCtx).Max(testCtx)

	fmt.Printf("Predicted: ")
	visual.Print(testCtx, predicted)

}
