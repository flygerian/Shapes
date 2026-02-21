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

	for _, word := range words[:5] {
		context := make([]int8, blockSize)

		runes := []rune(word + ".")
		for _, ch := range runes {
			x = append(x, context)
			ix := stoi[ch]
			y = append(y, ix)

			fmt.Printf("%s ---> %s\n", printFromAphabetPos(context, itos), printFromAphabetPos([]int8{ix}, itos))

			context = append(append([]int8{}, context[1:]...), ix)
		}
	}

	X := shapesCtx.FromInt8(x)
	Y := shapesCtx.FromInt8(y)

	newSection()

	fmt.Printf("%v, %v, %v, %v\n", X.Shape(), X.Dtype(), Y.Shape(), Y.Dtype())

	newSection()

	C := shapesCtx.FloatRandom(shapes.Shape{27, 2})

	idx := shapesCtx.FromInt8([]int8{5})

	visual.Print(idx)

	oneHot := shapesCtx.OneHot(idx, 27)

	newSection()

	fmt.Printf("One hot shapes: %v\n", oneHot.Squeeze().Shape())
	visual.Print(oneHot)

	p := oneHot.Mul(C).Squeeze()

	fmt.Printf("Mul\n")
	visual.Print(p)

	newSection()

	emb := C.Get(X)
	fmt.Printf("Embedding table shape %v\n", emb.Shape())

	l1 := layer.Dense(6, 100)
	l2 := layer.Dense(100, 27)

	h := l1(emb.Reshape(-1, 6))
	logits := l2(h)

	counts := logits.Exp()

	fmt.Printf("Counts shape: %v\n", counts.Shape())

	prob := counts.Divide(counts.Sum(1))

	fmt.Printf("Probs.shape: %v\n", prob.Shape())

	loss := prob.Get(shapesCtx.Arange(32).I32(), Y).Log().Mean().Negate()

	visual.Print(loss)

	newSection()

	fmt.Println("Large exp\n")

	logits = shapesCtx.FromFloat32(shapes.Shape{4}, []float32{-100, -3, 0, 100}).F64()
	counts = logits.Exp()
	prob = counts.Divide(counts.Sum(0))

	visual.Print(prob)
}
