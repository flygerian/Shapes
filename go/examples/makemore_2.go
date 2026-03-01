package examples

import (
	"bufio"
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/activation"
	"github.com/flygerian/shapes/layer"
	"github.com/flygerian/shapes/loss"
	"github.com/flygerian/shapes/optimizer"
	"github.com/flygerian/shapes/visual"
)

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

	sgd := optimizer.SGD(shapesCtx, 0.0001)

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

	numEpochs := 50000
	type uiState struct {
		epoch                int
		numEpochs            int
		loss                 float32
		memorySampleHistoryX []int
		usedBlocksHistory    []int
		lossHistoryX         []int
		lossHistory          []int
		version              int
	}
	var (
		stateMu sync.RWMutex
		state   = uiState{
			numEpochs:            numEpochs,
			memorySampleHistoryX: make([]int, 0, numEpochs),
			usedBlocksHistory:    make([]int, 0, numEpochs),
			lossHistoryX:         make([]int, 0, numEpochs/100+1),
			lossHistory:          make([]int, 0, numEpochs/100+1),
		}
		done = make(chan struct{})
		wg   sync.WaitGroup
	)

	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(time.Second / 60)
		defer ticker.Stop()
		fmt.Fprint(os.Stdout, "\033[?25l\033[2J\033[H")
		defer fmt.Fprint(os.Stdout, "\033[?25h")

		lastVersion := -1

		render := func(force bool) {
			stateMu.RLock()
			epoch := state.epoch
			totalEpochs := state.numEpochs
			loss := state.loss
			memorySampleHistoryX := append([]int(nil), state.memorySampleHistoryX...)
			usedBlocksHistory := append([]int(nil), state.usedBlocksHistory...)
			lossHistoryX := append([]int(nil), state.lossHistoryX...)
			lossHistory := append([]int(nil), state.lossHistory...)
			version := state.version
			stateMu.RUnlock()

			if len(memorySampleHistoryX) == 0 || len(lossHistoryX) == 0 {
				return
			}
			if !force && version == lastVersion {
				return
			}
			lastVersion = version

			header := fmt.Sprintf("Epoch %d/%d | Loss %.6f", epoch, totalEpochs, loss)
			dashboard := visual.Flex(
				visual.DirectionColumn,
				visual.Box("Makemore Training", header),
				visual.Flex(
					visual.DirectionRow,
					visual.Flex(
						visual.DirectionColumn,
						visual.Text("memory"),
						visual.Chart(memorySampleHistoryX, usedBlocksHistory),
					),
					visual.Flex(
						visual.DirectionColumn,
						visual.Text("loss"),
						visual.Chart(lossHistoryX, lossHistory),
					),
				),
			)
			var frame bytes.Buffer

			dashboard.Render(&frame, visual.Bounds{X: 1, Y: 1, Width: 110, Height: 48})
			_, _ = os.Stdout.Write([]byte("\033[H"))
			_, _ = os.Stdout.Write(frame.Bytes())
		}

		for {
			select {
			case <-done:
				render(true)
				return
			case <-ticker.C:
				render(false)
			}
		}
	}()

	// track memory
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()

		sampleCount := 1
		sample := func() {
			usedBlocks := shapesCtx.NumAllocatedBlocks()
			stateMu.Lock()
			state.memorySampleHistoryX = append(state.memorySampleHistoryX, sampleCount)
			state.usedBlocksHistory = append(state.usedBlocksHistory, usedBlocks)
			state.version++
			stateMu.Unlock()
			sampleCount++
		}

		sample()
		for {
			select {
			case <-done:
				sample()
				return
			case <-ticker.C:
				sample()
			}
		}
	}()

	for i := range numEpochs {
		epochCtx := shapesCtx.Epoch()
		ix := shapes.FloatRandom(epochCtx, shapes.Shape{32}, 0, float32(X.Shape()[0])).I64(epochCtx)

		// Forward pass
		emb := embLayer.Forward(epochCtx, X.Get(epochCtx, ix))
		yBatch := Y.Get(epochCtx, ix)

		// fmt.Printf("Emb shape: %v\n", emb.Shape())

		yOneHot := shapes.OneHot(epochCtx, yBatch, 27)
		logits := forward(epochCtx, emb)

		lossValue := crossEnthropy(epochCtx, yOneHot, logits)

		lossScalar := lossValue.Get(epochCtx, 0).Item().(float32)

		// Backward pass
		graph := lossValue.Backward(epochCtx)

		// Update parameters
		sgd(graph)

		optimizer.ZeroGrad(shapesCtx, graph)

		if i%100 == 0 {
			stateMu.Lock()
			state.epoch = i
			state.loss = lossScalar
			state.lossHistoryX = append(state.lossHistoryX, i+1)
			state.lossHistory = append(state.lossHistory, int(lossScalar*1000))
			state.version++
			stateMu.Unlock()
		}
		epochCtx.Finish()

	}
	close(done)
	wg.Wait()

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
