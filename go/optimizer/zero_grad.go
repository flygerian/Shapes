package optimizer

/*
#include "shapes.h"
#include "common.h"
#include <string.h>

*/
import "C"

import (
	"fmt"
	"os"
	shapes "github.com/flygerian/shapes"
	"time"
)

func shouldLogZeroGrad() bool {
	value := os.Getenv("SHAPES_LOG_ZERO_GRAD")
	return value != "" && value != "0"
}

func ZeroGrad(ctx shapes.Context, cg shapes.ComputationGraph) {
	start := time.Now()
	zeroed := 0

	for _, node := range cg {
		// Instead of allocating a new zero tensor, just zero out the existing gradient's values
		if node.Grad() != nil {
			node.SetValuesToZero()
			zeroed++
		}

		// Zero hidden state (parameter) gradients — these are not graph nodes themselves
		// so they are not visited by the loop above, but they accumulate gradients via backward.
		for _, p := range node.HiddenState() {
			if p.Grad() != nil {
				p.SetValuesToZero()
				zeroed++
			}
		}
	}

	if shouldLogZeroGrad() {
		fmt.Fprintf(os.Stderr, "[zeroGrad] tensors=%d ms=%.3f\n", zeroed,
			float64(time.Since(start))/float64(time.Millisecond))
	}
}
