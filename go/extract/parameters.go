package extract

import (
	shapes "github.com/flygerian/shapes"
)

func Parameters(c shapes.ComputationGraph) []shapes.Tensor {
	var parameters []shapes.Tensor
	for i := range len(c) {
		h := c[i].HiddenState()
		if h == nil {
			continue
		}

		parameters = append(parameters, h...)
	}

	return parameters
}
