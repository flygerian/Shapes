package extract

import (
	shapes "github.com/flygerian/shapes"
)

func Parameters(c *shapes.ComputationGraph) []*shapes.Tensor {
	var parameters []*shapes.Tensor
	for i := range len(c.Nodes) {
		p := c.Nodes[i].Parameters
		if p == nil {
			continue
		}

		parameters = append(parameters, p...)
	}

	return parameters
}
