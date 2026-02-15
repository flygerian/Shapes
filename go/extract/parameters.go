package extract

import "github.com/flygerian/shapes/tensor"

func Parameters(c *tensor.ComputationGraph) []*tensor.Tensor {
	var parameters []*tensor.Tensor
	for i := range len(c.Nodes) {
		p := c.Nodes[i].Parameters
		if p == nil {
			continue
		}

		parameters = append(parameters, p...)
	}

	return parameters
}
