package shapesnn
import "../"

LayerWithState :: struct {
	weights:   shapes.Tensor,
	bias:      shapes.Tensor,
	withBias:  bool,
	layerData: any,
}

LayerWithNoState :: struct {}

Layer :: union {
	LayerWithNoState,
	LayerWithState,
}

