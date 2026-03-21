package layer

import shapes "github.com/flygerian/shapes"

type embedding struct {
	ctx       shapes.Context
	c         shapes.Tensor // embedding matrix [vocabSize, embDim]
	vocabSize int
	embDim    int
}

// Forward looks up embeddings for each index in x.
// x must be an integer tensor of indices.
// Returns a tensor of shape [..., embDim].
func (e *embedding) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	out := e.c.Get(ctx, x)
	out.AttachHiddenState(e.c)
	return out
}

// Embedding creates an embedding layer with a randomly initialised matrix of
// shape [vocabSize, embDim].
func Embedding(ctx shapes.Context, vocabSize int, embDim int) *embedding {
	c := shapes.FloatRandom(ctx, shapes.Shape{uint(vocabSize), uint(embDim)})
	return &embedding{ctx: ctx, c: c, vocabSize: vocabSize, embDim: embDim}
}
