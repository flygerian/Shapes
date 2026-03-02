package visual

import "io"

type weightedArtefact struct {
	child  Artefact
	weight int
}

var _ Artefact = (*weightedArtefact)(nil)

type WeightOptions struct {
	Child  Artefact
	Weight int
}

// Weighted overrides an artefact's layout weight for flex budget distribution.
func Weighted(options WeightOptions) Artefact {
	return &weightedArtefact{
		child:  options.Child,
		weight: options.Weight,
	}
}

func (w *weightedArtefact) Weight() int {
	if w == nil || w.weight < 0 {
		return 0
	}
	return w.weight
}

func (w *weightedArtefact) Measure(budget Bounds) Bounds {
	if w == nil || w.child == nil {
		return Bounds{}
	}
	return w.child.Measure(budget)
}

func (w *weightedArtefact) Render(target io.Writer, bounds Bounds) {
	if w == nil || w.child == nil {
		return
	}
	w.child.Render(target, bounds)
}
