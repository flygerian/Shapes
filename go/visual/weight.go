package visual

import "io"

type weightedArtefact struct {
	layoutState
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

func (w *weightedArtefact) Measure(budget Area) Area {
	if w == nil || w.child == nil {
		return Area{}
	}
	return measure(w.child, budget)
}

func (w *weightedArtefact) Render(target io.Writer) {
	if w == nil || w.child == nil {
		return
	}
	frame := layoutOf(w)
	setLayout(w.child, frame.Point, frame.Area)
	w.child.Render(target)
}
