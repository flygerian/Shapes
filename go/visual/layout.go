package visual

import "io"

func measure(artefact Artefact, budget Area) Area {
	if artefact == nil {
		return Area{}
	}

	size := artefact.Measure(budget)
	if size.Width <= 0 {
		size.Width = 1
	}
	if size.Height <= 0 {
		size.Height = 1
	}
	setLayout(artefact, layoutOf(artefact).Point, size)
	return size
}

func preferredWidth(artefact Artefact) int {
	return measure(artefact, Area{}).Width
}

func preferredHeight(artefact Artefact, width int) int {
	return measure(artefact, Area{Width: width}).Height
}

func layoutAt(artefact Artefact, point Point, budget Area) Area {
	size := measure(artefact, budget)
	setLayout(artefact, point, size)
	return size
}

// RenderAt measures an artefact with the provided growth budget, places it at point, and renders it.
func RenderAt(target io.Writer, artefact Artefact, point Point, budget Area) Area {
	size := layoutAt(artefact, point, budget)
	if artefact == nil {
		return size
	}
	artefact.Render(target)
	return size
}
