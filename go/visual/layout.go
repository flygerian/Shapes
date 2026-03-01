package visual

func measure(artefact Artefact, budget Bounds) Bounds {
	if artefact == nil {
		return Bounds{}
	}

	size := artefact.Measure(budget)
	if size.Width <= 0 {
		size.Width = 1
	}
	if size.Height <= 0 {
		size.Height = 1
	}
	return size
}

func preferredWidth(artefact Artefact) int {
	return measure(artefact, Bounds{}).Width
}

func preferredHeight(artefact Artefact, width int) int {
	return measure(artefact, Bounds{Width: width}).Height
}
