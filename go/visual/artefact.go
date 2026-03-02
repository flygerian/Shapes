package visual

import "io"

type Artefact interface {
	Render(target io.Writer, bounds Bounds)
	Measure(budget Bounds) Bounds
	Weight() int
}
