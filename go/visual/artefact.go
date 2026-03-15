package visual

import "io"

type Artefact interface {
	Render(target io.Writer)
	Measure(area Area) Area
	Weight() int
}
