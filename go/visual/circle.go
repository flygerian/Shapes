package visual

import (
	"fmt"
	"io"
)

type circle struct {
	layoutState
}

var _ Artefact = (*circle)(nil)

// Circle creates a small circle artefact.
func Circle() Artefact {
	return &circle{}
}

func (c *circle) Weight() int {
	return 1
}

func (c *circle) Measure(budget Area) Area {
	size := Area{Width: 1, Height: 1}
	if budget.Width > 0 && size.Width > budget.Width {
		size.Width = budget.Width
	}
	if budget.Height > 0 && size.Height > budget.Height {
		size.Height = budget.Height
	}
	return size
}

// Render draws a centered circle glyph within the measured area.
func (c *circle) Render(target io.Writer) {
	frame := layoutOf(c)
	if c == nil || frame.Width <= 0 || frame.Height <= 0 {
		return
	}

	x := frame.X + frame.Width/2
	y := frame.Y + frame.Height/2
	fmt.Fprintf(target, "\033[%d;%dH●", y, x)
}
