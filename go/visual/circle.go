package visual

import (
	"fmt"
	"io"
)

type circle struct{}

var _ Artefact = (*circle)(nil)

// Circle creates a small circle artefact.
func Circle() Artefact {
	return &circle{}
}

func (c *circle) Measure(budget Bounds) Bounds {
	size := Bounds{Width: 1, Height: 1}
	if budget.Width > 0 && size.Width > budget.Width {
		size.Width = budget.Width
	}
	if budget.Height > 0 && size.Height > budget.Height {
		size.Height = budget.Height
	}
	return size
}

// Render draws a centered circle glyph within the provided bounds.
func (c *circle) Render(target io.Writer, bounds Bounds) {
	if c == nil || bounds.Width <= 0 || bounds.Height <= 0 {
		return
	}

	x := bounds.X + bounds.Width/2
	y := bounds.Y + bounds.Height/2
	fmt.Fprintf(target, "\033[%d;%dH●", y, x)
}
