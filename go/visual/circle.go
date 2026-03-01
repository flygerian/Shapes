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

// Render draws a centered circle glyph within the provided bounds.
func (c *circle) Render(target io.Writer, bounds Bounds) {
	if c == nil || bounds.Width <= 0 || bounds.Height <= 0 {
		return
	}

	x := bounds.X + bounds.Width/2
	y := bounds.Y + bounds.Height/2
	fmt.Fprintf(target, "\033[%d;%dH●", y, x)
}
