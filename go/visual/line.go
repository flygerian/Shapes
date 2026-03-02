package visual

import (
	"fmt"
	"io"
)

// line draws a simple line within its allocated bounds.
type line struct{}

var _ Artefact = (*line)(nil)

// Line creates a line artefact.
func Line() Artefact {
	return &line{}
}

func (line *line) Weight() int {
	return 1
}

func (line *line) Measure(budget Bounds) Bounds {
	size := Bounds{Width: 1, Height: 1}
	if budget.Width > 0 {
		size.Width = budget.Width
	}
	if budget.Height > 0 {
		size.Height = budget.Height
	}
	return size
}

// Render draws a horizontal or vertical line based on the bounds aspect ratio.
func (line *line) Render(target io.Writer, bounds Bounds) {
	if line == nil || bounds.Width <= 0 || bounds.Height <= 0 {
		return
	}

	if bounds.Width >= bounds.Height {
		y := bounds.Y + bounds.Height/2
		for x := bounds.X; x < bounds.X+bounds.Width; x++ {
			fmt.Fprintf(target, "\033[%d;%dH─", y, x)
		}
		return
	}

	x := bounds.X + bounds.Width/2
	for y := bounds.Y; y < bounds.Y+bounds.Height; y++ {
		fmt.Fprintf(target, "\033[%d;%dH│", y, x)
	}
}

// connector draws the graph edge between two boxes.
type connector struct {
	from Bounds
	to   Bounds
}

var _ Artefact = (*connector)(nil)

func graphConnector(from, to Bounds) Artefact {
	return &connector{from: from, to: to}
}

func (line *connector) Weight() int {
	return 1
}

func (line *connector) Measure(budget Bounds) Bounds {
	width := line.to.X - line.from.X
	if width < 0 {
		width = -width
	}
	width++

	height := line.to.Y - line.from.Y
	if height < 0 {
		height = -height
	}
	height++

	if budget.Width > 0 && width > budget.Width {
		width = budget.Width
	}
	if budget.Height > 0 && height > budget.Height {
		height = budget.Height
	}

	if width < 1 {
		width = 1
	}
	if height < 1 {
		height = 1
	}

	return Bounds{Width: width, Height: height}
}

func (line *connector) Render(target io.Writer, bounds Bounds) {
	if line == nil {
		return
	}

	startX := line.from.X + line.from.Width/2 + bounds.X
	startY := line.from.Y + line.from.Height + bounds.Y
	endX := line.to.X + line.to.Width/2 + bounds.X
	endY := line.to.Y - 1 + bounds.Y
	midY := (startY + endY) / 2

	for y := startY; y <= midY; y++ {
		fmt.Fprintf(target, "\033[%d;%dH│", y, startX)
	}

	if startX < endX {
		fmt.Fprintf(target, "\033[%d;%dH└", midY, startX)
		for x := startX + 1; x < endX; x++ {
			fmt.Fprintf(target, "\033[%d;%dH─", midY, x)
		}
		fmt.Fprintf(target, "\033[%d;%dH┐", midY, endX)
	} else if startX > endX {
		fmt.Fprintf(target, "\033[%d;%dH┘", midY, startX)
		for x := endX + 1; x < startX; x++ {
			fmt.Fprintf(target, "\033[%d;%dH─", midY, x)
		}
		fmt.Fprintf(target, "\033[%d;%dH┌", midY, endX)
	}

	for y := midY + 1; y <= endY; y++ {
		fmt.Fprintf(target, "\033[%d;%dH│", y, endX)
	}
}
