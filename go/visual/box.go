package visual

import (
	"fmt"
	"io"
	"strings"
)

// Box represents a drawable box in the terminal.
type box struct {
	layoutState
	child Artefact
}

type BoxOptions struct {
	Child Artefact
}

var _ Artefact = (*box)(nil)

// Box creates a box artefact with top and bottom text content.
func Box(options BoxOptions) Artefact {
	return &box{
		child: options.Child,
	}
}

func (box *box) Weight() int {
	return 1
}

func (box *box) Measure(budget Area) Area {
	if box == nil {
		return Area{Width: 2, Height: 2}
	}

	innerBudget := budget
	if innerBudget.Width > 0 {
		innerBudget.Width -= 2
	}
	if innerBudget.Width < 0 {
		innerBudget.Width = 0
	}
	if innerBudget.Height > 0 {
		innerBudget.Height -= 2
	}
	if innerBudget.Height < 0 {
		innerBudget.Height = 0
	}

	childSize := Area{Width: 1, Height: 1}
	if box.child != nil {
		childSize = measure(box.child, innerBudget)
	}

	width := childSize.Width + 2
	height := childSize.Height + 2

	return Area{Width: width, Height: height}
}

// Render draws the box into the measured area.
func (box *box) Render(target io.Writer) {
	frame := layoutOf(box)
	if box == nil {
		return
	}
	if frame.Width < 2 || frame.Height < 2 {
		return
	}
	innerHeight := frame.Height - 2
	if innerHeight <= 0 {
		return
	}

	x := frame.X
	y := frame.Y

	// Top border
	fmt.Fprintf(target, "\033[%d;%dH", y, x)
	fmt.Fprint(target, "┌")
	fmt.Fprint(target, strings.Repeat("─", frame.Width-2))
	fmt.Fprint(target, "┐")

	for i := 1; i <= innerHeight; i++ {
		fmt.Fprintf(target, "\033[%d;%dH", y+i, x)
		fmt.Fprint(target, "│")
		fmt.Fprint(target, strings.Repeat(" ", frame.Width-2))
		fmt.Fprint(target, "│")
	}

	if box.child != nil {
		childFrame := layoutOf(box.child)
		setLayout(box.child, Point{X: x + 1, Y: y + 1}, childFrame.Area)
		box.child.Render(target)
	}

	// Bottom border
	fmt.Fprintf(target, "\033[%d;%dH", y+frame.Height-1, x)
	fmt.Fprint(target, "└")
	fmt.Fprint(target, strings.Repeat("─", frame.Width-2))
	fmt.Fprint(target, "┘")
}
