package visual

import (
	"fmt"
	"io"
	"strings"
)

// Box represents a drawable box in the terminal.
type box struct {
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

func (box *box) Measure(budget Bounds) Bounds {
	if box == nil {
		return Bounds{Width: 2, Height: 2}
	}

	innerBudget := budget
	if innerBudget.Width > 0 {
		innerBudget.Width -= 2
	}
	if innerBudget.Width < 0 {
		innerBudget.Width = 0
	}
	innerBudget.Height = 0

	childSize := Bounds{Width: 1, Height: 1}
	if box.child != nil {
		childSize = measure(box.child, innerBudget)
	}

	width := childSize.Width + 2
	height := childSize.Height + 2

	return Bounds{Width: width, Height: height}
}

// Render draws the box into the provided bounds.
func (box *box) Render(target io.Writer, bounds Bounds) {
	if box == nil {
		return
	}
	size := measure(box, Bounds{Width: bounds.Width, Height: bounds.Height})
	bounds.Width = size.Width
	bounds.Height = size.Height
	if bounds.Width < 2 || bounds.Height < 2 {
		return
	}
	innerHeight := bounds.Height - 2
	if innerHeight <= 0 {
		return
	}

	x := bounds.X
	y := bounds.Y

	// Top border
	fmt.Fprintf(target, "\033[%d;%dH", y, x)
	fmt.Fprint(target, "┌")
	fmt.Fprint(target, strings.Repeat("─", bounds.Width-2))
	fmt.Fprint(target, "┐")

	for i := 1; i <= innerHeight; i++ {
		fmt.Fprintf(target, "\033[%d;%dH", y+i, x)
		fmt.Fprint(target, "│")
		fmt.Fprint(target, strings.Repeat(" ", bounds.Width-2))
		fmt.Fprint(target, "│")
	}

	if box.child != nil {
		childSize := measure(box.child, Bounds{
			Width:  bounds.Width - 2,
			Height: 0,
		})
		contentBounds := Bounds{
			X:      x + 1,
			Y:      y + 1,
			Width:  childSize.Width,
			Height: childSize.Height,
		}
		box.child.Render(target, contentBounds)
	}

	// Bottom border
	fmt.Fprintf(target, "\033[%d;%dH", y+bounds.Height-1, x)
	fmt.Fprint(target, "└")
	fmt.Fprint(target, strings.Repeat("─", bounds.Width-2))
	fmt.Fprint(target, "┘")
}
