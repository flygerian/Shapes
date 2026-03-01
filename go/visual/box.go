package visual

import (
	"fmt"
	"io"
	"strings"
	"unicode/utf8"
)

// Box represents a drawable box in the terminal.
type box struct {
	topText    string
	bottomText string
}

var _ Artefact = (*box)(nil)

// Box creates a box artefact with top and bottom text content.
func Box(topText, bottomText string) Artefact {
	return &box{
		topText:    topText,
		bottomText: bottomText,
	}
}

// Render draws the box into the provided bounds.
func (box *box) Render(target io.Writer, bounds Bounds) {
	if box == nil {
		return
	}
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

	topTextRow := 1
	separatorRow := -1
	bottomTextRow := innerHeight
	if innerHeight >= 3 {
		topTextRow = innerHeight/2 - 1
		if topTextRow < 1 {
			topTextRow = 1
		}
		separatorRow = topTextRow + 1
		bottomTextRow = separatorRow + 1
	}

	for i := 1; i <= innerHeight; i++ {
		fmt.Fprintf(target, "\033[%d;%dH", y+i, x)

		switch i {
		case topTextRow:
			fmt.Fprint(target, "│")
			writeCenter(target, box.topText, bounds.Width-2)
			fmt.Fprint(target, "│")
		case separatorRow:
			fmt.Fprint(target, "├")
			fmt.Fprint(target, strings.Repeat("─", bounds.Width-2))
			fmt.Fprint(target, "┤")
		case bottomTextRow:
			fmt.Fprint(target, "│")
			writeCenter(target, box.bottomText, bounds.Width-2)
			fmt.Fprint(target, "│")
		default:
			fmt.Fprint(target, "│")
			fmt.Fprint(target, strings.Repeat(" ", bounds.Width-2))
			fmt.Fprint(target, "│")
		}
	}

	// Bottom border
	fmt.Fprintf(target, "\033[%d;%dH", y+bounds.Height-1, x)
	fmt.Fprint(target, "└")
	fmt.Fprint(target, strings.Repeat("─", bounds.Width-2))
	fmt.Fprint(target, "┘")
}

// writeCenter writes text centered in the given width, or truncated if too long.
func writeCenter(w io.Writer, text string, width int) {
	if text == "" {
		fmt.Fprint(w, strings.Repeat(" ", width))
		return
	}
	textLen := utf8.RuneCountInString(text)
	if textLen > width {
		r := []rune(text)
		fmt.Fprint(w, string(r[:width]))
		return
	}
	pad := (width - textLen) / 2
	fmt.Fprint(w, strings.Repeat(" ", pad))
	fmt.Fprint(w, text)
	fmt.Fprint(w, strings.Repeat(" ", width-pad-textLen))
}
