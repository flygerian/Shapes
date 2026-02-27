package visual

import (
	"fmt"
	"io"
	"os"
	"strings"
	"unicode/utf8"

	shapes "github.com/flygerian/shapes"
)

const (
	maxBoxWidth = 50
	boxHeight   = 7
	vSpacing    = 5
	maxNodes    = 64
	boxGap      = 4
	screenWidth = 220
)

// Box represents a drawable box in the terminal.
type Box struct {
	x, y          int
	width, height int
	topText       string
	bottomText    string
}

// ClearScreen clears the terminal and moves cursor to origin.
func ClearScreen(w io.Writer) {
	fmt.Fprint(w, "\033[2J\033[H")
}

// DrawBox renders a single box with centered top/bottom text and a separator.
func DrawBox(w io.Writer, box *Box) {
	// Top border
	fmt.Fprintf(w, "\033[%d;%dH", box.y, box.x)
	fmt.Fprint(w, "┌")
	fmt.Fprint(w, strings.Repeat("─", box.width-2))
	fmt.Fprint(w, "┐")

	for i := 1; i < box.height-1; i++ {
		fmt.Fprintf(w, "\033[%d;%dH", box.y+i, box.x)

		switch i {
		case 2:
			// Top text line
			fmt.Fprint(w, "│")
			writeCenter(w, box.topText, box.width-2)
			fmt.Fprint(w, "│")
		case 3:
			// Separator
			fmt.Fprint(w, "├")
			fmt.Fprint(w, strings.Repeat("─", box.width-2))
			fmt.Fprint(w, "┤")
		case 4:
			// Bottom text line
			fmt.Fprint(w, "│")
			writeCenter(w, box.bottomText, box.width-2)
			fmt.Fprint(w, "│")
		default:
			// Empty line
			fmt.Fprint(w, "│")
			fmt.Fprint(w, strings.Repeat(" ", box.width-2))
			fmt.Fprint(w, "│")
		}
	}

	// Bottom border
	fmt.Fprintf(w, "\033[%d;%dH", box.y+box.height-1, box.x)
	fmt.Fprint(w, "└")
	fmt.Fprint(w, strings.Repeat("─", box.width-2))
	fmt.Fprint(w, "┘")
}

// DrawLine connects the bottom-center of 'from' to the top-center of 'to'.
func DrawLine(w io.Writer, from, to *Box) {
	startX := from.x + from.width/2
	startY := from.y + from.height

	endX := to.x + to.width/2
	endY := to.y - 1

	midY := (startY + endY) / 2

	// Vertical segment down from source
	for y := startY; y <= midY; y++ {
		fmt.Fprintf(w, "\033[%d;%dH│", y, startX)
	}

	// Horizontal segment
	if startX < endX {
		fmt.Fprintf(w, "\033[%d;%dH└", midY, startX)
		for x := startX + 1; x < endX; x++ {
			fmt.Fprintf(w, "\033[%d;%dH─", midY, x)
		}
		fmt.Fprintf(w, "\033[%d;%dH┐", midY, endX)
	} else if startX > endX {
		fmt.Fprintf(w, "\033[%d;%dH┘", midY, startX)
		for x := endX + 1; x < startX; x++ {
			fmt.Fprintf(w, "\033[%d;%dH─", midY, x)
		}
		fmt.Fprintf(w, "\033[%d;%dH┌", midY, endX)
	}

	// Vertical segment down to destination
	for y := midY + 1; y <= endY; y++ {
		fmt.Fprintf(w, "\033[%d;%dH│", y, endX)
	}
}

// Visualize renders the computation graph rooted at t to the terminal (os.Stdout).
func Visualize(ctx *shapes.MainContext, t *shapes.Tensor) {
	VisualizeTo(ctx, os.Stdout, t)
}

// VisualizeTo renders the computation graph rooted at t to the given writer.
func VisualizeTo(ctx *shapes.MainContext, w io.Writer, t *shapes.Tensor) {
	type entry struct {
		tensor    *shapes.Tensor
		level     int
		parentIdx int
	}

	queue := make([]entry, 0, maxNodes)
	queue = append(queue, entry{tensor: t, level: 0, parentIdx: -1})

	// BFS expand
	for i := 0; i < len(queue) && len(queue) < maxNodes; i++ {
		node := queue[i].tensor.Computation
		if node == nil {
			continue
		}
		for _, inp := range node.Inputs() {
			if len(queue) >= maxNodes {
				break
			}
			queue = append(queue, entry{tensor: inp, level: queue[i].level + 1, parentIdx: i})
		}
	}

	// Count nodes per level and find max level
	maxLevel := 0
	levelCount := make(map[int]int)
	for _, e := range queue {
		levelCount[e.level]++
		if e.level > maxLevel {
			maxLevel = e.level
		}
	}

	// Find the widest level to determine box width
	maxNodesAtLevel := 0
	for _, c := range levelCount {
		if c > maxNodesAtLevel {
			maxNodesAtLevel = c
		}
	}

	boxWidth := maxBoxWidth
	if maxNodesAtLevel > 1 {
		available := (screenWidth - (maxNodesAtLevel-1)*boxGap) / maxNodesAtLevel
		if available < boxWidth {
			boxWidth = available
		}
		if boxWidth < 20 {
			boxWidth = 20
		}
	}

	// Assign box positions
	levelIdx := make(map[int]int)
	boxes := make([]Box, len(queue))

	for i, e := range queue {
		lv := e.level
		nodesAtLevel := levelCount[lv]
		idx := levelIdx[lv]
		levelIdx[lv]++

		totalWidth := nodesAtLevel*boxWidth + (nodesAtLevel-1)*boxGap
		startX := (screenWidth - totalWidth) / 2
		if startX < 1 {
			startX = 1
		}

		top, bottom := displayLabels(ctx, queue[i].tensor)
		boxes[i] = Box{
			x:          startX + idx*(boxWidth+boxGap),
			y:          2 + lv*(boxHeight+vSpacing),
			width:      boxWidth,
			height:     boxHeight,
			topText:    top,
			bottomText: bottom,
		}
	}

	ClearScreen(w)

	for i := range boxes {
		DrawBox(w, &boxes[i])
	}

	for i, e := range queue {
		if e.parentIdx >= 0 {
			DrawLine(w, &boxes[e.parentIdx], &boxes[i])
		}
	}

	// Move cursor below the drawing
	bottomY := 2 + (maxLevel+1)*(boxHeight+vSpacing) + 1
	fmt.Fprintf(w, "\033[%d;%dH", bottomY, 0)
}

// displayLabels builds the top and bottom text for a tensor node.
func displayLabels(ctx *shapes.MainContext, t *shapes.Tensor) (top, bottom string) {
	label := t.Label
	if label == "" {
		label = "?"
	}

	val := formatScalar(ctx, t)

	node := t.Computation
	if node == nil {
		return fmt.Sprintf("%s | %s", label, val), ""
	}

	// Build top line with grad if available
	if node.Grad() != nil {
		gradVal := formatScalar(ctx, node.Grad().(*shapes.Tensor))
		top = fmt.Sprintf("%s | %s | grad [%s]", label, val, gradVal)
	} else {
		top = fmt.Sprintf("%s | %s", label, val)
	}

	// Bottom line shows op name if this is not a leaf
	if len(node.Inputs()) > 0 && node.Op != shapes.OpNone {
		bottom = fmt.Sprintf("(%s)", node.Op)
	}

	return top, bottom
}

// formatScalar reads the scalar (index-0) value from a tensor and trims trailing zeros.
func formatScalar(ctx *shapes.MainContext, t *shapes.Tensor) string {
	v := t.Get(ctx, 0).Item().(float32)
	s := fmt.Sprintf("%f", v)
	// Trim trailing zeros but keep at least one digit after decimal
	if idx := strings.IndexByte(s, '.'); idx >= 0 {
		s = strings.TrimRight(s, "0")
		if s[len(s)-1] == '.' {
			s += "0"
		}
	}
	return s
}

// Print renders the tensor's contents to os.Stdout in a nested bracket format.
func Print(t *shapes.WrappedTensor) {
	PrintTo(os.Stdout, t)
}

// PrintTo renders the tensor's contents to the given writer in a nested bracket format.
func PrintTo(w io.Writer, t *shapes.WrappedTensor) {
	shape := t.Shape()
	if len(shape) == 0 {
		return
	}

	if t.Tensor().Label != "" {
		fmt.Fprintf(w, "%s: ", t.Tensor().Label)
	}

	coords := make([]uint32, len(shape))
	printRecursive(w, t, shape, coords, 0)
	fmt.Fprintln(w)
}

// printRecursive walks dimension by dimension, printing brackets and values.
func printRecursive(w io.Writer, t *shapes.WrappedTensor, shape, coords []uint32, dim int) {
	if dim == len(shape)-1 {
		fmt.Fprint(w, "[")
		for i := range shape[dim] {
			if i > 0 {
				fmt.Fprint(w, ", ")
			}
			coords[dim] = uint32(i)
			fmt.Fprint(w, formatValue(t, coords))
		}
		fmt.Fprint(w, "]")
		return
	}

	fmt.Fprint(w, "[")
	for i := range shape[dim] {
		if i > 0 {
			fmt.Fprint(w, ",\n")
			fmt.Fprint(w, strings.Repeat(" ", dim+1))
			if t.Tensor().Label != "" {
				fmt.Fprint(w, strings.Repeat(" ", len(t.Tensor().Label)+2))
			}
		}
		coords[dim] = uint32(i)
		printRecursive(w, t, shape, coords, dim+1)
	}
	fmt.Fprint(w, "]")
}

// formatValue reads a single element from the tensor and returns its string representation.
func formatValue(t *shapes.WrappedTensor, coords []uint32) string {
	// Convert []uint32 to []interface{} for Get function
	args := make([]interface{}, len(coords))
	for i, c := range coords {
		args[i] = c
	}
	switch t.Dtype() {
	case shapes.DtypeF16, shapes.DtypeF32, shapes.DtypeF64:
		v := t.Get(args...).Item().(float32)
		s := fmt.Sprintf("%f", v)
		if idx := strings.IndexByte(s, '.'); idx >= 0 {
			s = strings.TrimRight(s, "0")
			if s[len(s)-1] == '.' {
				s += "0"
			}
		}
		return s
	default:
		v := t.Get(args...).Item().(int8)
		return fmt.Sprintf("%d", v)
	}
}

// writeCenter writes text centered in the given width, or truncated if too long.
func writeCenter(w io.Writer, text string, width int) {
	if text == "" {
		fmt.Fprint(w, strings.Repeat(" ", width))
		return
	}
	textLen := utf8.RuneCountInString(text)
	if textLen > width {
		// Truncate to width runes
		r := []rune(text)
		fmt.Fprint(w, string(r[:width]))
		return
	}
	pad := (width - textLen) / 2
	fmt.Fprint(w, strings.Repeat(" ", pad))
	fmt.Fprint(w, text)
	fmt.Fprint(w, strings.Repeat(" ", width-pad-textLen))
}
