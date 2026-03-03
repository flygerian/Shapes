package visual

import (
	"fmt"
	"io"
	"os"
	"strings"

	shapes "github.com/flygerian/shapes"
)

const (
	maxBoxWidth = 50
	vSpacing    = 5
	maxNodes    = 64
	screenWidth = 220
)

// Visualize renders the computation graph rooted at t to the terminal (os.Stdout).
func Visualize(ctx shapes.Context, t shapes.Tensor) {
	VisualizeTo(ctx, os.Stdout, t)
}

// VisualizeTo renders the computation graph rooted at t to the given writer.
func VisualizeTo(ctx shapes.Context, w io.Writer, t shapes.ComputationGraphNode) {
	type entry struct {
		tensor    shapes.ComputationGraphNode
		level     int
		parentIdx int
	}

	queue := make([]entry, 0, maxNodes)
	queue = append(queue, entry{tensor: t, level: 0, parentIdx: -1})

	// BFS expand
	for i := 0; i < len(queue) && len(queue) < maxNodes; i++ {
		node := queue[i].tensor
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

	// Count nodes per level and find max level.
	maxLevel := 0
	levelCount := make(map[int]int)
	levelNodes := make(map[int][]int)
	for i, e := range queue {
		levelCount[e.level]++
		levelNodes[e.level] = append(levelNodes[e.level], i)
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
		available := (screenWidth - (maxNodesAtLevel-1)*flexGap) / maxNodesAtLevel
		if available < boxWidth {
			boxWidth = available
		}
		if boxWidth < 20 {
			boxWidth = 20
		}
	}

	// Build rows with flex and keep per-node bounds for edge rendering.
	rows := make([]Artefact, maxLevel+1)
	rowBounds := make([]Bounds, maxLevel+1)
	boxBounds := make([]Bounds, len(queue))
	currentY := 2
	bottomY := 2

	for lv := range maxLevel + 1 {
		indices := levelNodes[lv]
		nodesAtLevel := len(indices)
		if nodesAtLevel == 0 {
			continue
		}

		children := make([]Artefact, nodesAtLevel)
		childSizes := make([]Bounds, nodesAtLevel)
		rowHeight := 1
		totalWidth := 0
		for idx, nodeIdx := range indices {
			top, bottom := displas(ctx, queue[nodeIdx].tensor.(shapes.Tensor))
			var content Artefact = Text(top)
			if bottom != "" {
				content = Flex(FlexOptions{
					Direction: DirectionColumn,
					Children:  []Artefact{Text(top), Text(bottom)},
				})
			}
			nodeBox := Box(BoxOptions{
				Child: content,
			})
			children[idx] = nodeBox

			size := measure(nodeBox, Bounds{Width: boxWidth})
			childSizes[idx] = size
			if size.Height > rowHeight {
				rowHeight = size.Height
			}
			if idx > 0 {
				totalWidth += flexGap
			}
			totalWidth += size.Width
		}

		startX := (screenWidth - totalWidth) / 2
		if startX < 1 {
			startX = 1
		}

		x := startX
		for idx, nodeIdx := range indices {
			boxBounds[nodeIdx] = Bounds{
				X:      x,
				Y:      currentY,
				Width:  childSizes[idx].Width,
				Height: childSizes[idx].Height,
			}
			x += childSizes[idx].Width + flexGap
		}

		rows[lv] = Flex(FlexOptions{
			Direction: DirectionRow,
			Children:  children,
		})
		rowBounds[lv] = Bounds{
			X:      startX,
			Y:      currentY,
			Width:  totalWidth,
			Height: 0,
		}

		levelBottom := currentY + rowHeight
		if levelBottom > bottomY {
			bottomY = levelBottom
		}
		currentY += rowHeight + vSpacing
	}

	ClearScreen(w)

	for lv := range rows {
		if rows[lv] != nil {
			rows[lv].Render(w, rowBounds[lv])
		}
	}

	for i, e := range queue {
		if e.parentIdx >= 0 {
			graphConnector(boxBounds[e.parentIdx], boxBounds[i]).Render(w, Bounds{})
		}
	}

	// Move cursor below the drawing.
	fmt.Fprintf(w, "\033[%d;%dH", bottomY+1, 0)
}

// displas builds the top and bottom text for a tensor node.
func displas(ctx shapes.Context, t shapes.Tensor) (top, bottom string) {
	label := t.Label()
	if label == "" {
		label = "?"
	}

	val := formatScalar(ctx, t)

	node := t
	if node == nil {
		return fmt.Sprintf("%s | %s", label, val), ""
	}

	// Build top line with grad if available
	if node.Grad() != nil {
		gradVal := formatScalar(ctx, node.Grad().(shapes.Tensor))
		top = fmt.Sprintf("%s | %s | grad [%s]", label, val, gradVal)
	} else {
		top = fmt.Sprintf("%s | %s", label, val)
	}

	// Bottom line shows op name if this is not a leaf
	if len(node.Inputs()) > 0 && node.Op() != shapes.OpNone {
		bottom = fmt.Sprintf("(%s)", node.Op())
	}

	return top, bottom
}

// formatScalar reads the scalar (index-0) value from a tensor and trims trailing zeros.
func formatScalar(ctx shapes.Context, t shapes.Tensor) string {
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
func Print(ctx shapes.Context, t shapes.Tensor) {
	PrintTo(ctx, os.Stdout, t)
}

// PrintTo renders the tensor's contents to the given writer in a nested bracket format.
func PrintTo(ctx shapes.Context, w io.Writer, t shapes.Tensor) {
	shape := t.Shape()
	if len(shape) == 0 {
		return
	}

	if t.Label() != "" {
		fmt.Fprintf(w, "%s: ", t)
	}

	coords := make([]uint, len(shape))
	printRecursive(ctx, w, t, shape, coords, 0)
	fmt.Fprintln(w)
}

// printRecursive walks dimension by dimension, printing brackets and values.
func printRecursive(ctx shapes.Context, w io.Writer, t shapes.Tensor, shape, coords []uint, dim int) {
	if dim == len(shape)-1 {
		fmt.Fprint(w, "[")
		for i := range shape[dim] {
			if i > 0 {
				fmt.Fprint(w, ", ")
			}
			coords[dim] = uint(i)
			fmt.Fprint(w, formatValue(ctx, t, coords))
		}
		fmt.Fprint(w, "]")
		return
	}

	fmt.Fprint(w, "[")
	for i := range shape[dim] {
		if i > 0 {
			fmt.Fprint(w, ",\n")
			fmt.Fprint(w, strings.Repeat(" ", dim+1))
			if t.Label() != "" {
				fmt.Fprint(w, strings.Repeat(" ", len(t.Label())+2))
			}
		}
		coords[dim] = uint(i)
		printRecursive(ctx, w, t, shape, coords, dim+1)
	}
	fmt.Fprint(w, "]")
}

// formatValue reads a single element from the tensor and returns its string representation.
func formatValue(ctx shapes.Context, t shapes.Tensor, coords []uint) string {
	// Convert []uint to []interface{} for Get function
	args := make([]interface{}, len(coords))
	for i, c := range coords {
		args[i] = c
	}
	switch t.Dtype() {
	case shapes.DtypeF16, shapes.DtypeF32, shapes.DtypeF64:
		v := t.Get(ctx, args...).Item().(float32)
		s := fmt.Sprintf("%f", v)
		if idx := strings.IndexByte(s, '.'); idx >= 0 {
			s = strings.TrimRight(s, "0")
			if s[len(s)-1] == '.' {
				s += "0"
			}
		}
		return s
	default:
		v := t.Get(ctx, args...).Item().(int8)
		return fmt.Sprintf("%d", v)
	}
}
