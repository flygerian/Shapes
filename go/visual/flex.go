package visual

import "io"

// Direction controls how flex lays out its children.
type Direction int

const (
	DirectionRow Direction = iota
	DirectionColumn
)

const flexGap = 1

type flex struct {
	layoutState
	direction Direction
	children  []Artefact
	minHeight int
}

type FlexOptions struct {
	Direction Direction
	Children  []Artefact
	MinHeight int
}

var _ Artefact = (*flex)(nil)

// Flex creates a flex container that lays out children in a row or column.
func Flex(options FlexOptions) Artefact {
	childCopy := make([]Artefact, len(options.Children))
	copy(childCopy, options.Children)

	direction := DirectionRow
	if options.Direction != DirectionRow {
		direction = options.Direction
	}

	return &flex{
		direction: direction,
		children:  childCopy,
		minHeight: max(options.MinHeight, 0),
	}
}

func (f *flex) Weight() int {
	return 1
}

func (f *flex) Measure(budget Area) Area {
	if f == nil {
		return Area{}
	}
	if len(f.children) == 0 {
		return Area{Height: f.minHeight}
	}

	var size Area
	switch f.direction {
	case DirectionColumn:
		size = f.measureColumn(budget)
	default:
		size = f.measureRow(budget)
	}

	if size.Height < f.minHeight {
		size.Height = f.minHeight
	}
	return size
}

func (f *flex) measureRow(budget Area) Area {
	count := len(f.children)
	if count == 0 {
		return Area{}
	}

	totalWidth := 0
	maxHeight := 1

	if budget.Width > 0 {
		slotWidths := partitionBudgetWeighted(budget.Width, flexGap, f.children)
		for i, child := range f.children {
			if i > 0 {
				totalWidth += flexGap
			}
			size := measure(child, Area{
				Width:  slotWidths[i],
				Height: budget.Height,
			})
			totalWidth += size.Width
			if size.Height > maxHeight {
				maxHeight = size.Height
			}
		}
		if totalWidth > budget.Width {
			totalWidth = budget.Width
		}
		return Area{Width: totalWidth, Height: maxHeight}
	}

	for i, child := range f.children {
		if i > 0 {
			totalWidth += flexGap
		}
		size := measure(child, Area{Height: budget.Height})
		totalWidth += size.Width
		if size.Height > maxHeight {
			maxHeight = size.Height
		}
	}

	return Area{Width: totalWidth, Height: maxHeight}
}

func (f *flex) measureColumn(budget Area) Area {
	count := len(f.children)
	if count == 0 {
		return Area{}
	}

	maxWidth := 1
	totalHeight := 0

	if budget.Height > 0 {
		slotHeights := partitionBudgetWeighted(budget.Height, flexGap, f.children)
		for i, child := range f.children {
			if i > 0 {
				totalHeight += flexGap
			}
			size := measure(child, Area{
				Width:  budget.Width,
				Height: slotHeights[i],
			})
			if size.Width > maxWidth {
				maxWidth = size.Width
			}
			totalHeight += size.Height
		}
		if totalHeight > budget.Height {
			totalHeight = budget.Height
		}
		return Area{Width: maxWidth, Height: totalHeight}
	}

	for i, child := range f.children {
		if i > 0 {
			totalHeight += flexGap
		}
		size := measure(child, Area{Width: budget.Width})
		if size.Width > maxWidth {
			maxWidth = size.Width
		}
		totalHeight += size.Height
	}

	return Area{Width: maxWidth, Height: totalHeight}
}

// Render assigns bounds to each child and renders them.
func (f *flex) Render(target io.Writer) {
	if f == nil || len(f.children) == 0 {
		return
	}

	switch f.direction {
	case DirectionColumn:
		f.renderColumn(target)
	default:
		f.renderRow(target)
	}
}

func (f *flex) renderRow(target io.Writer) {
	frame := layoutOf(f)
	x := frame.X
	for i, child := range f.children {
		if i > 0 {
			x += flexGap
		}
		childFrame := layoutOf(child)
		setLayout(child, Point{X: x, Y: frame.Y}, childFrame.Area)
		child.Render(target)
		x += childFrame.Width
	}
}

func (f *flex) renderColumn(target io.Writer) {
	frame := layoutOf(f)
	y := frame.Y
	for i, child := range f.children {
		if i > 0 {
			y += flexGap
		}
		childFrame := layoutOf(child)
		setLayout(child, Point{X: frame.X, Y: y}, childFrame.Area)
		child.Render(target)
		y += childFrame.Height
	}
}

func partitionBudgetWeighted(total, gap int, children []Artefact) []int {
	count := len(children)
	slots := make([]int, count)
	if count == 0 {
		return slots
	}

	available := total - gap*(count-1)
	if available <= 0 {
		return slots
	}

	weights := make([]int, count)
	totalWeight := 0
	for i, child := range children {
		w := child.Weight()
		if w < 0 {
			w = 0
		}
		weights[i] = w
		totalWeight += w
	}

	if totalWeight == 0 {
		for i := range count {
			weights[i] = 1
		}
		totalWeight = count
	}

	remainders := make([]int, count)
	allocated := 0
	for i := range count {
		product := available * weights[i]
		slots[i] = product / totalWeight
		remainders[i] = product % totalWeight
		allocated += slots[i]
	}

	for remaining := available - allocated; remaining > 0; remaining-- {
		best := 0
		for i := 1; i < count; i++ {
			if remainders[i] > remainders[best] {
				best = i
			}
		}
		slots[best]++
		remainders[best] = -1
	}

	return slots
}
