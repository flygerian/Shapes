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

func (f *flex) Measure(budget Bounds) Bounds {
	if f == nil {
		return Bounds{}
	}
	if len(f.children) == 0 {
		return Bounds{Height: f.minHeight}
	}

	var size Bounds
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

func (f *flex) measureRow(budget Bounds) Bounds {
	count := len(f.children)
	if count == 0 {
		return Bounds{}
	}

	if budget.Width > 0 {
		slotWidths := partitionBudgetWeighted(budget.Width, flexGap, f.children)
		maxHeight := 1
		for i, child := range f.children {
			size := measure(child, Bounds{
				Width:  slotWidths[i],
				Height: budget.Height,
			})
			if size.Height > maxHeight {
				maxHeight = size.Height
			}
		}
		return Bounds{Width: budget.Width, Height: maxHeight}
	}

	totalWidth := 0
	maxHeight := 1
	for i, child := range f.children {
		if i > 0 {
			totalWidth += flexGap
		}
		size := measure(child, Bounds{Height: budget.Height})
		totalWidth += size.Width
		if size.Height > maxHeight {
			maxHeight = size.Height
		}
	}

	return Bounds{Width: totalWidth, Height: maxHeight}
}

func (f *flex) measureColumn(budget Bounds) Bounds {
	count := len(f.children)
	if count == 0 {
		return Bounds{}
	}

	if budget.Height > 0 {
		slotHeights := partitionBudgetWeighted(budget.Height, flexGap, f.children)
		maxWidth := 1
		for i, child := range f.children {
			size := measure(child, Bounds{
				Width:  budget.Width,
				Height: slotHeights[i],
			})
			if size.Width > maxWidth {
				maxWidth = size.Width
			}
		}
		return Bounds{Width: maxWidth, Height: budget.Height}
	}

	maxWidth := 1
	totalHeight := 0
	for i, child := range f.children {
		if i > 0 {
			totalHeight += flexGap
		}
		size := measure(child, Bounds{Width: budget.Width})
		if size.Width > maxWidth {
			maxWidth = size.Width
		}
		totalHeight += size.Height
	}

	return Bounds{Width: maxWidth, Height: totalHeight}
}

// Render assigns bounds to each child and renders them.
func (f *flex) Render(target io.Writer, bounds Bounds) {
	if f == nil || len(f.children) == 0 {
		return
	}

	size := measure(f, Bounds{Width: bounds.Width, Height: bounds.Height})
	if bounds.Width <= 0 {
		bounds.Width = size.Width
	}
	if bounds.Height <= 0 {
		bounds.Height = size.Height
	}
	if bounds.Height < f.minHeight {
		bounds.Height = f.minHeight
	}
	if bounds.Width <= 0 || bounds.Height <= 0 {
		return
	}

	switch f.direction {
	case DirectionColumn:
		f.renderColumn(target, bounds)
	default:
		f.renderRow(target, bounds)
	}
}

func (f *flex) renderRow(target io.Writer, bounds Bounds) {
	if bounds.Width > 0 {
		slotWidths := partitionBudgetWeighted(bounds.Width, flexGap, f.children)
		x := bounds.X
		for i, child := range f.children {
			slotWidth := slotWidths[i]
			if slotWidth > 0 {
				size := measure(child, Bounds{
					Width:  slotWidth,
					Height: bounds.Height,
				})
				childHeight := size.Height
				if childHeight > bounds.Height {
					childHeight = bounds.Height
				}
				if childHeight < 1 {
					childHeight = 1
				}
				child.Render(target, Bounds{
					X:      x,
					Y:      bounds.Y,
					Width:  slotWidth,
					Height: childHeight,
				})
			}
			x += slotWidth + flexGap
		}
		return
	}

	x := bounds.X

	for i, child := range f.children {
		if i > 0 {
			x += flexGap
		}

		size := measure(child, Bounds{Height: bounds.Height})

		child.Render(target, Bounds{
			X:      x,
			Y:      bounds.Y,
			Width:  size.Width,
			Height: size.Height,
		})

		x += size.Width
	}
}

func (f *flex) renderColumn(target io.Writer, bounds Bounds) {
	if bounds.Height > 0 {
		slotHeights := partitionBudgetWeighted(bounds.Height, flexGap, f.children)
		y := bounds.Y
		for i, child := range f.children {
			slotHeight := slotHeights[i]
			if slotHeight > 0 {
				size := measure(child, Bounds{
					Width:  bounds.Width,
					Height: slotHeight,
				})
				childWidth := size.Width
				if bounds.Width > 0 && childWidth > bounds.Width {
					childWidth = bounds.Width
				}
				if childWidth < 1 {
					childWidth = 1
				}
				child.Render(target, Bounds{
					X:      bounds.X,
					Y:      y,
					Width:  childWidth,
					Height: slotHeight,
				})
			}
			y += slotHeight + flexGap
		}
		return
	}

	y := bounds.Y

	for i, child := range f.children {
		if i > 0 {
			y += flexGap
		}

		size := measure(child, Bounds{Width: bounds.Width})

		child.Render(target, Bounds{
			X:      bounds.X,
			Y:      y,
			Width:  size.Width,
			Height: size.Height,
		})

		y += size.Height
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
