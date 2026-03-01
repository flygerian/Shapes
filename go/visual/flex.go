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
}

var _ Artefact = (*flex)(nil)

// Flex creates a flex container that lays out children in a row or column.
func Flex(direction Direction, children ...Artefact) Artefact {
	return &flex{
		direction: direction,
		children:  children,
	}
}

// Render assigns bounds to each child and renders them.
func (f *flex) Render(target io.Writer, bounds Bounds) {
	if f == nil || len(f.children) == 0 || bounds.Width <= 0 || bounds.Height <= 0 {
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
	count := len(f.children)
	totalGap := flexGap * (count - 1)
	available := bounds.Width - totalGap
	if available <= 0 {
		return
	}

	base := available / count
	rem := available % count
	x := bounds.X

	for i, child := range f.children {
		w := base
		if i < rem {
			w++
		}
		if w > 0 {
			child.Render(target, Bounds{
				X:      x,
				Y:      bounds.Y,
				Width:  w,
				Height: bounds.Height,
			})
		}
		x += w + flexGap
	}
}

func (f *flex) renderColumn(target io.Writer, bounds Bounds) {
	count := len(f.children)
	totalGap := flexGap * (count - 1)
	available := bounds.Height - totalGap
	if available <= 0 {
		return
	}

	base := available / count
	rem := available % count
	y := bounds.Y

	for i, child := range f.children {
		h := base
		if i < rem {
			h++
		}
		if h > 0 {
			child.Render(target, Bounds{
				X:      bounds.X,
				Y:      y,
				Width:  bounds.Width,
				Height: h,
			})
		}
		y += h + flexGap
	}
}
