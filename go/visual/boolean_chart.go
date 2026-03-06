package visual

import (
	"fmt"
	"io"
)

const booleanChartDot = "●"

type booleanChart struct {
	values []bool
	rows   int
}

var _ Artefact = (*booleanChart)(nil)

type BooleanChartOptions struct {
	Values []bool
	Rows   int
}

// BooleanChart renders booleans as colored circle glyphs.
func BooleanChart(options BooleanChartOptions) Artefact {
	valuesCopy := make([]bool, len(options.Values))
	copy(valuesCopy, options.Values)

	rows := 0
	if options.Rows > 0 {
		rows = options.Rows
	}

	return &booleanChart{values: valuesCopy, rows: rows}
}

func (c *booleanChart) Weight() int {
	return 1
}

func (c *booleanChart) Measure(budget Bounds) Bounds {
	if c.rows > 0 {
		cols := max(1, divideRoundUp(max(len(c.values), 1), c.rows))
		return Bounds{Width: cols, Height: c.rows}
	}

	width := max(len(c.values), 1)
	height := 1
	if budget.Width > 0 {
		width = budget.Width
	}
	if budget.Height > 0 {
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

func (c *booleanChart) Render(target io.Writer, bounds Bounds) {
	if c == nil || len(c.values) == 0 || bounds.Width <= 0 || bounds.Height <= 0 {
		return
	}

	gridWidth := bounds.Width
	gridHeight := bounds.Height
	if c.rows > 0 {
		gridHeight = min(c.rows, bounds.Height)
		gridWidth = min(max(1, divideRoundUp(len(c.values), c.rows)), bounds.Width)
	}

	capacity := gridWidth * gridHeight
	if capacity <= 0 {
		return
	}

	count := min(len(c.values), capacity)
	start := len(c.values) - count
	for i := range count {
		value := c.values[start+i]
		x := bounds.X + (i % gridWidth)
		y := bounds.Y + (i / gridWidth)
		if y >= bounds.Y+gridHeight {
			break
		}

		color := 90
		if value {
			color = 97
		}
		fmt.Fprintf(target, "\033[%dm\033[%d;%dH%s\033[0m", color, y, x, booleanChartDot)
	}
}

func divideRoundUp(a int, b int) int {
	if b <= 0 {
		return 0
	}
	return (a + b - 1) / b
}
