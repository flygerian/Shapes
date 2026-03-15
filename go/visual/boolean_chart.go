package visual

import (
	"fmt"
	"io"
)

const booleanChartDot = "●"

type booleanChart struct {
	layoutState
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

	rows := max(options.Rows, 0)

	return &booleanChart{values: valuesCopy, rows: rows}
}

func (c *booleanChart) Weight() int {
	return 1
}

func (c *booleanChart) Measure(budget Area) Area {
	if c.rows > 0 {
		cols := max(1, divideRoundUp(max(len(c.values), 1), c.rows))
		height := c.rows
		if budget.Height > 0 && height > budget.Height {
			height = budget.Height
		}
		if height < 1 {
			height = 1
		}
		if height != c.rows {
			cols = max(1, divideRoundUp(max(len(c.values), 1), height))
		}
		if budget.Width > 0 && cols > budget.Width {
			cols = budget.Width
		}
		return Area{Width: cols, Height: height}
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
	return Area{Width: width, Height: height}
}

func (c *booleanChart) Render(target io.Writer) {
	frame := layoutOf(c)
	if c == nil || len(c.values) == 0 || frame.Width <= 0 || frame.Height <= 0 {
		return
	}

	gridWidth := frame.Width
	gridHeight := frame.Height
	if c.rows > 0 {
		gridHeight = min(c.rows, frame.Height)
		gridWidth = min(max(1, divideRoundUp(len(c.values), c.rows)), frame.Width)
	}

	capacity := gridWidth * gridHeight
	if capacity <= 0 {
		return
	}

	count := min(len(c.values), capacity)
	start := len(c.values) - count
	for i := range count {
		value := c.values[start+i]
		x := frame.X + (i % gridWidth)
		y := frame.Y + (i / gridWidth)
		if y >= frame.Y+gridHeight {
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
