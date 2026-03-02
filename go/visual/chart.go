package visual

import (
	"fmt"
	"io"
)

type chart struct {
	xAxis []int
	yAxis []int
	mode  ChartMode
}

var _ Artefact = (*chart)(nil)

const chartTrackDot = "●"
const chartDataDot = "●"

type ChartOptions struct {
	XAxis []int
	YAxis []int
	Mode  ChartMode
}

type ChartMode int

const (
	ChartWindowed ChartMode = iota
	ChartFitToViewport
)

// Chart creates a simple chart.
// Each x/y point is rendered as a gray track with a white data marker.
func Chart(options ChartOptions) Artefact {
	xCopy := make([]int, len(options.XAxis))
	yCopy := make([]int, len(options.YAxis))
	copy(xCopy, options.XAxis)
	copy(yCopy, options.YAxis)

	return &chart{
		xAxis: xCopy,
		yAxis: yCopy,
		mode:  options.Mode,
	}
}

func (c *chart) Weight() int {
	return 1
}

func (c *chart) Measure(budget Bounds) Bounds {
	width := max(len(c.xAxis), 1)
	height := 2

	if budget.Width > 0 {
		width = budget.Width
	}
	if budget.Height > 0 {
		height = budget.Height
	}
	if width < 1 {
		width = 1
	}
	if height < 2 {
		height = 2
	}

	return Bounds{Width: width, Height: height}
}

// Render draws the chart inside the provided bounds.
func (c *chart) Render(target io.Writer, bounds Bounds) {
	if c == nil || len(c.xAxis) == 0 || len(c.yAxis) == 0 || bounds.Width <= 0 || bounds.Height <= 0 {
		return
	}
	nPoints := min(len(c.yAxis), len(c.xAxis))
	if nPoints == 0 || bounds.Height < 2 {
		return
	}

	yData := c.yAxis[:nPoints]

	// Use all available vertical budget minus one row for the baseline axis.
	plotHeight := bounds.Height - 1
	if plotHeight < 1 {
		plotHeight = 1
	}
	plotY := bounds.Y + (bounds.Height - 1 - plotHeight)
	const barWidth = 1
	maxBars := max(bounds.Width/barWidth, 1)

	values, barOffset := c.valuesForBars(yData, nPoints, maxBars)
	if len(values) == 0 {
		return
	}

	scaleData := yData
	if c.mode == ChartWindowed {
		scaleData = values
	}

	minY, maxY := scaleData[0], scaleData[0]
	for _, v := range scaleData {
		if v < minY {
			minY = v
		}
		if v > maxY {
			maxY = v
		}
	}

	bars := make([]Artefact, maxBars)
	for i := range maxBars {
		bar := &chartBar{min: minY, max: maxY}
		bars[i] = bar
	}

	for i, value := range values {
		barPos := barOffset + i
		if barPos < 0 || barPos >= maxBars {
			continue
		}
		bar := bars[barPos].(*chartBar)
		bar.hasValue = true
		bar.value = value
	}

	for i, bar := range bars {
		bar.Render(target, Bounds{
			X:      bounds.X + i,
			Y:      plotY,
			Width:  barWidth,
			Height: plotHeight,
		})
	}

	Line().Render(target, Bounds{
		X:      bounds.X,
		Y:      bounds.Y + bounds.Height - 1,
		Width:  bounds.Width,
		Height: 1,
	})
}

func (c *chart) valuesForBars(yData []int, nPoints, maxBars int) ([]int, int) {
	if nPoints <= 0 || maxBars <= 0 {
		return nil, 0
	}

	if c.mode == ChartFitToViewport {
		indices := sampleIndices(nPoints, maxBars)
		values := make([]int, len(indices))
		for i, idx := range indices {
			values[i] = yData[idx]
		}
		return values, 0
	}

	count := min(nPoints, maxBars)
	start := nPoints - count
	values := make([]int, count)
	copy(values, yData[start:])
	return values, maxBars - count
}

type chartBar struct {
	value    int
	min      int
	max      int
	hasValue bool
}

var _ Artefact = (*chartBar)(nil)

func (b *chartBar) Weight() int {
	return 1
}

func (b *chartBar) Measure(budget Bounds) Bounds {
	width := 1
	height := 1
	if budget.Width > 0 {
		width = budget.Width
	}
	if budget.Height > 0 {
		height = budget.Height
	}
	return Bounds{Width: width, Height: height}
}

func (b *chartBar) Render(target io.Writer, bounds Bounds) {
	if b == nil || bounds.Width <= 0 || bounds.Height <= 0 {
		return
	}

	x := bounds.X + bounds.Width/2

	for y := bounds.Y; y < bounds.Y+bounds.Height; y++ {
		fmt.Fprintf(target, "\033[90m\033[%d;%dH%s\033[0m", y, x, chartTrackDot)
	}

	if !b.hasValue {
		return
	}

	y := valueToY(b.value, b.min, b.max, bounds)
	for yy := y; yy < bounds.Y+bounds.Height; yy++ {
		fmt.Fprintf(target, "\033[97m\033[%d;%dH%s\033[0m", yy, x, chartDataDot)
	}
}

func valueToY(value, minValue, maxValue int, bounds Bounds) int {
	height := bounds.Height
	if height <= 1 {
		return bounds.Y
	}
	if maxValue <= minValue {
		return bounds.Y + (height-1)/2
	}
	normalized := (value - minValue) * (height - 1) / (maxValue - minValue)
	if normalized > height-1 {
		normalized = height - 1
	}
	if normalized < 0 {
		normalized = 0
	}
	offset := (height - 1) - normalized
	return bounds.Y + offset
}

func sampleIndices(total, limit int) []int {
	if total <= 0 || limit <= 0 {
		return nil
	}
	if total <= limit {
		indices := make([]int, total)
		for i := range total {
			indices[i] = i
		}
		return indices
	}

	indices := make([]int, limit)
	last := total - 1
	for i := range limit {
		indices[i] = (i * last) / (limit - 1)
	}
	return indices
}
