package visual

import (
	"fmt"
	"io"
)

type chart struct {
	layoutState
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

func (c *chart) Measure(budget Area) Area {
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

	return Area{Width: width, Height: height}
}

// Render draws the chart inside the measured area.
func (c *chart) Render(target io.Writer) {
	frame := layoutOf(c)
	if c == nil || len(c.xAxis) == 0 || len(c.yAxis) == 0 || frame.Width <= 0 || frame.Height <= 0 {
		return
	}
	nPoints := min(len(c.yAxis), len(c.xAxis))
	if nPoints == 0 || frame.Height < 2 {
		return
	}

	yData := c.yAxis[:nPoints]

	// Use all available vertical budget minus one row for the baseline axis.
	plotHeight := frame.Height - 1
	if plotHeight < 1 {
		plotHeight = 1
	}
	plotY := frame.Y + (frame.Height - 1 - plotHeight)
	const barWidth = 1
	maxBars := max(frame.Width/barWidth, 1)

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
		layoutAt(bar, Point{X: frame.X + i, Y: plotY}, Area{Width: barWidth, Height: plotHeight})
		bar.Render(target)
	}

	RenderAt(
		target,
		Line(),
		Point{X: frame.X, Y: frame.Y + frame.Height - 1},
		Area{Width: frame.Width, Height: 1},
	)
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
	layoutState
	value    int
	min      int
	max      int
	hasValue bool
}

var _ Artefact = (*chartBar)(nil)

func (b *chartBar) Weight() int {
	return 1
}

func (b *chartBar) Measure(budget Area) Area {
	width := 1
	height := 1
	if budget.Width > 0 {
		width = budget.Width
	}
	if budget.Height > 0 {
		height = budget.Height
	}
	return Area{Width: width, Height: height}
}

func (b *chartBar) Render(target io.Writer) {
	frame := layoutOf(b)
	if b == nil || frame.Width <= 0 || frame.Height <= 0 {
		return
	}

	x := frame.X + frame.Width/2

	for y := frame.Y; y < frame.Y+frame.Height; y++ {
		fmt.Fprintf(target, "\033[90m\033[%d;%dH%s\033[0m", y, x, chartTrackDot)
	}

	if !b.hasValue {
		return
	}

	y := valueToY(b.value, b.min, b.max, frame)
	for yy := y; yy < frame.Y+frame.Height; yy++ {
		fmt.Fprintf(target, "\033[97m\033[%d;%dH%s\033[0m", yy, x, chartDataDot)
	}
}

func valueToY(value, minValue, maxValue int, plot frame) int {
	height := plot.Height
	if height <= 1 {
		return plot.Y
	}
	if maxValue <= minValue {
		return plot.Y + (height-1)/2
	}
	normalized := (value - minValue) * (height - 1) / (maxValue - minValue)
	if normalized > height-1 {
		normalized = height - 1
	}
	if normalized < 0 {
		normalized = 0
	}
	offset := (height - 1) - normalized
	return plot.Y + offset
}

func sampleIndices(total, limit int) []int {
	if total <= 0 || limit <= 0 {
		return nil
	}
	if limit == 1 {
		return []int{0}
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
