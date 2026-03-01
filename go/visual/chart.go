package visual

import (
	"fmt"
	"io"
)

type chart struct {
	xAxis []int
	yAxis []int
}

var _ Artefact = (*chart)(nil)

const chartDot = "·"

// Chart creates a simple chart.
// Each x/y point is rendered as a gray track with a white data marker.
func Chart(xAxis, yAxis []int) Artefact {
	xCopy := make([]int, len(xAxis))
	yCopy := make([]int, len(yAxis))
	copy(xCopy, xAxis)
	copy(yCopy, yAxis)
	return &chart{xAxis: xCopy, yAxis: yCopy}
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

	xData := c.xAxis[:nPoints]
	yData := c.yAxis[:nPoints]

	plotHeight := max((bounds.Height-1)/2, 40)
	if plotHeight > bounds.Height-1 {
		plotHeight = bounds.Height - 1
	}
	plotY := bounds.Y + (bounds.Height - 1 - plotHeight)
	const barWidth = 1
	maxBars := max(bounds.Width/barWidth, 1)

	// Use the latest samples that fit in the current number of chart columns.
	if nPoints > maxBars {
		start := nPoints - maxBars
		xData = xData[start:]
		yData = yData[start:]
		nPoints = maxBars
	}

	sampledIndices := sampleIndices(nPoints, maxBars)
	minY, maxY := yData[sampledIndices[0]], yData[sampledIndices[0]]
	for _, idx := range sampledIndices {
		v := yData[idx]
		if v < minY {
			minY = v
		}
		if v > maxY {
			maxY = v
		}
	}

	bars := make([]Artefact, maxBars)
	for i := range maxBars {
		bar := &memoryBar{min: minY, max: maxY}
		if nPoints <= maxBars {
			leftPadding := maxBars - nPoints
			if i >= leftPadding {
				bar.hasValue = true
				bar.value = yData[i-leftPadding]
			}
		} else {
			bar.hasValue = true
			bar.value = yData[sampledIndices[i]]
		}
		bars[i] = bar
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

type memoryBar struct {
	value    int
	min      int
	max      int
	hasValue bool
}

var _ Artefact = (*memoryBar)(nil)

func (b *memoryBar) Render(target io.Writer, bounds Bounds) {
	if b == nil || bounds.Width <= 0 || bounds.Height <= 0 {
		return
	}

	x := bounds.X + bounds.Width/2

	for y := bounds.Y; y < bounds.Y+bounds.Height; y++ {
		fmt.Fprintf(target, "\033[90m\033[%d;%dH%s\033[0m", y, x, chartDot)
	}

	if !b.hasValue {
		return
	}

	y := valueToY(b.value, b.min, b.max, bounds)
	for yy := y; yy < bounds.Y+bounds.Height; yy++ {
		fmt.Fprintf(target, "\033[97m\033[%d;%dH%s\033[0m", yy, x, chartDot)
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
