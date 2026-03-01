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
	nPoints := len(c.xAxis)
	if len(c.yAxis) < nPoints {
		nPoints = len(c.yAxis)
	}
	if nPoints == 0 || bounds.Height < 2 {
		return
	}

	plotHeight := bounds.Height - 1
	const barWidth = 1
	maxBars := (bounds.Width + flexGap) / (barWidth + flexGap)
	if maxBars < 1 {
		maxBars = 1
	}

	sampledIndices := sampleIndices(nPoints, maxBars)
	minY, maxY := c.yAxis[sampledIndices[0]], c.yAxis[sampledIndices[0]]
	for _, idx := range sampledIndices {
		v := c.yAxis[idx]
		if v < minY {
			minY = v
		}
		if v > maxY {
			maxY = v
		}
	}

	bars := make([]Artefact, maxBars)
	for i := 0; i < maxBars; i++ {
		bar := &memoryBar{min: minY, max: maxY}
		if nPoints <= maxBars {
			leftPadding := maxBars - nPoints
			if i >= leftPadding {
				bar.hasValue = true
				bar.value = c.yAxis[i-leftPadding]
			}
		} else {
			bar.hasValue = true
			bar.value = c.yAxis[sampledIndices[i]]
		}
		bars[i] = bar
	}

	Flex(DirectionRow, bars...).Render(target, Bounds{
		X:      bounds.X,
		Y:      bounds.Y,
		Width:  bounds.Width,
		Height: plotHeight,
	})

	Line().Render(target, Bounds{
		X:      bounds.X,
		Y:      bounds.Y + bounds.Height - 1,
		Width:  bounds.Width,
		Height: 1,
	})
	label := fmt.Sprintf("x:%d..%d y:blocks (%d..%d)", c.xAxis[sampledIndices[0]], c.xAxis[sampledIndices[len(sampledIndices)-1]], minY, maxY)
	Text(label).Render(target, Bounds{
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
		fmt.Fprintf(target, "\033[90m\033[%d;%dH●\033[0m", y, x)
	}

	if !b.hasValue {
		return
	}

	y := valueToY(b.value, b.min, b.max, bounds)
	for yy := y; yy < bounds.Y+bounds.Height; yy++ {
		fmt.Fprintf(target, "\033[97m\033[%d;%dH●\033[0m", yy, x)
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
		for i := 0; i < total; i++ {
			indices[i] = i
		}
		return indices
	}

	indices := make([]int, limit)
	last := total - 1
	for i := 0; i < limit; i++ {
		indices[i] = (i * last) / (limit - 1)
	}
	return indices
}
