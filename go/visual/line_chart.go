package visual

import (
	"fmt"
	"io"
)

const lineChartDot = "●"

type chartPoint struct {
	x int
	y int
}

type LineSeries struct {
	YAxis []int
	Color int
}

type lineChartSeries struct {
	values []int
	offset int
}

type lineChart struct {
	layoutState
	xAxis  []int
	series []LineSeries
	mode   ChartMode
}

var _ Artefact = (*lineChart)(nil)

type LineChartOptions struct {
	XAxis  []int
	Series []LineSeries
	Mode   ChartMode
}

// LineChart creates a connected line chart artefact.
func LineChart(options LineChartOptions) Artefact {
	xCopy := make([]int, len(options.XAxis))
	copy(xCopy, options.XAxis)

	seriesCopy := make([]LineSeries, len(options.Series))
	for i, s := range options.Series {
		yCopy := make([]int, len(s.YAxis))
		copy(yCopy, s.YAxis)
		seriesCopy[i] = LineSeries{YAxis: yCopy, Color: s.Color}
	}

	return &lineChart{
		xAxis:  xCopy,
		series: seriesCopy,
		mode:   options.Mode,
	}
}

func (c *lineChart) Weight() int {
	return 1
}

func (c *lineChart) Measure(budget Area) Area {
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

func (c *lineChart) Render(target io.Writer) {
	bounds := layoutOf(c)
	if c == nil || len(c.xAxis) == 0 || len(c.series) == 0 || bounds.Width <= 0 || bounds.Height <= 0 {
		return
	}
	if bounds.Height < 2 {
		return
	}

	plotHeight := bounds.Height - 1
	if plotHeight < 1 {
		plotHeight = 1
	}
	plotY := bounds.Y + (bounds.Height - 1 - plotHeight)
	plot := frame{
		Point: Point{X: bounds.X, Y: plotY},
		Area:  Area{Width: bounds.Width, Height: plotHeight},
	}

	renderSeries := make([]lineChartSeries, len(c.series))
	hasScale := false
	var globalMinY, globalMaxY int

	for i, s := range c.series {
		nPoints := min(len(s.YAxis), len(c.xAxis))
		if nPoints == 0 {
			continue
		}

		yData := s.YAxis[:nPoints]
		values, xOffset := c.valuesForColumns(yData, nPoints, bounds.Width)
		if len(values) == 0 {
			continue
		}
		renderSeries[i] = lineChartSeries{values: values, offset: xOffset}

		scaleData := values
		if c.mode == ChartFitToViewport {
			scaleData = yData
		}
		for _, v := range scaleData {
			if !hasScale {
				globalMinY, globalMaxY = v, v
				hasScale = true
				continue
			}
			if v < globalMinY {
				globalMinY = v
			}
			if v > globalMaxY {
				globalMaxY = v
			}
		}
	}

	if !hasScale {
		return
	}

	for i, s := range c.series {
		seriesData := renderSeries[i]
		if len(seriesData.values) == 0 {
			continue
		}

		color := s.Color
		if color <= 0 {
			color = 97
		}

		points := make([]chartPoint, len(seriesData.values))
		for j, v := range seriesData.values {
			points[j] = chartPoint{
				x: bounds.X + seriesData.offset + j,
				y: valueToY(v, globalMinY, globalMaxY, plot),
			}
		}

		drawn := make(map[chartPoint]struct{})
		for j := 1; j < len(points); j++ {
			drawSegment(points[j-1], points[j], drawn, color, target)
		}
		for _, p := range points {
			drawPoint(p, drawn, color, target)
		}
	}

	RenderAt(
		target,
		Line(),
		Point{X: bounds.X, Y: bounds.Y + bounds.Height - 1},
		Area{Width: bounds.Width, Height: 1},
	)
}

func (c *lineChart) valuesForColumns(yData []int, nPoints, width int) ([]int, int) {
	if nPoints <= 0 || width <= 0 {
		return nil, 0
	}

	if c.mode == ChartFitToViewport {
		indices := sampleIndices(nPoints, width)
		values := make([]int, len(indices))
		for i, idx := range indices {
			values[i] = yData[idx]
		}
		return values, 0
	}

	count := min(nPoints, width)
	start := nPoints - count
	values := make([]int, count)
	copy(values, yData[start:])
	return values, width - count
}

func drawSegment(from, to chartPoint, drawn map[chartPoint]struct{}, color int, target io.Writer) {
	x0, y0 := from.x, from.y
	x1, y1 := to.x, to.y

	dx := x1 - x0
	if dx < 0 {
		dx = -dx
	}
	sx := -1
	if x0 < x1 {
		sx = 1
	}

	dy := y1 - y0
	if dy < 0 {
		dy = -dy
	}
	sy := -1
	if y0 < y1 {
		sy = 1
	}

	err := dx - dy

	for {
		p := chartPoint{x: x0, y: y0}
		if _, exists := drawn[p]; !exists {
			fmt.Fprintf(target, "\033[%dm\033[%d;%dH%s\033[0m", color, y0, x0, lineChartDot)
			drawn[p] = struct{}{}
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x0 += sx
		}
		if e2 < dx {
			err += dx
			y0 += sy
		}
	}
}

func drawPoint(p chartPoint, drawn map[chartPoint]struct{}, color int, target io.Writer) {
	if _, exists := drawn[p]; exists {
		return
	}
	fmt.Fprintf(target, "\033[%dm\033[%d;%dH%s\033[0m", color, p.y, p.x, lineChartDot)
	drawn[p] = struct{}{}
}
