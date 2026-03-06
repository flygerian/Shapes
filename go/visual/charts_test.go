package visual

import (
	"bytes"
	"strings"
	"testing"
)

func TestSampleIndicesSingleLimit(t *testing.T) {
	indices := sampleIndices(10, 1)
	if len(indices) != 1 {
		t.Fatalf("expected 1 index, got %d", len(indices))
	}
	if indices[0] != 0 {
		t.Fatalf("expected first index 0, got %d", indices[0])
	}
}

func TestLineChartRenderConnectsPoints(t *testing.T) {
	var out bytes.Buffer
	chart := LineChart(LineChartOptions{
		XAxis: []int{0, 1, 2},
		Series: []LineSeries{
			{YAxis: []int{0, 10, 0}},
		},
		Mode: ChartWindowed,
	})

	chart.Render(&out, Bounds{X: 1, Y: 1, Width: 3, Height: 3})
	rendered := out.String()

	expected := []string{
		"\033[97m\033[2;1H●\033[0m",
		"\033[97m\033[1;2H●\033[0m",
		"\033[97m\033[2;3H●\033[0m",
		"\033[3;1H─",
	}
	for _, token := range expected {
		if !strings.Contains(rendered, token) {
			t.Fatalf("expected rendered output to contain %q, got %q", token, rendered)
		}
	}
}

func TestLineChartFitToViewportWidthOneRendersSinglePoint(t *testing.T) {
	var out bytes.Buffer
	chart := LineChart(LineChartOptions{
		XAxis: []int{0, 1, 2, 3},
		Series: []LineSeries{
			{YAxis: []int{2, 4, 6, 8}},
		},
		Mode: ChartFitToViewport,
	})

	chart.Render(&out, Bounds{X: 1, Y: 1, Width: 1, Height: 2})
	rendered := out.String()

	if strings.Count(rendered, "●") != 1 {
		t.Fatalf("expected exactly one data point, got %q", rendered)
	}
	if !strings.Contains(rendered, "\033[2;1H─") {
		t.Fatalf("expected baseline line in output, got %q", rendered)
	}
}

func TestLineChartRenderMultipleSeriesColors(t *testing.T) {
	var out bytes.Buffer
	chart := LineChart(LineChartOptions{
		XAxis: []int{0, 1, 2},
		Series: []LineSeries{
			{YAxis: []int{0, 5, 0}, Color: 96},
			{YAxis: []int{5, 0, 5}, Color: 92},
		},
		Mode: ChartWindowed,
	})

	chart.Render(&out, Bounds{X: 1, Y: 1, Width: 3, Height: 3})
	rendered := out.String()

	if !strings.Contains(rendered, "\033[96m") {
		t.Fatalf("expected first series color to be rendered, got %q", rendered)
	}
	if !strings.Contains(rendered, "\033[92m") {
		t.Fatalf("expected second series color to be rendered, got %q", rendered)
	}
}

func TestBooleanChartRenderColorsAndWrap(t *testing.T) {
	var out bytes.Buffer
	chart := BooleanChart(BooleanChartOptions{
		Values: []bool{true, false, true, false, true},
	})

	chart.Render(&out, Bounds{X: 1, Y: 1, Width: 3, Height: 2})
	rendered := out.String()

	expected := []string{
		"\033[97m\033[1;1H●\033[0m",
		"\033[90m\033[1;2H●\033[0m",
		"\033[97m\033[1;3H●\033[0m",
		"\033[90m\033[2;1H●\033[0m",
		"\033[97m\033[2;2H●\033[0m",
	}
	for _, token := range expected {
		if !strings.Contains(rendered, token) {
			t.Fatalf("expected rendered output to contain %q, got %q", token, rendered)
		}
	}
}

func TestBooleanChartRenderWindowsToLatestValues(t *testing.T) {
	var out bytes.Buffer
	chart := BooleanChart(BooleanChartOptions{
		Values: []bool{false, false, true, true},
	})

	chart.Render(&out, Bounds{X: 1, Y: 1, Width: 2, Height: 1})
	rendered := out.String()

	if strings.Contains(rendered, "\033[90m\033[1;1H●\033[0m") {
		t.Fatalf("expected oldest values to be dropped, got %q", rendered)
	}
	if strings.Count(rendered, "\033[97m") != 2 {
		t.Fatalf("expected two latest true values, got %q", rendered)
	}
}

func TestBooleanChartMeasureRowsOption(t *testing.T) {
	chart := BooleanChart(BooleanChartOptions{
		Values: []bool{true, false, true, false, true},
		Rows:   2,
	})

	size := chart.Measure(Bounds{})
	if size.Height != 2 {
		t.Fatalf("expected height 2, got %d", size.Height)
	}
	if size.Width != 3 {
		t.Fatalf("expected width 3 (ceil(5/2)), got %d", size.Width)
	}
}

func TestBooleanChartRenderRowsOption(t *testing.T) {
	var out bytes.Buffer
	chart := BooleanChart(BooleanChartOptions{
		Values: []bool{true, false, true, false, true},
		Rows:   2,
	})

	chart.Render(&out, Bounds{X: 1, Y: 1, Width: 10, Height: 10})
	rendered := out.String()

	expected := []string{
		"\033[97m\033[1;1H●\033[0m",
		"\033[90m\033[1;2H●\033[0m",
		"\033[97m\033[1;3H●\033[0m",
		"\033[90m\033[2;1H●\033[0m",
		"\033[97m\033[2;2H●\033[0m",
	}
	for _, token := range expected {
		if !strings.Contains(rendered, token) {
			t.Fatalf("expected rendered output to contain %q, got %q", token, rendered)
		}
	}
}

func TestBooleanChartRenderRowsOptionRespectsBounds(t *testing.T) {
	var out bytes.Buffer
	chart := BooleanChart(BooleanChartOptions{
		Values: []bool{true, false, true, false, true, false},
		Rows:   4,
	})

	chart.Render(&out, Bounds{X: 1, Y: 1, Width: 2, Height: 2})
	rendered := out.String()

	if strings.Contains(rendered, "\033[3;") || strings.Contains(rendered, "\033[4;") {
		t.Fatalf("expected output to be clipped to 2 rows, got %q", rendered)
	}
}
