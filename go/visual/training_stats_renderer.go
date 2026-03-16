package visual

import (
	"bytes"
	"fmt"
	"os"
	"strconv"
	"syscall"
	"time"
	"unsafe"

	"github.com/flygerian/shapes"
)

type TrainingStatsRenderer struct {
	ctx shapes.MainContext
}

func (r *TrainingStatsRenderer) SetTrainingContext(ctx shapes.MainContext) {
	r.ctx = ctx

	go r.launchTrainingDashboard()
}

func (r *TrainingStatsRenderer) renderTrainingHeader(header string) Artefact {
	return Flex(FlexOptions{
		Direction: DirectionColumn,
		Children: []Artefact{
			Text("Training"),
			Text(header),
		},
	})
}

func (r *TrainingStatsRenderer) renderMemoryAndLossCharts(stats *shapes.TrainingStats) Artefact {
	lossCharts := []Artefact{
		metricChart("Loss", stats.LossHistoryX, stats.LossHistory),
		metricChart("Val Loss", stats.ValidationLossHistoryX, stats.ValidationLossHistory),
		metricChart("Accuracy", stats.AccuracyHistoryX, stats.AccuracyHistory),
	}
	if len(stats.StepLossHistoryX) > 0 {
		lossCharts = append(lossCharts, metricChart("Step Loss", stats.StepLossHistoryX, stats.StepLossHistory))
	}

	return Flex(FlexOptions{
		Direction: DirectionColumn,
		Children: []Artefact{
			Weighted(WeightOptions{
				Weight: 2,
				Child:  metricChartWithMode("Memory", stats.MemorySampleHistoryX, stats.UsedBlocksHistory, ChartWindowed),
			}),
			Weighted(WeightOptions{
				Weight: 3,
				Child: Flex(FlexOptions{
					Direction: DirectionRow,
					Children:  lossCharts,
				}),
			}),
		},
	})
}

func metricChart(label string, xAxis, yAxis []int) Artefact {
	return metricChartWithMode(label, xAxis, yAxis, ChartFitToViewport)
}

func metricChartWithMode(label string, xAxis, yAxis []int, mode ChartMode) Artefact {
	return Flex(FlexOptions{
		Direction: DirectionColumn,
		Children: []Artefact{
			Chart(ChartOptions{
				XAxis: xAxis,
				YAxis: yAxis,
				Mode:  mode,
			}),
			Box(BoxOptions{
				Child: Text(label),
			}),
		},
	})
}

func (r *TrainingStatsRenderer) renderSampledTensors(stats *shapes.TrainingStats) Artefact {
	var charts []Artefact

	for key, sampleTensor := range stats.SampleTensors {
		if sampleTensor.Dtype() != shapes.DtypeBool {
			continue
		}

		chart := Flex(
			FlexOptions{
				Direction: DirectionColumn,
				Children: []Artefact{
					BooleanChart(BooleanChartOptions{
						Rows:   int(sampleTensor.Shape()[0]),
						Values: sampleTensor.Values().([]bool),
					}),
					Box(
						BoxOptions{
							Child: Text(key),
						},
					),
				},
			},
		)
		charts = append(charts, chart)
	}

	return Flex(
		FlexOptions{
			Children:  charts,
			Direction: DirectionColumn,
		},
	)
}

func (r *TrainingStatsRenderer) launchTrainingDashboard() {
	ticker := time.NewTicker(time.Second / 60)
	defer ticker.Stop()
	fmt.Fprint(os.Stdout, "\033[?25l\033[2J\033[H")
	defer fmt.Fprint(os.Stdout, "\033[?25h")

	lastVersion := -1

	stats := r.ctx.TrainingStats()

	render := func(force bool) {
		if len(stats.MemorySampleHistoryX) == 0 {
			return
		}
		if !force && stats.Version == lastVersion {
			return
		}
		lastVersion = stats.Version

		header := trainingHeader(stats)
		dashboard := Flex(FlexOptions{
			MinHeight: 60,
			Direction: DirectionColumn,
			Children: []Artefact{
				Box(BoxOptions{
					Child: r.renderTrainingHeader(header),
				}),
				Weighted(WeightOptions{
					Weight: 3,
					Child:  r.renderMemoryAndLossCharts(stats),
				}),
				r.renderSampledTensors(stats),
			},
		})

		var frame bytes.Buffer
		size := terminalArea()

		RenderAt(&frame, dashboard, Point{X: 1, Y: 1}, size)
		_, _ = os.Stdout.Write([]byte("\033[H"))
		_, _ = os.Stdout.Write(frame.Bytes())
	}

	for {
		select {
		case <-stats.TrainingDone:
			render(true)
			return
		case <-ticker.C:
			if stats.Epoch >= stats.NumEpochs {
				render(true)
				return
			}
			render(false)
		}
	}
}

func trainingHeader(stats *shapes.TrainingStats) string {
	header := fmt.Sprintf(
		"Epoch %d/%d | Loss %.6f | Val Loss %.6f | Acc %.2f%%",
		stats.Epoch,
		stats.NumEpochs,
		stats.Loss,
		stats.ValidationLoss,
		stats.Accuracy*100,
	)
	if len(stats.StepLossHistoryX) == 0 {
		return header
	}
	if stats.NumSteps > 0 {
		stepInEpoch := ((stats.Step - 1) % stats.NumSteps) + 1
		return fmt.Sprintf("%s | Step %d/%d Loss %.6f", header, stepInEpoch, stats.NumSteps, stats.StepLoss)
	}
	return fmt.Sprintf("%s | Step %d Loss %.6f", header, stats.Step, stats.StepLoss)
}

func terminalArea() Area {
	width, height, ok := terminalSize(os.Stdout.Fd())
	return resolveTerminalArea(width, height, ok)
}

func resolveTerminalArea(width, height int, ok bool) Area {
	if !ok {
		width = envInt("COLUMNS", 120)
		height = envInt("LINES", 90)
	}

	if width <= 0 {
		width = 120
	}
	if height <= 0 {
		height = 90
	}

	return Area{Width: width, Height: height}
}

func terminalSize(fd uintptr) (width int, height int, ok bool) {
	type winsize struct {
		row    uint16
		col    uint16
		xpixel uint16
		ypixel uint16
	}

	ws := winsize{}
	_, _, errno := syscall.Syscall(
		syscall.SYS_IOCTL,
		fd,
		uintptr(syscall.TIOCGWINSZ),
		uintptr(unsafe.Pointer(&ws)),
	)
	if errno != 0 || ws.col == 0 || ws.row == 0 {
		return 0, 0, false
	}

	return int(ws.col), int(ws.row), true
}

func envInt(key string, fallback int) int {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}

	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}

	return parsed
}
