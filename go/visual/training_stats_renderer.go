package visual

import (
	"bytes"
	"fmt"
	"os"
	"time"

	"github.com/flygerian/shapes"
)

type TrainingStatsRenderer struct {
	ctx shapes.MainContext
}

func (r *TrainingStatsRenderer) SetTrainingContext(ctx shapes.MainContext) {
	r.ctx = ctx

	go r.launchTrainingDashboard()
}

func (r *TrainingStatsRenderer) launchTrainingDashboard() {
	ticker := time.NewTicker(time.Second / 60)
	defer ticker.Stop()
	fmt.Fprint(os.Stdout, "\033[?25l\033[2J\033[H")
	defer fmt.Fprint(os.Stdout, "\033[?25h")

	lastVersion := -1

	stats := r.ctx.TrainingStats()

	render := func(force bool) {

		if len(stats.MemorySampleHistoryX) == 0 || len(stats.LossHistoryX) == 0 {
			return
		}
		if !force && stats.Version == lastVersion {
			return
		}
		lastVersion = stats.Version

		header := fmt.Sprintf("Epoch %d/%d | Loss %.6f", stats.Epoch, stats.NumEpochs, stats.Loss)
		dashboard := Flex(FlexOptions{
			MinHeight: 50,
			Direction: DirectionColumn,
			Children: []Artefact{
				Box(BoxOptions{
					Child: Flex(FlexOptions{
						Direction: DirectionColumn,
						Children: []Artefact{
							Text("Makemore Training"),
							Text(header),
						},
					}),
				}),
				Weighted(WeightOptions{
					Weight: 3,
					Child: Flex(FlexOptions{
						Direction: DirectionColumn,
						Children: []Artefact{
							Flex(FlexOptions{
								Direction: DirectionColumn,
								Children: []Artefact{
									Chart(ChartOptions{
										XAxis: stats.MemorySampleHistoryX,
										YAxis: stats.UsedBlocksHistory,
										Mode:  ChartWindowed,
									}),
									Box(BoxOptions{
										Child: Text("Memory"),
									}),
								},
							}),
							Weighted(WeightOptions{
								Weight: 2,
								Child: Flex(FlexOptions{
									Direction: DirectionColumn,
									Children: []Artefact{
										Chart(ChartOptions{
											XAxis: stats.LossHistoryX,
											YAxis: stats.LossHistory,
											Mode:  ChartFitToViewport,
										}),
										Box(BoxOptions{
											Child: Text("Loss"),
										}),
									},
								}),
							}),
						},
					}),
				}),
			},
		})

		var frame bytes.Buffer

		dashboard.Render(&frame, Bounds{X: 1, Y: 1, Width: 90, Height: 0})
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
