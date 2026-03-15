package visual

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"strings"

	shapes "github.com/flygerian/shapes"
)

type imageArtefact struct {
	layoutState
	tensor   shapes.Tensor
	ctx      shapes.Context
	maxWidth int
}

type ImageOptions struct {
	Tensor   shapes.Tensor
	Context  shapes.Context
	MaxWidth int
}

var _ Artefact = (*imageArtefact)(nil)

func Image(options ImageOptions) Artefact {
	return &imageArtefact{
		tensor:   options.Tensor,
		ctx:      options.Context,
		maxWidth: options.MaxWidth,
	}
}

func (i *imageArtefact) Weight() int {
	return 1
}

func (i *imageArtefact) Measure(budget Area) Area {
	if i == nil || i.tensor == nil {
		return Area{Width: 1, Height: 1}
	}

	shape := i.tensor.Shape()
	if len(shape) != 3 {
		return Area{Width: 1, Height: 1}
	}

	height := int(shape[1])
	width := int(shape[2])

	if budget.Width > 0 && width > budget.Width {
		scale := float64(budget.Width) / float64(width)
		width = budget.Width
		height = int(float64(height) * scale)
	}

	if i.maxWidth > 0 && width > i.maxWidth {
		scale := float64(i.maxWidth) / float64(width)
		width = i.maxWidth
		height = int(float64(height) * scale)
	}

	return Area{Width: width, Height: height}
}

func (i *imageArtefact) Render(target io.Writer) {
	frame := layoutOf(i)
	if i == nil || i.tensor == nil || frame.Width <= 0 || frame.Height <= 0 {
		return
	}

	shape := i.tensor.Shape()
	if len(shape) != 3 {
		return
	}

	channels := int(shape[0])
	height := int(shape[1])
	width := int(shape[2])

	img := image.NewRGBA(image.Rect(0, 0, width, height))

	// For uint8 tensors, use direct array access (faster and avoids Get bug)
	if i.tensor.Dtype() == shapes.DtypeU8 {
		vals := i.tensor.Values().([]uint8)
		pixelSize := height * width

		for y := range height {
			for x := range width {
				var r, g, b uint8

				switch channels {
				case 1:
					idx := y*width + x
					r = vals[idx]
					g = vals[idx]
					b = vals[idx]
				case 3:
					r = vals[0*pixelSize+y*width+x]
					g = vals[1*pixelSize+y*width+x]
					b = vals[2*pixelSize+y*width+x]
				case 4:
					r = vals[0*pixelSize+y*width+x]
					g = vals[1*pixelSize+y*width+x]
					b = vals[2*pixelSize+y*width+x]
				default:
					idx := y*width + x
					r = vals[idx]
					g = vals[idx]
					b = vals[idx]
				}

				img.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
			}
		}
	} else {
		// Fallback for other dtypes - use Get (may have bugs)
		for y := range height {
			for x := range width {
				var r, g, b uint8

				switch channels {
				case 1:
					val := i.getPixel(0, y, x)
					r = val
					g = val
					b = val
				case 3:
					r = i.getPixel(0, y, x)
					g = i.getPixel(1, y, x)
					b = i.getPixel(2, y, x)
				case 4:
					r = i.getPixel(0, y, x)
					g = i.getPixel(1, y, x)
					b = i.getPixel(2, y, x)
				default:
					val := i.getPixel(0, y, x)
					r = val
					g = val
					b = val
				}

				img.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
			}
		}
	}

	displayWidth := frame.Width
	displayHeight := frame.Height

	if width != displayWidth || height != displayHeight {
		img = resizeImage(img, displayWidth, displayHeight)
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return
	}

	encoded := base64.StdEncoding.EncodeToString(buf.Bytes())

	renderKittyImage(target, encoded, frame.X, frame.Y, displayWidth, displayHeight)
}

func (i *imageArtefact) getPixel(channel, y, x int) uint8 {
	if i.tensor == nil || i.ctx == nil {
		return 0
	}

	val := i.tensor.Get(i.ctx, uint(channel), uint(y), uint(x)).Item()

	var normalized float32
	switch v := val.(type) {
	case float32:
		normalized = v
	case float64:
		normalized = float32(v)
	case int8:
		// Raw image bytes stored as int8 - treat as unsigned 0-255
		// Values 128-255 would be negative in int8, so convert via uint8
		normalized = float32(uint8(v)) / 255.0
	case uint8:
		normalized = float32(v) / 255.0
	case int:
		normalized = float32(v) / 255.0
	default:
		normalized = 0
	}

	if normalized < 0 {
		normalized = 0
	}
	if normalized > 1 {
		normalized = 1
	}

	return uint8(normalized * 255)
}

func resizeImage(src image.Image, newWidth, newHeight int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))

	bounds := src.Bounds()
	oldWidth := bounds.Dx()
	oldHeight := bounds.Dy()

	if oldWidth <= 0 || oldHeight <= 0 {
		return dst
	}

	xRatio := float64(oldWidth) / float64(newWidth)
	yRatio := float64(oldHeight) / float64(newHeight)

	for y := 0; y < newHeight; y++ {
		for x := 0; x < newWidth; x++ {
			srcX := int(float64(x) * xRatio)
			srcY := int(float64(y) * yRatio)

			if srcX >= oldWidth {
				srcX = oldWidth - 1
			}
			if srcY >= oldHeight {
				srcY = oldHeight - 1
			}

			dst.Set(x, y, src.At(srcX+bounds.Min.X, srcY+bounds.Min.Y))
		}
	}

	return dst
}

func renderKittyImage(w io.Writer, base64Data string, x, y, width, height int) {
	if width <= 0 || height <= 0 {
		return
	}

	chunkSize := 4096

	fmt.Fprintf(w, "\033[%d;%dH", y+1, x+1)

	// c=0,r=0 lets Kitty calculate size from actual PNG pixel dimensions
	fmt.Fprintf(w, "\033_Ga=T,f=100,t=d,c=0,r=0;")

	for i := 0; i < len(base64Data); i += chunkSize {
		end := i + chunkSize
		if end > len(base64Data) {
			end = len(base64Data)
		}
		chunk := base64Data[i:end]
		fmt.Fprint(w, chunk)

		if end < len(base64Data) {
			fmt.Fprint(w, "\033\\\033_G")
		}
	}

	fmt.Fprint(w, "\033\\")
}

func isKittyTerminal() bool {
	term := strings.ToLower(getenv("TERM"))
	kittyTerm := strings.ToLower(getenv("KITTY_TERMINAL"))
	termProgram := strings.ToLower(getenv("TERM_PROGRAM"))

	return strings.Contains(term, "kitty") ||
		kittyTerm != "" ||
		termProgram == "kitty"
}

func getenv(key string) string {
	return os.Getenv(key)
}
