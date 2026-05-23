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

type imageLayout int

const (
	imageLayoutCHW imageLayout = iota
	imageLayoutHWC
)

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

	layout, channels, width, height := imageShape(shape)
	_ = layout
	_ = channels

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

	layout, channels, width, height := imageShape(shape)

	img := tensorImage(i.tensor, i.ctx, layout, channels, width, height)

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

func imageShape(shape shapes.Shape) (imageLayout, int, int, int) {
	if len(shape) != 3 {
		return imageLayoutCHW, 0, 0, 0
	}

	if shape[0] == 1 || shape[0] == 3 || shape[0] == 4 {
		return imageLayoutCHW, int(shape[0]), int(shape[2]), int(shape[1])
	}

	return imageLayoutHWC, int(shape[2]), int(shape[1]), int(shape[0])
}

func tensorImage(t shapes.Tensor, ctx shapes.Context, layout imageLayout, channels, width, height int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	if t == nil {
		return img
	}

	pixelSize := height * width
	read := tensorValueReader(t, ctx, layout, pixelSize, channels)
	for y := range height {
		for x := range width {
			idx := y*width + x
			r, g, b, a := tensorPixel(read, channels, idx, y, x)
			img.Set(x, y, color.RGBA{R: r, G: g, B: b, A: a})
		}
	}

	return img
}

func tensorPixel(read func(channel, idx, y, x int) uint8, channels, idx, y, x int) (uint8, uint8, uint8, uint8) {
	switch channels {
	case 1:
		value := read(0, idx, y, x)
		return value, value, value, 255
	case 3:
		return read(0, idx, y, x), read(1, idx, y, x), read(2, idx, y, x), 255
	case 4:
		return read(0, idx, y, x), read(1, idx, y, x), read(2, idx, y, x), read(3, idx, y, x)
	default:
		value := read(0, idx, y, x)
		return value, value, value, 255
	}
}

func tensorValueReader(t shapes.Tensor, ctx shapes.Context, layout imageLayout, pixelSize int, channels int) func(channel, idx, y, x int) uint8 {
	if t == nil {
		return func(channel, idx, y, x int) uint8 { return 0 }
	}

	switch values := t.Values().(type) {
	case []float32:
		if layout == imageLayoutHWC {
			return func(channel, idx, y, x int) uint8 { return normalizedToByte(values[idx*channels+channel]) }
		}
		return func(channel, idx, y, x int) uint8 { return normalizedToByte(values[channel*pixelSize+idx]) }
	case []float64:
		if layout == imageLayoutHWC {
			return func(channel, idx, y, x int) uint8 {
				return normalizedToByte(float32(values[idx*channels+channel]))
			}
		}
		return func(channel, idx, y, x int) uint8 { return normalizedToByte(float32(values[channel*pixelSize+idx])) }
	case []uint8:
		if layout == imageLayoutHWC {
			return func(channel, idx, y, x int) uint8 { return values[idx*channels+channel] }
		}
		return func(channel, idx, y, x int) uint8 { return values[channel*pixelSize+idx] }
	case []int8:
		if layout == imageLayoutHWC {
			return func(channel, idx, y, x int) uint8 { return uint8(values[idx*channels+channel]) }
		}
		return func(channel, idx, y, x int) uint8 { return uint8(values[channel*pixelSize+idx]) }
	default:
		return func(channel, idx, y, x int) uint8 { return getPixelViaTensorGet(t, ctx, layout, channel, y, x) }
	}
}

func getPixelViaTensorGet(t shapes.Tensor, ctx shapes.Context, layout imageLayout, channel, y, x int) uint8 {
	if t == nil || ctx == nil {
		return 0
	}

	var val interface{}
	if layout == imageLayoutHWC {
		val = t.Get(ctx, uint(y), uint(x), uint(channel)).Item()
	} else {
		val = t.Get(ctx, uint(channel), uint(y), uint(x)).Item()
	}
	switch v := val.(type) {
	case float32:
		return normalizedToByte(v)
	case float64:
		return normalizedToByte(float32(v))
	case int8:
		return uint8(v)
	case uint8:
		return v
	case int:
		return uint8(v)
	default:
		return 0
	}
}

func normalizedToByte(v float32) uint8 {
	if v < 0 {
		v = 0
	}
	if v > 1 {
		v = 1
	}
	return uint8(v * 255)
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
