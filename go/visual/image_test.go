package visual

import (
	"context"
	"image/color"
	"testing"

	shapes "github.com/flygerian/shapes"
)

func TestTensorImagePreservesRGBChannelsForFloatTensor(t *testing.T) {
	ctx := shapes.New(context.Background())

	// CHW layout: red plane, then green plane, then blue plane.
	tensor := shapes.FromFloat32(ctx, shapes.Shape{3, 1, 2}, []float32{
		1.0, 0.0,
		0.0, 1.0,
		0.0, 0.0,
	})

	img := tensorImage(tensor, ctx, 3, 2, 1)

	left := color.RGBAModel.Convert(img.At(0, 0)).(color.RGBA)
	right := color.RGBAModel.Convert(img.At(1, 0)).(color.RGBA)

	if left != (color.RGBA{R: 255, G: 0, B: 0, A: 255}) {
		t.Fatalf("left pixel = %#v, want red", left)
	}
	if right != (color.RGBA{R: 0, G: 255, B: 0, A: 255}) {
		t.Fatalf("right pixel = %#v, want green", right)
	}
}
