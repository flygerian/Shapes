package visual

import (
	"fmt"
	"io"
	"strings"
	"unicode/utf8"
)

type text struct {
	lines []string
}

var _ Artefact = (*text)(nil)

// Text creates a text artefact.
func Text(content string) Artefact {
	return &text{lines: splitLines(content)}
}

func (t *text) Measure(budget Bounds) Bounds {
	if t == nil || len(t.lines) == 0 {
		return Bounds{Width: 1, Height: 1}
	}
	maxWidth := 1
	for _, line := range t.lines {
		w := utf8.RuneCountInString(line)
		if w > maxWidth {
			maxWidth = w
		}
	}
	if budget.Width > 0 && maxWidth > budget.Width {
		maxWidth = budget.Width
	}

	height := len(t.lines)
	if budget.Height > 0 && height > budget.Height {
		height = budget.Height
	}

	return Bounds{Width: maxWidth, Height: height}
}

// Render draws text lines within bounds, clipping horizontally and vertically.
func (t *text) Render(target io.Writer, bounds Bounds) {
	if t == nil || bounds.Width <= 0 || bounds.Height <= 0 {
		return
	}

	for i := 0; i < bounds.Height && i < len(t.lines); i++ {
		rendered := truncateRunes(t.lines[i], bounds.Width)
		fmt.Fprintf(target, "\033[%d;%dH%s", bounds.Y+i, bounds.X, rendered)
	}
}

func truncateRunes(s string, max int) string {
	if max <= 0 {
		return ""
	}
	if utf8.RuneCountInString(s) <= max {
		return s
	}
	r := []rune(s)
	return string(r[:max])
}

func splitLines(text string) []string {
	if text == "" {
		return []string{""}
	}
	return strings.Split(text, "\n")
}
