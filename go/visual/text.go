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

// Text creates a text artefact. Each argument is rendered on a new line.
func Text(lines ...string) Artefact {
	flat := make([]string, 0, len(lines))
	for _, line := range lines {
		flat = append(flat, splitLines(line)...)
	}
	return &text{lines: flat}
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
